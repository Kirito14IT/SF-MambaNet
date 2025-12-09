import torch
import torch.nn as nn
from loss import batch_episym

from mamba_ssm.modules.mamba_simple import Mamba

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True) #xx[32,1,2000]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k) [32,2000,9] [32,1000,6]

    return idx[:, :, :]

def get_graph_feature(x, k=20, idx=None):
    #x[32,128,2000,1],k=9
    # x[32,128,1000,1],k=6
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points) #x[32,128,2000]
    if idx is None:
        idx_out = knn(x, k=k) #idx_out[32,2000,9]
    else:
        idx_out = idx

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base #idx[32,2000,9] 把32个批次的标号连续了

    idx = idx.view(-1) #idx[32*2000*9] 把32个批次连在一起了 [32*1000*6]

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() #x[32,2000,128]
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) #feature[32,2000,9,128]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) #x[32,2000,9,128]
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous() #feature[32,256,2000,9] 图特征
    return feature

class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv




def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4 [32,1,500,4] logits[32,2,500,1]
    mask = logits[:, 0, :, 0] #[32,500] logits的第一层
    weights = logits[:, 1, :, 0] #[32,500] logits的第二层

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

class DGCNN_MAX_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_MAX_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel*2, self.in_channel, (1, 1)), #[32,128,2000,9]→[32,128,2000,3]
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)), #[32,128,2000,3]→[32,128,2000,1]
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            )

    def forward(self, features):
        #feature[32,128,2000,1]
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out = self.conv(out) #out[32,128,2000,1]
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.unsqueeze(3)
        return out

#############  Mamba_Block  #############

class Mamba_Block(nn.Module):
    def __init__(self,channels,d_state):
        super(Mamba_Block,self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels,d_state=d_state)

    def forward(self,x):
        # x:bcn1
        data = x
        x = self.norm(x.transpose(1,2).squeeze(-1))#bnc
        out = self.mamba(x) #bnc
        out = self.norm(out + data.transpose(1,2).squeeze(-1))
        return out.transpose(1,2).unsqueeze(-1) #bcn1

#############  Mamba_Block  #############



#############  CSSM   #############
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class CSM(nn.Module):
    def __init__(self, channels, points, out_channels = None, d_state=8):
        super(CSM, self).__init__()
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size = 1)

        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps = 1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size = 1),  # b*c*k*1
            )
        # Spatial Aggregation Mamba
        self.norm = nn.LayerNorm(points)
        self.SAM = Mamba(points,d_state) #bck
        self.conv3 = nn.Sequential(
            nn.InstanceNorm2d(out_channels, eps = 1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 1)
        )

    def forward(self, x):
        # print(x.size(0))
        # x:bck1
        out = self.conv1(x).squeeze(-1) #out:bck
        out_n = self.norm(out)
        out = out + self.SAM(out_n)  # out:bck
        out = self.conv3(out.unsqueeze(-1)) #out:bck1
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out #out:bck1


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps = 1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size = 1))

    def forward(self, x):
        #x[32,128,2000,1]
        embed = self.conv(x)  # b*k*n*1,embed:[b,256,2000,1]
        S = torch.softmax(embed, dim = 2).squeeze(3) # b*k*n，计算点属于哪个类别的权重
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3) #out[b,c,k]，将类别的权重乘以特征图，得到新的特征图
        return out #b*c*k*1


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps = 1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size = 1))

    def forward(self, x_up, x_down):
        # x_up: b*c*n*1
        # x_down: b*c*k*1
        embed = self.conv(x_up)  # b*k*n*1
        S = torch.softmax(embed, dim = 1).squeeze(3)  # b*k*n，计算每个特征对每个点的贡献程度
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3) #bcn1，原来的特征乘以特征得分得到新的含有特征得分的特征图
        return out


class CSSM(nn.Module):
    def __init__(self, net_channels, depth = 6, clusters = 64):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = CSM(channels,l2_nums)
        self.up1 = diff_unpool(channels, l2_nums)
        self.output = nn.Conv2d(channels, 1, kernel_size = 1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, kernel_size = 1)

        self.mamba_block = Mamba_Block(channels,8)

    def forward(self, data):
        # data: b*c*n*1
        x1_1 = data
        x1_2 = self.mamba_block(x1_1) #bcn1
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = torch.cat([x1_1, x_up], dim = 1)
        return self.shot_cut(out) + x1_2

#############  CSSM   #############


#############  CFBM  #############

class CFBM(nn.Module):
    def __init__(self, channels,d_state = 16):
        super(CFBM, self).__init__()
        self.channels = channels
        self.cfbm = Mamba_Block(channels,d_state)

    def forward(self, x):
        #x:bcn1
        x_r = x
        x1 = self.cfbm(x)  # bcn1
        x1_f = x1.flip(dims=[2])
        x2 = self.cfbm(x1_f)
        out = x2.flip(dims=[2]) + x_r
        return out  # bcn1

#############  CFBM  #############

class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)), #4或6 → 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.LPFE = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            DGCNN_MAX_Block(self.k_num * 2, self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            CSSM(self.out_channel, clusters=256),
            CSSM(self.out_channel, clusters=256),
            CSSM(self.out_channel, clusters=256),
        )

        self.GCC = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            CFBM(self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.dropout = nn.Dropout(p=0.3)

        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)] #indices[32,1000]剪枝剪掉一半
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices) #y_out 剪枝后保留的标签[32,1000]
            w_out = torch.gather(weights, dim=-1, index=indices) #w_out 剪枝后保留的w0[32,1000]
        indices = indices.view(B, 1, -1, 1) #indices[32,1,1000,1]

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) #x_out 剪枝后保留的x[32,1,1000,4]
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) #x_out 剪枝后保留的x[32,1,500,4]
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1)) #feature_out 剪枝后保留的features[32,128,500,1]
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        # x[32,1,2000,4],y[32,2000]
        # x_[32,1,1000,6],y1[32,1000]
        B, _, N , _ = x.size()
        out = x.transpose(1, 3).contiguous() #contiguous断开out与x的依赖关系。out[32,4或6,2000,1]
        out = self.conv(out) #out[32,128,2000,1]

        out = self.LPFE(out) #out[32,128,2000,1] [32,128,1000,1]
        out = self.dropout(out)
        w0 = self.linear_0(out).view(B, -1) #w0[32,2000]

        out_g = self.GCC(out) #out_g[32,128,2000,1]
        out_g = self.dropout(out_g)
        out = out_g + out

        out = self.embed_1(out)
        w1 = self.linear_1(out).view(B, -1) #w1[32,2000]

        if self.predict == False: #剪枝，不预测
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) #w1排序,w1_ds[32,2000],indices[32,2000]是索引
            w1_ds = w1_ds[:, :int(N*self.sr)] #w1_ds[32,1000]剪枝？剪掉一半 self.sr=0.5
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            #x_ds[32,1,1000,4],y_ds[32,1000],w0_ds[32,1000],ds：剪枝后？
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else: #剪枝，出预测结果
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) #w1排序,w1_ds[32,1000],indices[32,1000]是索引
            w1_ds = w1_ds[:, :int(N*self.sr)] #w1_ds[32,500]剪枝？剪掉一半 self.sr=0.5
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            # x_ds[32,1,500,4],y_ds[32,500],w0_ds[32,500],out[32,128,500,1]也是剪枝后,ds：剪枝后？
            out = self.embed_2(out)
            w2 = self.linear_2(out) #[32,2,500,1]
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat

class MatchMamba(nn.Module):
    def __init__(self, config):
        super(MatchMamba, self).__init__()

        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=config.sr)#sampling_rate=0.5
        self.ds_1 = DS_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y):
        #x[32,1,2000,4],y[32,2000]
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0 = self.ds_0(x, y) # 返回的是x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1) #变成0到1的权重[32,1,1000,1]
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1) #变成0到1的权重[32,1,1000,1]
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1) #x_[32,1,1000,6] 剪枝后的特征并带上了权重信息

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1) #x_[32,1,1000,6],y1[32,1000]

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat) #y_hat对称极线距离
        #print(y_hat)
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat

