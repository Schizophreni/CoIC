from models.CoIM import BaseConv, TConvTranspose
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluation import psnr as compare_psnr
from evaluation import rgb_to_y


class TConv(nn.Module):
    """
    Conv transformation
    """

    def __init__(self, in_channels, out_channels, dim_in=32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 use_bias=True, return_temp=True):
        super(TConv, self).__init__()
        ## base conv
        self.kernel_size = kernel_size
        self.base_conv = BaseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, use_bias=use_bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.groups = groups
        self.return_temp = return_temp  # whether to return temperature in forward pass
        if self.groups > 1:
            self.g2pa_wgen = nn.Sequential(
                nn.Linear(dim_in + in_channels, out_channels // 16),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels // 16, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, in_channels // groups + out_channels+kernel_size**2),
            )   
        else:
            self.g2pa_wgen = nn.Sequential(
                nn.Linear(dim_in + in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, in_channels // groups + out_channels+kernel_size**2),
            )    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, base_x, tran_x=None, mode='normal'):
        if mode == 'normal' or self.kernel_size == 1:
            base_out = self.base_conv(base_x)
            return base_out, None, list()
        else:
            b_size = tran_x.shape[0]
            in_x = torch.mean(base_x, dim=[-2, -1])
            att = self.g2pa_wgen(torch.cat([tran_x, in_x], dim=-1))  # [B, D]
            w1, w2, w3 = att[:, :self.out_channels], att[:, self.out_channels:self.out_channels+self.in_channels // self.groups], att[:, self.in_channels // self.groups+self.out_channels:]
            att = w1.unsqueeze(1) + w2.unsqueeze(-1)  # [B, C_o, C_i // G]
            att = torch.sigmoid(att)
            if self.return_temp:
                temp = 1.0 / att.detach().cpu()
                temp = temp / ((w3 - w3.min()).detach().cpu().mean() + 1e-8)
            else:
                temp = None
            att = att.unsqueeze(-1) * w3.view(b_size, 1, 1, -1)          
            if self.kernel_size > 1:
                att = torch.softmax(att, dim=-1)
            else:
                att = torch.sigmoid(att)
            att = att.view(b_size, self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
            att = att.contiguous()
            att = att * (self.kernel_size**2)
            base_out = self.base_conv.forward(base_x, w=att)
            return base_out, tran_x, temp

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1, dim_in=128, return_temp=True):
    return TConv(in_channels, out_channels, dim_in, kernel_size, stride=stride, padding=(kernel_size//2), use_bias=bias, return_temp=return_temp)


def conv_down(in_chn, out_chn, bias=False, dim_in=128):
    layer = TConv(in_chn, out_chn, dim_in, kernel_size=4, stride=2, padding=1, use_bias=bias)
    return layer

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True, dim_in=128, return_temp=True):
    return TConv(in_channels, out_channels, dim_in, kernel_size=kernel_size, padding=(kernel_size//2), stride=stride, use_bias=bias, return_temp=return_temp)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1, dim_in=128, return_temp=True):

        super(ResBlock, self).__init__()
        m = []
        self.body = nn.Sequential(
            conv(n_feats, 64, kernel_size, bias=bias, dim_in=dim_in, return_temp=return_temp),
            nn.PReLU(),
            conv(64, n_feats, kernel_size, bias=bias, dim_in=dim_in, return_temp=return_temp)
        )
        self.res_scale = res_scale

    def forward(self, x, tran_x, mode='normal'):
        res, _, t1 = self.body[0](x, tran_x, mode)
        res = self.body[1](res)
        res, _, t2 = self.body[2](res, tran_x, mode)
        # res = res.mul(self.res_scale)
        res += x
        return res, [t1, t2]

##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, dim_in, return_temp=True):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dim_in=dim_in, return_temp=return_temp))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dim_in=dim_in, return_temp=return_temp))

        self.CA = CALayer(n_feat, reduction, bias=bias, dim_in=dim_in)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, tran_x, mode='normal'):
        res, _, t1 = self.body[0](x, tran_x, mode)
        res = self.body[1](res)
        res, _, t2 = self.body[2](res, tran_x, mode)
        res = self.CA(res)
        res += x
        return res, [t1, t2]
    
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False, dim_in=128):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Sigmoid()
        )
    
    # def forward(self, x, tran_x, mode='normal'):
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y 

    
##########################################################################
## Compute inter-stage features
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, dim_in, return_temp=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias, dim_in=dim_in, return_temp=return_temp)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias, dim_in=dim_in, return_temp=return_temp)

    def forward(self, x, x_img, tran_x, mode='normal'):
        x1, _, t1 = self.conv1(x, tran_x, mode)
        img, _, t2 = self.conv2(x, tran_x, mode)
        img = img + x_img
        x1 = x1 + x
        return x1, img, [t1, t2]

class mergeblock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, subspace_dim=16, dim_in=128):
        super(mergeblock, self).__init__()
        self.conv_block = conv(n_feat * 2, n_feat, kernel_size, bias=bias, dim_in=dim_in)
        self.num_subspace = subspace_dim
        self.subnet = conv(n_feat * 2, self.num_subspace, kernel_size, bias=bias, dim_in=dim_in)

    def forward(self, x, bridge, tran_x, mode):
        out = torch.cat([x, bridge], 1)
        b_, c_, h_, w_ = bridge.shape
        sub = self.subnet(out, tran_x, mode)[0]
        # sub = self.subnet(out, None, "normal")[0]
        V_t = sub.view(b_, self.num_subspace, h_*w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        V = V_t.permute(0, 2, 1)
        mat = torch.matmul(V_t, V)
        mat_inv = torch.inverse(mat)
        # print(mat.max(), mat.min(), mat.std())
        project_mat = torch.matmul(mat_inv, V_t)
        bridge_ = bridge.view(b_, c_, h_*w_)
        project_feature = torch.matmul(project_mat, bridge_.permute(0, 2, 1))
        bridge = torch.matmul(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)
        out = torch.cat([x, bridge], 1)
        out = self.conv_block(out, tran_x, mode)[0]
        return out+x

##########################################################################
## U-Net    
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff,depth=5, dim_in=128, return_temp=True):
        super(Encoder, self).__init__()
        self.body=nn.ModuleList()#[]
        self.depth=depth
        for i in range(depth-1):
            self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*i, out_size=n_feat+scale_unetfeats*(i+1), downsample=True, relu_slope=0.2, use_csff=csff, use_HIN=True, dim_in=dim_in))
        self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*(depth-1), out_size=n_feat+scale_unetfeats*(depth-1), downsample=False, relu_slope=0.2, use_csff=csff, use_HIN=True, dim_in=dim_in))

    def forward(self, x, encoder_outs=None, decoder_outs=None, tran_x=None, mode='normal'):
        res, temps = [], []
        if encoder_outs is not None and decoder_outs is not None:
            for i,down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up, t = down(x,encoder_outs[i], decoder_outs[-i-1],tran_x, mode)
                    res.append(x_up)  # which formulates encoder out
                    temps.append(t)
                else:
                    x, t = down(x, tran_x=tran_x, mode=mode)
                    temps.append(t)
        else:
            for i,down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up, t = down(x, tran_x=tran_x, mode=mode)
                    res.append(x_up)  # which is the first stage with no previous results
                    temps.append(t)
                else:
                    x, t = down(x, tran_x=tran_x, mode=mode)
                    temps.append(t)
        return res, x, temps  # res indicates the encoder_outs for encoder in next stage
    
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False, dim_in=128, 
                 return_temp=True):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = TConv(in_size, out_size, dim_in, kernel_size=1, stride=1, padding=0, return_temp=return_temp)
        self.use_csff = use_csff

        self.conv_1 = TConv(in_size, out_size, dim_in, 3, 1, 1, return_temp=return_temp)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = TConv(out_size, out_size, dim_in, 3, 1, 1, return_temp=return_temp)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = TConv(out_size, out_size, dim_in, 3, 1, 1, return_temp=return_temp)
            self.csff_dec = TConv(in_size, out_size, dim_in, 3, 1, 1, return_temp=return_temp)
            self.phi = TConv(out_size, out_size, dim_in, 3, 1, 1, return_temp=return_temp)
            self.gamma = TConv(out_size, out_size, dim_in, 3, 1, 1, return_temp=return_temp)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False, dim_in=dim_in)

    def forward(self, x, enc=None, dec=None, tran_x=None, mode='normal'):
        out, _, t1 = self.conv_1(x, tran_x, mode)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out, _, t2 = self.conv_2(out, tran_x, mode)
        out = self.relu_2(out)
        r_out, _, t3 = self.identity(x, tran_x, mode)
        out = out + r_out

        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(enc, tran_x, mode)[0] + self.csff_dec(dec, tran_x, mode)[0], 0.1, inplace=True)
            out = out*F.sigmoid(self.phi(skip_, tran_x, mode)[0]) + self.gamma(skip_, tran_x, mode)[0] + out
        if self.downsample:
            out_down = self.downsample(out, tran_x, mode)[0]
            return out_down, out, [t1]
        else:
            return out, [t1]

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope, dim_in=128, return_temp=True):
        super(UNetUpBlock, self).__init__()
        self.up = TConvTranspose(in_size, out_size, dim_in, kernel_size=2, stride=2, padding=0, use_bias=True)
        self.conv_block = UNetConvBlock(out_size*2, out_size, False, relu_slope, dim_in=dim_in, return_temp=True)

    def forward(self, x, bridge, tran_x, mode='normal'):
        up = self.up(x, tran_x, mode)[0]
        h, w = bridge.shape[2], bridge.shape[3]
        up = up[:, :, :h, :w]
        out = torch.cat([up, bridge], 1)
        out, temp = self.conv_block(out, tran_x=tran_x, mode=mode)
        return out, temp

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=5,dim_in=128, return_temp=True):
        super(Decoder, self).__init__()
        
        self.body=nn.ModuleList()
        self.skip_conv=nn.ModuleList()#[]
        for i in range(depth-1):
            self.body.append(UNetUpBlock(in_size=n_feat+scale_unetfeats*(depth-i-1), out_size=n_feat+scale_unetfeats*(depth-i-2), relu_slope=0.2,dim_in=dim_in, return_temp=return_temp))
            self.skip_conv.append(TConv(n_feat+scale_unetfeats*(depth-i-1), n_feat+scale_unetfeats*(depth-i-2), dim_in, 3, 1, 1, return_temp=return_temp))
            
    def forward(self, x, bridges, tran_x=None, mode="normal"):
        res=[]
        temps = []
        for i,up in enumerate(self.body):
            x, t = up(x,self.skip_conv[i](bridges[-i-1], tran_x, mode)[0], tran_x=tran_x, mode=mode)
            res.append(x)
            temps.append(t)
        return res, temps

##########################################################################
## One Stage
class Basic_block(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False, dim_in=128,
                 return_temp=True):
        super(Basic_block, self).__init__()
        act = nn.PReLU()
        self.phi_1 = ResBlock(default_conv, 3, 3, dim_in=dim_in)
        self.phit_1 = ResBlock(default_conv, 3, 3, dim_in=dim_in)
        self.shallow_feat2 = nn.Sequential(TConv(3, n_feat, dim_in, kernel_size, use_bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act, dim_in=dim_in))
        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True, dim_in=dim_in, return_temp=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, dim_in=dim_in, return_temp=True)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias, dim_in=dim_in, return_temp=return_temp)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        # self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias, dim_in=dim_in)
        
        self.merge12=mergeblock(n_feat,3,True, dim_in=dim_in)
    
    def forward(self, img,stage1_img,feat1,res1,x2_samfeats, tran_x, mode='normal'):
        ## GDM
        enc_temps, dec_temps = [], []
        phixsy_2 = self.phi_1(stage1_img, tran_x, mode)[0] - img
        x2_img = stage1_img - self.r1*self.phit_1(phixsy_2, tran_x, mode)[0]
        ## PMM
        x2 = self.shallow_feat2[0](x2_img, tran_x, mode)[0]
        x2 = self.shallow_feat2[1](x2, tran_x, mode)[0]
        x2_cat = self.merge12(x2, x2_samfeats, tran_x, mode)
        feat2,feat_fin2, t = self.stage2_encoder.forward(x2_cat, feat1, res1, tran_x=tran_x, mode=mode)
        enc_temps.extend(t)
        res2, t = self.stage2_decoder.forward(feat_fin2, feat2, tran_x=tran_x, mode=mode)
        dec_temps.extend(t)
        x3_samfeats, stage2_img, _ = self.sam23.forward(res2[-1], x2_img, tran_x=tran_x, mode=mode)
        return x3_samfeats, stage2_img, feat2, res2, enc_temps, dec_temps

##########################################################################
## DGUNet_plus
class DGUNet(nn.Module):
    # def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False, depth=5):
    def __init__(self, opt=None):
        super(DGUNet, self).__init__()
        in_c, out_c = 3, 3
        n_feat, scale_unetfeats, scale_orsnetfeats = 40, 20, 16
        num_cab, kernel_size, reduction = 8, 3, 4
        bias = False
        dim_in = 128

        act = nn.PReLU()
        self.depth=5
        self.basic=Basic_block(in_c, out_c, n_feat, scale_unetfeats, scale_orsnetfeats, num_cab, kernel_size, reduction, bias, dim_in=dim_in)
        self.shallow_feat1 = nn.Sequential(TConv(3, n_feat, dim_in, kernel_size, use_bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act, dim_in=dim_in))
        self.shallow_feat7 = nn.Sequential(TConv(3, n_feat, dim_in, kernel_size, use_bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act, dim_in=dim_in))

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=False, dim_in=dim_in)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, dim_in=dim_in)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias, dim_in=dim_in)
        
        self.phi_0 = ResBlock(default_conv,3,3, dim_in=dim_in)
        self.phit_0 = ResBlock(default_conv,3,3, dim_in=dim_in)
        self.phi_6 = ResBlock(default_conv,3,3, dim_in=dim_in)
        self.phit_6 = ResBlock(default_conv,3,3, dim_in=dim_in)
        self.r0 = nn.Parameter(torch.Tensor([0.5]))
        self.r6 = nn.Parameter(torch.Tensor([0.5]))

        self.concat67 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias, dim_in=dim_in)
        self.tail = conv(n_feat + scale_orsnetfeats, 3, kernel_size, bias=bias, dim_in=dim_in)

    def forward(self, img, tran_x, mode='normal'):
        res=[]
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## GDM
        # pdb.set_trace()
        temps = []
        names = []
        phixsy_1, t = self.phi_0(img, tran_x, mode)
        temps.append(t)
        names.append("A estimate")
        phixsy_1 = phixsy_1 - img
        x1_img = img - self.r0*self.phit_0(phixsy_1, tran_x, mode)[0]
        _, t = self.phit_0(phixsy_1, tran_x, mode)
        ## PMM
        x1, _, t = self.shallow_feat1[0](x1_img, tran_x, mode)
        # temps.append(t)
        # names.append("PMM")
        x1 = self.shallow_feat1[1](x1, tran_x, mode)[0]

        feat1,feat_fin1, t = self.stage1_encoder.forward(x1, tran_x=tran_x, mode=mode)
        temps.append(t)
        names.append("stage_1 encoder")
        res1, t = self.stage1_decoder.forward(feat_fin1, feat1, tran_x=tran_x, mode=mode)
        temps.append(t)
        names.append("stage_1 decoder")
        x2_samfeats, stage1_img, _ = self.sam12(res1[-1], x1_img, tran_x=tran_x, mode=mode)
        res.append(stage1_img)

        ##-------------------------------------------
        ##-------------- Stage 2-6 ---------------------
        ##-------------------------------------------
        for dep in range(self.depth):
            x2_samfeats, stage1_img, feat1, res1, enc_t, dec_t = self.basic(img,stage1_img,feat1,res1,x2_samfeats, tran_x=tran_x, mode=mode)
            res.append(stage1_img)
            if dep == 0:
                temps.append(enc_t)
                names.append("stage middle encoder")
                temps.append(dec_t)
                names.append("stage middle decoder")
        ##-------------------------------------------
        ##-------------- Stage 7---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_7 = self.phi_6(stage1_img, tran_x, mode)[0] - img
        x7_img = stage1_img - self.r6*self.phit_6(phixsy_7, tran_x, mode)[0]
        x7 = self.shallow_feat7[0](x7_img, tran_x, mode)[0]
        x7 = self.shallow_feat7[1](x7, tran_x, mode)[0]
        ## PMM
        x7_cat = self.concat67(torch.cat([x7, x2_samfeats], 1), tran_x, mode)[0]
        stage7_img = self.tail(x7_cat, tran_x, mode)[0]+ img
        res.append(stage7_img)

        return res[::-1], temps, names

if __name__ == "__main__":
    from models.contrastive import MoCo
    from PIL import Image
    import torchvision.transforms as transforms
    from tqdm import tqdm

    transform = transforms.Compose([
        transforms.CenterCrop(size=(256, 256)),
        transforms.ToTensor(),
    ])

    net = DGUNet().cuda()
    encoder = MoCo(3, 32, dim=128).cuda()
    # load ckp
    net_ckp = "logs/DGUNet-H8L1214-CoIC-rerun_DGUNet_128d_0.2contra/net_epoch_50.pth"
    feat_ckp = "logs/DGUNet-H8L1214-CoIC-rerun_DGUNet_128d_0.2contra/feat_epoch_50.pth"
    net.load_state_dict(torch.load(net_ckp, map_location="cuda"))
    encoder.load_state_dict(torch.load(feat_ckp, map_location="cuda"))
    # load img
    all_temperatures = torch.zeros(100, 16).float()
    for idx in tqdm(range(100), ncols=60):
        # img = Image.open("/home/rw/Public/datasets/derain/Rain200H/test/rain/X2/norain-%dx2.png" % (idx+1))
        # gt = Image.open("/home/rw/Public/datasets/derain/Rain200H/test/norain/norain-%d.png" % (idx+1))
        img = Image.open("/home/rw/Public/datasets/derain/Rain800/test/rain/rain-%03d.png" % (idx+1))
        gt = Image.open("/home/rw/Public/datasets/derain/Rain800/test/norain/norain-%03d.png" % (idx+1))
        img = transform(img).unsqueeze(0)
        gt = transform(gt).unsqueeze(0)
        x = img
        with torch.no_grad():
            z, _ = encoder.encoder_q(x.cuda())
            outs, temps, names = net(x.cuda(), z, 'tran')
        layer_cnt = 0
        for ts, n in zip(temps, names):
            for t in ts:
                if isinstance(t, list):
                    n, v = 0, 0
                    for q in t:
                        n += 1
                        v += torch.log(q).mean()
                    all_temperatures[idx, layer_cnt] = v / n
                else:
                    all_temperatures[idx, layer_cnt] = torch.log(t).mean()
                layer_cnt += 1
    print(all_temperatures.mean(dim=0), all_temperatures.std(dim=0))

