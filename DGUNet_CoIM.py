import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CoIM import TConv, TConvTranspose


print("utilize DGUNet with IPG in old mode. ")
##########################################################################
# Basic modules

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1, dim_in=128):
    return TConv(in_channels, out_channels, dim_in, kernel_size, stride=stride, padding=(kernel_size//2), use_bias=bias)


def conv_down(in_chn, out_chn, bias=False, dim_in=128):
    layer = TConv(in_chn, out_chn, dim_in, kernel_size=4, stride=2, padding=1, use_bias=bias)
    return layer

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True, dim_in=128):
    return TConv(in_channels, out_channels, dim_in, kernel_size=kernel_size, padding=(kernel_size//2), stride=stride, use_bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1, dim_in=128):

        super(ResBlock, self).__init__()
        m = []
        self.body = nn.Sequential(
            conv(n_feats, 64, kernel_size, bias=bias, dim_in=dim_in),
            nn.PReLU(),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            conv(64, n_feats, kernel_size, bias=bias, dim_in=dim_in)
        )
        self.res_scale = res_scale

    def forward(self, x, tran_x, mode='normal'):
        res, _ = self.body[0](x, tran_x, mode)
        res = self.body[1](res)
        res, _ = self.body[2](res, tran_x, mode)
        # res = res.mul(self.res_scale)
        res += x

        return res

##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, dim_in):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dim_in=dim_in))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dim_in=dim_in))

        self.CA = CALayer(n_feat, reduction, bias=bias, dim_in=dim_in)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, tran_x, mode='normal'):
        res, _ = self.body[0](x, tran_x, mode)
        res = self.body[1](res)
        res, _ = self.body[2](res, tran_x, mode)
        # res = self.CA(res, tran_x, mode)
        res = self.CA(res)
        res += x
        return res
    
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
    def __init__(self, n_feat, kernel_size, bias, dim_in):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias, dim_in=dim_in)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias, dim_in=dim_in)

    def forward(self, x, x_img, tran_x, mode='normal'):
        x1, _ = self.conv1(x, tran_x, mode)
        img = self.conv2(x, tran_x, mode)[0] + x_img
        x1 = x1 + x
        return x1, img


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
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff,depth=5, dim_in=128):
        super(Encoder, self).__init__()
        self.body=nn.ModuleList()#[]
        self.depth=depth
        for i in range(depth-1):
            self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*i, out_size=n_feat+scale_unetfeats*(i+1), downsample=True, relu_slope=0.2, use_csff=csff, use_HIN=True, dim_in=dim_in))
        self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*(depth-1), out_size=n_feat+scale_unetfeats*(depth-1), downsample=False, relu_slope=0.2, use_csff=csff, use_HIN=True, dim_in=dim_in))

    def forward(self, x, encoder_outs=None, decoder_outs=None, tran_x=None, mode='normal'):
        res=[]
        if encoder_outs is not None and decoder_outs is not None:
            for i,down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up = down(x,encoder_outs[i],decoder_outs[-i-1],tran_x, mode)
                    res.append(x_up)  # which formulates encoder out
                else:
                    x = down(x, tran_x=tran_x, mode=mode)
        else:
            for i,down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up = down(x, tran_x=tran_x, mode=mode)
                    res.append(x_up)  # which is the first stage with no previous results
                else:
                    x = down(x, tran_x=tran_x, mode=mode)
        return res,x  # res indicates the encoder_outs for encoder in next stage
    
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False, dim_in=128):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = TConv(in_size, out_size, dim_in, kernel_size=1, stride=1, padding=0)
        self.use_csff = use_csff

        self.conv_1 = TConv(in_size, out_size, dim_in, 3, 1, 1)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = TConv(out_size, out_size, dim_in, 3, 1, 1)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = TConv(out_size, out_size, dim_in, 3, 1, 1)
            self.csff_dec = TConv(in_size, out_size, dim_in, 3, 1, 1)
            self.phi = TConv(out_size, out_size, dim_in, 3, 1, 1)
            self.gamma = TConv(out_size, out_size, dim_in, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False, dim_in=dim_in)

    def forward(self, x, enc=None, dec=None, tran_x=None, mode='normal'):
        out = self.conv_1(x, tran_x, mode)[0]

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out, tran_x, mode)[0])
        out += self.identity(x, tran_x, mode)[0]
        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(enc, tran_x, mode)[0] + self.csff_dec(dec, tran_x, mode)[0], 0.1, inplace=True)
            out = out*F.sigmoid(self.phi(skip_, tran_x, mode)[0]) + self.gamma(skip_, tran_x, mode)[0] + out
        if self.downsample:
            out_down = self.downsample(out, tran_x, mode)[0]
            return out_down, out
        else:
            return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope, dim_in=128):
        super(UNetUpBlock, self).__init__()
        self.up = TConvTranspose(in_size, out_size, dim_in, kernel_size=2, stride=2, padding=0, use_bias=True)
        self.conv_block = UNetConvBlock(out_size*2, out_size, False, relu_slope, dim_in=dim_in)

    def forward(self, x, bridge, tran_x, mode='normal'):
        up = self.up(x, tran_x, mode)[0]
        # h, w = bridge.shape[2], bridge.shape[3]
        # up = up[:, :, :h, :w]
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out, tran_x=tran_x, mode=mode)
        return out

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=5,dim_in=128):
        super(Decoder, self).__init__()
        
        self.body=nn.ModuleList()
        self.skip_conv=nn.ModuleList()#[]
        for i in range(depth-1):
            self.body.append(UNetUpBlock(in_size=n_feat+scale_unetfeats*(depth-i-1), out_size=n_feat+scale_unetfeats*(depth-i-2), relu_slope=0.2,dim_in=dim_in))
            self.skip_conv.append(TConv(n_feat+scale_unetfeats*(depth-i-1), n_feat+scale_unetfeats*(depth-i-2), dim_in, 3, 1, 1))
            
    def forward(self, x, bridges, tran_x=None, mode="normal"):
        res=[]
        for i,up in enumerate(self.body):
            x=up(x,self.skip_conv[i](bridges[-i-1], tran_x, mode)[0], tran_x=tran_x, mode=mode)
            res.append(x)

        return res

##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor, dim_in=128):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, 1, 0, bias=False))
                                  # TConv(in_channels, in_channels+s_factor, dim_in, 1, 1, 0, use_bias=False))

    def forward(self, x, tran_x, mode):
        # x = self.down[0](x)
        # x = self.down[1](x, tran_x, mode)[0]
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, dim_in=128):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x, tran_x, mode='normal'):
        x = self.up(x)
        return x


##########################################################################
## One Stage
class Basic_block(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False, dim_in=128):
        super(Basic_block, self).__init__()
        act = nn.PReLU()
        self.phi_1 = ResBlock(default_conv, 3, 3, dim_in=dim_in)
        self.phit_1 = ResBlock(default_conv, 3, 3, dim_in=dim_in)
        self.shallow_feat2 = nn.Sequential(TConv(3, n_feat, dim_in, kernel_size, use_bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act, dim_in=dim_in))
        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True, dim_in=dim_in)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, dim_in=dim_in)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias, dim_in=dim_in)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        # self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias, dim_in=dim_in)
        
        self.merge12=mergeblock(n_feat,3,True, dim_in=dim_in)
    
    def forward(self, img,stage1_img,feat1,res1,x2_samfeats, tran_x, mode='normal'):
        ## GDM
        phixsy_2 = self.phi_1(stage1_img, tran_x, mode) - img
        x2_img = stage1_img - self.r1*self.phit_1(phixsy_2, tran_x, mode)
        ## PMM
        x2 = self.shallow_feat2[0](x2_img, tran_x, mode)[0]
        x2 = self.shallow_feat2[1](x2, tran_x, mode)
        x2_cat = self.merge12(x2, x2_samfeats, tran_x, mode)
        feat2,feat_fin2 = self.stage2_encoder.forward(x2_cat, feat1, res1, tran_x=tran_x, mode=mode)
        res2 = self.stage2_decoder.forward(feat_fin2, feat2, tran_x=tran_x, mode=mode)
        x3_samfeats, stage2_img = self.sam23.forward(res2[-1], x2_img, tran_x=tran_x, mode=mode)
        return x3_samfeats, stage2_img, feat2, res2

##########################################################################
## DGUNet_plus
class DGUNet(nn.Module):
    # def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False, depth=5):
    def __init__(self, opt):
        super(DGUNet, self).__init__()
        in_c, out_c = 3, 3
        n_feat, scale_unetfeats, scale_orsnetfeats = 40, 20, 16
        num_cab, kernel_size, reduction = 8, 3, 4
        bias = False
        dim_in = opt.dim_in


        act = nn.PReLU()
        self.depth=opt.depth
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
        phixsy_1 = self.phi_0(img, tran_x, mode) - img
        x1_img = img - self.r0*self.phit_0(phixsy_1, tran_x, mode)
        ## PMM
        x1 = self.shallow_feat1[0](x1_img, tran_x, mode)[0]
        x1 = self.shallow_feat1[1](x1, tran_x, mode)

        feat1,feat_fin1 = self.stage1_encoder.forward(x1, tran_x=tran_x, mode=mode)
        res1 = self.stage1_decoder.forward(feat_fin1, feat1, tran_x=tran_x, mode=mode)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img, tran_x=tran_x, mode=mode)
        res.append(stage1_img)

        ##-------------------------------------------
        ##-------------- Stage 2-6 ---------------------
        ##-------------------------------------------
        for _ in range(self.depth):
            x2_samfeats, stage1_img, feat1, res1 = self.basic(img,stage1_img,feat1,res1,x2_samfeats, tran_x=tran_x, mode=mode)
            res.append(stage1_img)

        ##-------------------------------------------
        ##-------------- Stage 7---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_7 = self.phi_6(stage1_img, tran_x, mode) - img
        x7_img = stage1_img - self.r6*self.phit_6(phixsy_7, tran_x, mode)
        x7 = self.shallow_feat7[0](x7_img, tran_x, mode)[0]
        x7 = self.shallow_feat7[1](x7, tran_x, mode)
        ## PMM
        x7_cat = self.concat67(torch.cat([x7, x2_samfeats], 1), tran_x, mode)[0]
        stage7_img = self.tail(x7_cat, tran_x, mode)[0]+ img
        res.append(stage7_img)

        return res[::-1]


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.append('..')
    from utils import print_network
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.depth, opt.dim_in = 5, 128
    net = DGUNet(opt)
    # print_network(net)
    x = torch.rand(1, 3, 96, 96)
    outs = net(x, None, 'normal')[0]
    print(outs[0].shape)

    base_params = 0
    tran_param = 0
    for name, param in net.named_parameters():
        # if param.requires_grad:
        if "g2pa" in name:
            tran_param += param.numel()
            param.requires_grad_(False)
            if "scale" in name:
                param.fill_(1.0)
            else:
                param.fill_(0.)
        else:
            base_params += param.numel()
            param.requires_grad_(False)
            if "weight" in name:
                param.fill_(1e-3)
            else:
                param.fill_(1e-2)
    print('Total number of parameters: [%d/%d]' % (base_params, tran_param))
    z = torch.zeros(1, 128)
    torch.manual_seed(1234)
    x = torch.rand(1, 3, 128, 128)
    out = net(x, z, 'tran')[0]
    print(out.shape, out.mean(), out.std(), out.max(), out.min())
