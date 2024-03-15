"""
Context-based Instance-specific Modulation
"""
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.backends import cudnn

print("utilize CoIM in simple | parallel mode (feature-aware multiplication) !")

class BaseConv(nn.Module):
    """
    base convolution module
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 use_bias=True):
        super(BaseConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = int(stride)
        self.padding = int(padding)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.dilation = dilation

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=use_bias)

    def forward(self, x, w=None):
        """
        forward using w, we do not generate biases
        w of shape [batch_size, C_o, C_i // G, f, f]
        """
        if w is None:
            out = self.conv(x)
            return out
        else:
            b_size, _, height, width = x.shape
            x = x.view(1, -1, height, width)  # [1, bc, h, w]
            weight = self.conv.weight.unsqueeze(0) * w  # [b, C_o, C_i // G, f, f]
            weight = weight.view(-1, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
            # use group conv to parallel
            if self.use_bias:
                bias = self.conv.bias.repeat(b_size)
            else: bias=None
            out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, 
                           groups=b_size * self.groups)
            height, width = out.shape[2:]
            out = out.view(b_size, self.out_channels, height, width)
            return out

class BaseConvTranspose(nn.Module):
    """
    base convolution module
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 use_bias=True):
        super(BaseConvTranspose, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = int(stride)
        self.padding = int(padding)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.dilation = dilation

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, 
                                       groups=1, bias=use_bias)

    def forward(self, x, w=None, b=None):
        """
        forward using w and b
        w = self.weight*w
        b = self.bias + b
        """
        if w is None and b is None:
            out = self.conv(x)
            return out
        else:
            b_size, _, height, width = x.shape
            x = x.view(1, -1, height, width)  # [1, bc, h, w]
            weight = self.conv.weight.unsqueeze(0) * w  # [b, C_o, C_i, f, f]
            weight = weight.view(-1, self.out_channels, self.kernel_size, self.kernel_size)
            if self.use_bias:
                bias = self.conv.bias.unsqueeze(0) * torch.ones(b_size, 1).to(x.device)
                bias = bias.view(-1)
            else: bias = None
            out = F.conv_transpose2d(input=x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, 
                           groups=b_size)
            height, width = out.shape[2:]
            out = out.view(b_size, self.out_channels, height, width)
            return out

class TConv(nn.Module):
    """
    Conv transformation
    """

    def __init__(self, in_channels, out_channels, dim_in=32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 use_bias=True):
        super(TConv, self).__init__()
        ## base conv
        self.kernel_size = kernel_size
        self.base_conv = BaseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, use_bias=use_bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.groups = groups
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
            return base_out, None
        else:
            b_size = tran_x.shape[0]
            in_x = torch.mean(base_x, dim=[-2, -1])
            att = self.g2pa_wgen(torch.cat([tran_x, in_x], dim=-1))  # [B, D]
            w1, w2, w3 = att[:, :self.out_channels], att[:, self.out_channels:self.out_channels+self.in_channels // self.groups], att[:, self.in_channels // self.groups+self.out_channels:]
            att = w1.unsqueeze(1) + w2.unsqueeze(-1)  # [B, C_o, C_i // G]
            att = torch.sigmoid(att)
            att = att.unsqueeze(-1) * w3.view(b_size, 1, 1, -1) 
            # print(att.min(), att.max(), att.std())           
            if self.kernel_size > 1:
                att = torch.softmax(att, dim=-1)
            else:
                att = torch.sigmoid(att)
            att = att.view(b_size, self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
            att = att.contiguous()
            att = att * (self.kernel_size**2)
            base_out = self.base_conv.forward(base_x, w=att)
            return base_out, tran_x

class TConvTranspose(nn.Module):
    """
    Conv transformation
    """

    def __init__(self, in_channels, out_channels, dim_in=32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 use_bias=True):
        super(TConvTranspose, self).__init__()
        ## base conv
        self.kernel_size = kernel_size
        self.base_conv = BaseConvTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, use_bias=use_bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.g2pa_wgen = nn.Sequential(
            nn.Linear(dim_in + in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, in_channels+out_channels+kernel_size**2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, base_x, tran_x=None, mode='normal'):
        """
        adptatively training according to mode
        If mode is 'normal', only trains base network
        Else trains paramter generator and fine tunes base network
        """
        if mode == 'normal' or self.kernel_size == 1:
            base_out = self.base_conv(base_x)
            return base_out, None
        else:
            b_size = tran_x.shape[0]
            in_x = torch.mean(base_x, dim=[-2, -1])
            att = self.g2pa_wgen(torch.cat([tran_x, in_x], dim=-1))
            w1, w2, w3 = att[:, :self.in_channels], att[:,self.in_channels:self.out_channels+self.in_channels], att[:, self.in_channels+self.out_channels:]
            att = w1.unsqueeze(1) + w2.unsqueeze(-1)  # [B, C_o, C_i]
            att = torch.sigmoid(att)
            att = att.unsqueeze(-1) * w3.view(b_size, 1, 1, -1)
            if self.kernel_size > 1:
                att = torch.softmax(att, dim=-1)
            else:
                att = torch.sigmoid(att)
            att = att.view(b_size, self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)
            att = att.contiguous()
            att = att * (self.kernel_size**2)
            base_out = self.base_conv.forward(base_x, w=att)
            return base_out, tran_x

def compare(rounds=1000):
    ## compare parallel coim with seralized coim
    groups = 4
    xs = torch.rand(8, 12, 64, 64).cuda()  # a batch of 8 examples
    ws = torch.rand(8, 16, 12 // groups, 3, 3).cuda()  # 8 specific weight
    bs = torch.rand(16).cuda()  # 8 biases
    # first imeplementation with for-loop
    tic = time.time()
    for _ in range(rounds):
        outs = []
        for i in range(xs.shape[0]):
            o = F.conv2d(xs[[i]], weight=ws[i], bias=bs, stride=1, padding=1, groups=groups)
            outs.append(o)
        out = torch.cat(outs, dim=0)
    toc = time.time()
    print(out.shape)
    print("using for-loop: {} s".format(toc-tic))
    # second implemention with group conv
    tic = time.time()
    for _ in range(rounds):
        xs2 = xs.view(1, -1, 64, 64)
        ws2 = ws.view(-1, 12 // groups, 3, 3)
        bs2 = bs.repeat(8)
        out2 = F.conv2d(xs2, ws2, bs2, stride=1, padding=1, groups=8*groups)
        out2 = out2.reshape(8, 16, 64, 64)
    toc = time.time()
    print("using group conv: {} s".format(toc-tic))
    diff = (out - out2).abs().sum()
    print("difference: ", diff)

if __name__ == "__main__":
    for _ in range(2):
        compare(rounds=1)
    layer = BaseConv(16, 12, 3, 1, 1, 1, groups=4, use_bias=True)
    x = torch.rand(4, 16, 24, 24)
    out = layer.forward(x, w=None)
    print(out.shape)
    w = torch.rand(4, 12, 4, 3, 3)
    b = torch.rand(12)
    out = layer.forward(x, w=w)
    print(out.shape)

    layer = TConv(in_channels=16, out_channels=12, dim_in=32, kernel_size=3)
    o, _ = layer(x)
    print(o.shape)
    tran_x = torch.rand(4, 32)
    o, _ = layer.forward(base_x=x, tran_x=tran_x, mode="tran")
    print(o.shape)

    layer = TConvTranspose(16, 12, 32)
    o, _ = layer(x)
    print(o.shape)
    o, _ = layer(x, tran_x, "tran")
    print(o.shape)
