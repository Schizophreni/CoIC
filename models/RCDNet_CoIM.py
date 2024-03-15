"""
This file defines the model architecture of RCDNet
code from the implementation of leftthomas
"""
import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CoIM import TConv


class RCDNet(nn.Module):
    def __init__(self, opt):
        super(RCDNet, self).__init__()
        # not include the initialization process
        self.iter = opt.inter_iter - 1
        num_map, num_channel, num_block = opt.num_map, opt.num_channel, opt.num_block
        num_stage = opt.inter_iter

        # step size
        self.m_step = nn.Parameter(torch.ones(self.iter, 1))
        self.b_step = nn.Parameter(torch.full(size=(num_stage, 1), fill_value=5.0))

        # rain kernel [3, 32, 9, 9]
        rain_kernel = torch.FloatTensor(io.loadmat('models/kernel.mat')['C9'])
        self.c0 = nn.Parameter(rain_kernel)
        self.c = nn.Parameter(rain_kernel)

        # filter for initializing B and Z
        self.cz = nn.Parameter(torch.full(size=(num_channel, 3, 3, 3), fill_value=1.0 / 9))
        # for sparse rain layer
        self.tau = nn.Parameter(torch.ones(1))

        # proxNet
        self.proxNet_M_S = nn.Sequential(*[MNet(num_map, num_block, opt.dim_in) for _ in range(num_stage)])
        self.proxNet_B_S = nn.Sequential(*[BNet(num_channel, num_block, opt.dim_in) for _ in range(num_stage)])
        self.proxNet_B_0 = BNet(num_channel, num_block, opt.dim_in)
        self.proxNet_B_L = BNet(num_channel, num_block, opt.dim_in)

    def forward(self, x, tran_x, mode='normal'):
        list_b, list_r = [], []
        # initialize B0 and M0
        bz = self.proxNet_B_0(torch.cat((x, F.conv2d(x, self.cz, stride=1, padding=1)), dim=1), tran_x, mode)
        b0, z0 = bz[:, :3, :, :], bz[:, 3:, :, :]

        # 1st iteration：Updating B0-->M1
        # for sparse rain layer
        r_hat = torch.relu(x - b0 - self.tau)
        # /10 for controlling the updating speed
        m = self.proxNet_M_S[0](F.conv_transpose2d(r_hat, self.c0.div(10), stride=1, padding=4), tran_x, mode)

        # 1st iteration: Updating M1-->B1
        r = F.conv2d(m, self.c.div(10), stride=1, padding=4)
        b_hat = (1 - self.b_step[0] / 10) * b0 + self.b_step[0] / 10 * (x - r)
        bz = self.proxNet_B_S[0](torch.cat((b_hat, z0), dim=1), tran_x, mode)
        b, z = bz[:, :3, :, :], bz[:, 3:, :, :]
        list_r.append(r)
        list_b.append(b)
        for i in range(self.iter):
            # M-net
            epsilon = self.m_step[i, :] / 10 * F.conv_transpose2d((r - (x - b)), self.c.div(10), stride=1, padding=4)
            m = self.proxNet_M_S[i + 1](m - epsilon, tran_x, mode)

            # B-net
            r = F.conv2d(m, self.c.div(10), stride=1, padding=4)
            b_hat = (1 - self.b_step[i + 1, :] / 10) * b + self.b_step[i + 1, :] / 10 * (x - r)
            bz = self.proxNet_B_S[i + 1](torch.cat((b_hat, z), dim=1), tran_x, mode)
            b, z = bz[:, :3, :, :], bz[:, 3:, :, :]
            list_r.append(r)
            list_b.append(b)
        b = self.proxNet_B_L(bz, tran_x, mode)[:, :3, :, :]
        list_b.append(b)
        return b0, list_b, list_r


class BaseBlock(nn.Module):
    def __init__(self, num_channel, dim_in):
        super(BaseBlock, self).__init__()
        self.conv1 = TConv(num_channel, num_channel, dim_in, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_channel)
        self.conv2 = TConv(num_channel, num_channel, dim_in, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_channel)
    
    def forward(self, x, tran_x, mode='normal'):
        out, _ = self.conv1(x, tran_x, mode)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out, _ = self.conv2(out, tran_x, mode)
        out = self.bn2(out)
        return out

def make_block(num_block, num_channel, dim_in):
    layers = [BaseBlock(num_channel, dim_in) for _ in range(num_block)]
    return nn.Sequential(*layers)


# proxNet_M
class MNet(nn.Module):
    def __init__(self, num_map, num_block, dim_in):
        super(MNet, self).__init__()
        self.channels = num_map
        self.num_block = num_block
        self.layer = make_block(self.num_block, self.channels, dim_in)
        # for sparse rain map
        self.tau = nn.Parameter(torch.full(size=(1, self.channels, 1, 1), fill_value=0.5))

    def forward(self, x, tran_x, mode='normal'):
        for i in range(self.num_block):
            x = torch.relu(x + self.layer[i](x, tran_x, mode))
        x = torch.relu(x - self.tau)
        return x


# proxNet_B
class BNet(nn.Module):
    def __init__(self, num_channel, num_block, dim_in):
        super(BNet, self).__init__()
        # 3 means R,G,B channels for color image
        self.channels = num_channel + 3
        self.num_block = num_block
        self.layer = make_block(self.num_block, self.channels, dim_in)

    def forward(self, x, tran_x, mode):
        for i in range(self.num_block):
            x = torch.relu(x + self.layer[i](x, tran_x, mode))
        return x

if __name__ == '__main__':
    import argparse
    import sys
    sys.path.append('../')
    from utils import print_network
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.inter_iter = 17
    opt.num_map, opt.num_channel, opt.num_block = 32, 32, 4
    opt.dim_in = 128
    net = RCDNet(opt)
    print_network(net)
    x = torch.rand(1, 3, 20, 20)
    outs = net(x, tran_x=None, mode='normal')[0]
    print(outs[0].shape)
    tran_x = torch.rand(1, opt.dim_in)
    outs = net(x, tran_x, mode='tran')[0]
    print(outs[0].shape)


    num_params = 0
    for param in net.parameters():
        # if param.requires_grad:
        num_params += param.numel()
    print('Total number of parameters:  %d' % num_params)