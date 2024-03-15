import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CoIM import TConv


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dim_in, activation='relu'):
        super(ConvUnit, self).__init__()
        self.conv = TConv(in_channels, out_channels, dim_in, 3, 1, 1)
        self.activation = activation
    
    def forward(self, x, tran_x, mode='normal'):
        out, _ = self.conv(x, tran_x, mode)
        if self.activation == 'relu':
            return torch.relu(out)
        if self.activation == 'sigmoid':
            return torch.sigmoid(out)
        if self.activation == 'tanh':
            return torch.tanh(out)
        if self.activation == None:
            return out

class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_in):
        super(BaseBlock, self).__init__()
        self.conv1 = TConv(in_channels, out_channels, dim_in, 3, 1, 1)
        self.conv2 = TConv(in_channels, out_channels, dim_in, 3, 1, 1)
    
    def forward(self, x, tran_x, mode='normal'):
        out, _ = self.conv1(x, tran_x, mode)
        out = torch.relu(out)
        out, _ = self.conv2(out, tran_x, mode)
        out = torch.relu(out)
        return out

class BRN(nn.Module):    
    def __init__(self, opt):
        super(BRN, self).__init__()
        self.iteration = opt.inter_iter
        self.use_GPU = opt.use_GPU

        self.conv0 = ConvUnit(9, 32, opt.dim_in, activation='relu')

        self.res_conv1 = BaseBlock(32, 32, opt.dim_in)
        self.res_conv2 = BaseBlock(32, 32, opt.dim_in)
        self.res_conv3 = BaseBlock(32, 32, opt.dim_in)
        self.res_conv4 = BaseBlock(32, 32, opt.dim_in)
        self.res_conv5 = BaseBlock(32, 32, opt.dim_in)

        self.conv_i = ConvUnit(32+32+32, 32, opt.dim_in, activation='sigmoid')
        self.conv_f = ConvUnit(32+32+32, 32, opt.dim_in, activation='sigmoid')
        self.conv_g = ConvUnit(32+32+32, 32, opt.dim_in, activation='tanh')
        self.conv_o = ConvUnit(32+32+32, 32, opt.dim_in, activation='sigmoid')

        self.conv = ConvUnit(32, 3, opt.dim_in, activation=None)
        self.conv0_r = ConvUnit(6, 32, opt.dim_in, activation='relu')

        self.res_conv1_r = BaseBlock(32, 32, opt.dim_in)
        self.res_conv2_r = BaseBlock(32, 32, opt.dim_in)
        self.res_conv3_r = BaseBlock(32, 32, opt.dim_in)
        """
        self.res_conv4_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        """
        self.conv_i_r = ConvUnit(32+32+32, 32, opt.dim_in, activation='sigmoid')
        self.conv_f_r = ConvUnit(32+32+32, 32, opt.dim_in, activation='sigmoid')
        self.conv_g_r = ConvUnit(32+32+32, 32, opt.dim_in, activation='tanh')
        self.conv_o_r = ConvUnit(32+32+32, 32, opt.dim_in, activation='sigmoid')

        self.conv_r = ConvUnit(32, 3, opt.dim_in, activation=None)


    def forward(self, input, tran_x, mode='normal'):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        
        x = input
        r = input
        h = torch.zeros(batch_size, 32, row, col)
        c = torch.zeros(batch_size, 32, row, col)
        h_r = torch.zeros(batch_size, 32, row, col)
        c_r = torch.zeros(batch_size, 32, row, col)
        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            h_r = h_r.cuda()
            c_r = c_r.cuda()

        # x_list = []
        # r_list = []
        for i in range(self.iteration):
            r = torch.cat((input, r), 1)
            r = self.conv0_r(r, tran_x, mode)
            r = torch.cat((r, h_r, h), 1)
            i_r = self.conv_i_r(r, tran_x, mode)
            f_r = self.conv_f_r(r, tran_x, mode)
            g_r = self.conv_g_r(r, tran_x, mode)
            o_r = self.conv_o_r(r, tran_x, mode)
            c_r = f_r * c_r + i_r * g_r
            h_r = o_r * torch.tanh(c_r)
            resr = h_r
            r = F.relu(self.res_conv1_r(h_r, tran_x, mode) + resr)
            resr = r
            r = F.relu(self.res_conv2_r(r, tran_x, mode) + resr)
            resr = r
            r = F.relu(self.res_conv3_r(r, tran_x, mode) + resr)
            r = self.conv_r(r, tran_x, mode)
            # r_list.append(r)

            x = torch.cat((input, x, r), 1)
            x = self.conv0(x, tran_x, mode)
            x = torch.cat((x, h, h_r), 1)
            i = self.conv_i(x, tran_x, mode)
            f = self.conv_f(x, tran_x, mode)
            g = self.conv_g(x, tran_x, mode)
            o = self.conv_o(x, tran_x, mode)
            c = f * c + i * g
            h = o * torch.tanh(c)
            resx = h
            x = F.relu(self.res_conv1(h, tran_x, mode) + resx)
            resx = x
            x = F.relu(self.res_conv2(x, tran_x, mode) + resx)
            resx = x
            x = F.relu(self.res_conv3(x, tran_x, mode) + resx)
            resx = x
            x = F.relu(self.res_conv4(x, tran_x, mode) + resx)
            resx = x
            x = F.relu(self.res_conv5(x, tran_x, mode) + resx)

            x = self.conv(x, tran_x, mode)
            # x_list.append(x)

        return x# , x_list, r, r_list


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.append('../')
    from utils import print_network
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.inter_iter = 8
    opt.dim_in = 128
    opt.use_GPU = False
    net = BRN(opt)
    print_network(net)
    x = torch.rand(1, 3, 20, 20)
    outs = net(x, tran_x=None, mode='normal')
    print(outs.shape)
    tran_x = torch.rand(1, 128)
    outs = net(x, tran_x, mode='tran')
    print(outs.shape)


    num_params = 0
    for param in net.parameters():
        # if param.requires_grad:
        num_params += param.numel()
    print('Total number of parameters:  %d' % num_params)