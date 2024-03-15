"""
encoder refers to AirNet
repository: https://github.com/XLearning-SCU/2022-CVPR-AirNet/blob/main/net/encoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.short_cut(x))

class FeatureExtract(nn.Module):
    def __init__(self, in_channels, out_channels, dim_in=128):
        super(FeatureExtract, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.resblock = nn.Sequential(
            ResBlock(out_channels, out_channels*2),  # 128 -> 64
            ResBlock(out_channels*2, out_channels*4),  # 64 -> 32
            ResBlock(out_channels*4, out_channels*8),   # 32 -> 16
        )
        self.mlp = nn.Sequential(
            nn.Linear(out_channels*8, dim_in),
            nn.LeakyReLU(0.1, True),
            nn.Linear(dim_in, dim_in)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock(out)
        out_feature = out
        out = self.avg_pool(out).squeeze(-1).squeeze(-1)  # 256-dimensional vector
        out_z = self.mlp(out)
        out_z = F.normalize(out_z, dim=-1)
        out_feature = F.normalize(out_feature, dim=1)  # [B, 128, 16, 16] (contrastive learning)
        return out_z, out_feature  # [B, D] shape, unit vectors of dimension D

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, in_channels, out_channels, dim, m=0.999, temperature=1.0):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.m = m
        self.temperature = temperature
        self.dim = dim

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = FeatureExtract(in_channels, out_channels, dim)
        self.encoder_k = FeatureExtract(in_channels, out_channels, dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, im_q, im_k, im_negs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        b, n_neg, c, h, w = im_negs.shape
        q, q_feat = self.encoder_q(im_q)
        _, fc, fh, fw = q_feat.shape  # resolution of features
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_negs = im_negs.view(b*n_neg, c, h, w).contiguous()
            _, kn_feat = self.encoder_k(torch.cat([im_k, im_negs], dim=0))  # [b + b*n_neg, c, h, w]
            kn_feat = nn.functional.normalize(kn_feat, dim=1)
            k_feat = kn_feat[:b]
            neg_feat = kn_feat[b:].view(b, n_neg, fc, fh, fw).contiguous()  # [b, n_neg, c, h, w]

        q_feat = q_feat.view(b, fc*fh*fw).contiguous()
        k_feat = k_feat.view(b, fc*fh*fw).contiguous()
        neg_feat = neg_feat.view(b, n_neg, fc*fh*fw).contiguous()
        l_pos = (q_feat * k_feat).sum(dim=-1, keepdims=True) / (fh*fw)
        l_neg = (q_feat.unsqueeze(1) * neg_feat).sum(dim=-1) / (fh*fw)
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.temperature
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return logits, labels, q