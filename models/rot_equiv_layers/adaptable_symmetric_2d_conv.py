#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:40:36 2022

@author: emcastro
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.parameter import Parameter


"""
T \in {"p1", "p2", "p4", "p1m", "p2m", "p4m"}
"""

def reduce_structure(W, b, first=False):
    W, b = transform(W, b, first=first)
    return W, b

# gather
def indices(group="p4m", weight_shape=None):
    if group=="p4m":
        idx = torch.arange(8).reshape([4, 2])
        idx = idx.repeat([4, 2, 1, 1, 1])

        idx[:, 1] = torch.roll(idx[:, 1], 1, 3)

        idx[1, 0,:,:,0] = torch.roll(idx[1, 0,:,:,0], 1, 1)
        idx[1, 0,:,:,1] = torch.roll(idx[1, 0,:,:,1], -1, 1)
        idx[2, 0,:,:,0] = torch.roll(idx[2, 0,:,:,0], 1, 1)
        idx[2, 0,:,:,1] = torch.roll(idx[2, 0,:,:,1], -1, 1)
        idx[3, 0,:,:,0] = torch.roll(idx[3, 0,:,:,0], 1, 1)
        idx[3, 0,:,:,1] = torch.roll(idx[3, 0,:,:,1], -1, 1)

        idx[1, 1,:,:,0] = torch.roll(idx[1, 0,:,:,0], -1, 1)
        idx[1, 1,:,:,1] = torch.roll(idx[1, 0,:,:,1], 1, 1)
        idx[2, 1,:,:,0] = torch.roll(idx[2, 0,:,:,0], -1, 1)
        idx[2, 1,:,:,1] = torch.roll(idx[2, 0,:,:,1], 1, 1)
        idx[3, 1,:,:,0] = torch.roll(idx[3, 0,:,:,0], -1, 1)
        idx[3, 1,:,:,1] = torch.roll(idx[3, 0,:,:,1], 1, 1)

        idx = idx.reshape([8,1,8,1])
        n, f, k1, k2 = weight_shape

        idx = idx.unsqueeze(-1).unsqueeze(-1).expand([8,n,8,f//8,k1,k2])

    return idx


def transform(W, b, rot=True, mirror=True, first=False):
    s = W.shape

    rot = 1 if rot else 4
    mirror = 1 if mirror else 2

    if first == False:
        W = W.reshape([rot, mirror, s[0], 4, 2, s[1]//8, s[2], s[3]])
        if rot == 1:
            W = torch.cat([W,
                           torch.rot90(W, 1, dims=(6, 7)),
                           torch.rot90(W, 2, dims=(6, 7)),
                           torch.rot90(W, 3, dims=(6, 7))], 0)
        if mirror == 1:
            W = torch.cat([W, torch.flip(W, dims=(-1,))], 1)

        W = W.reshape([8, s[0], 8, s[1]//8, s[2], s[3]])
        idx = indices("p4m", s).to(W.device)

        W = torch.gather(W, 2, idx)

    else:
        W = W.reshape([rot, mirror, s[0], s[1], s[2], s[3]])
        if rot == 1:
            W = torch.cat([W,
                           torch.rot90(W, 1, dims=(4, 5)),
                           torch.rot90(W, 2, dims=(4, 5)),
                           torch.rot90(W, 3, dims=(4, 5))], 0)
        if mirror == 1:
            W = torch.cat([W, torch.flip(W, dims=(-1,))], 1)

    W = W.reshape([-1, s[1], s[2], s[3]])

    b = b.expand([4, 2, s[0]])
    b = b.reshape([-1])

    return W, b

"""
def reduce_rot(W, b, previous, curr, first=False):
    s = W.shape
    if previous == "4" and curr == "1":
        in_c = W.shape[1] // 4

        if not first:
            W = torch.stack([torch.roll(torch.rot90(W, i, dims=(2,3)).contiguous(), i * in_c, dims=1).contiguous() for i in range(4)]).contiguous()
        else:
            W = torch.stack([torch.rot90(W, i, dims=(2,3)) for i in range(4)], 1).contiguous()

        #W = torch.stack([torch.rot90(W, i, dims=(2,3)) for i in range(4)], 1).contiguous()
        b = torch.tile(b, [1, 4])
        return W.reshape([s[0]*4, s[1], s[2], s[3]]), torch.reshape(b, [-1])
    return

def transform(W, b, rot, mirror, first=False):
    # W [C_out, C_in, k, k]
    # b [C_out]
    # rot - "1", "2", "4"
    # mirror - True, False
    n = 1
    s = W.shape
    if rot != "1":
        if rot == "2":
            n = 2
            rate = 2
        elif rot == "4":
            n = 4
            rate = 1
        in_c = W.shape[1] // 4
        if not first:
            W = torch.stack([torch.roll(torch.rot90(W, i * rate, dims=(2,3)).contiguous(), i * in_c, dims=1).contiguous() for i in range(4)]).contiguous()
        else:
            W = torch.stack([torch.rot90(W, i * rate, dims=(2,3)) for i in range(n)], 1).contiguous()
        b = torch.tile(b, [1, n])
    if mirror:
        n *= 2
        W = torch.stack([W, torch.flip(W, -1)], 2).contiguous()
        b = torch.tile(b, [1, 1, 2])
    W = W.reshape([s[0]*n, s[1], s[2], s[3]])
    b = b.reshape([s[0]*n])
    return W, b
"""
class adaptable_symmetric_2D_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rot_group, mirror_group, first=False):
        super(adaptable_symmetric_2D_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rot_group = rot_group
        self.mirror_group = mirror_group
        self.W = Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        self.b = Parameter(torch.empty(out_channels))
        self.reset_parameters()
        self.padding = (kernel_size-1)//2
        self.first = first

    def convert(self, rot_group, mirror_group):
        if rot_group != self.rot_group:
            W, b = reduce_structure(self.W, self.b, first=self.first)
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
            self.rot_group = rot_group

    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        if self.rot_group != "none":
            W, b = transform(self.W, self.b, first=self.first)
        else:
            W, b = self.W, self.b
        return F.conv2d(x, W, b, padding=self.padding)
