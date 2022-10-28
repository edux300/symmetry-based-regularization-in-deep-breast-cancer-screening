#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:13:53 2020

@author: emcastro
"""
import torch
from torch import nn
from models.rot_equiv_layers import RotConv2D, RotBatchNorm2d, GroupPooling2d

"""
class CBR(nn.Module):
    def __init__(self, in_channels=1, filters=[32, 64, 128, 256, 512], num_classes=5):
        super(CBR, self).__init__()
        if isinstance(filters, int):
            filters = [filters * int(i / 32) for i in [32, 64, 128, 256, 512]]
        in_filters = [in_channels, *filters[:-1]]
        out_filters = filters
        self.units = nn.ModuleList()
        for in_f, out_f in zip(in_filters, out_filters):
            self.units.append(unit(in_f, out_f))    
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_f, num_classes)
        torch.nn.init.xavier_normal_(self.fc.weight, 1)
        torch.nn.init.zeros_(self.fc.bias)
        print(self)

    def forward(self, x):
        for i, unit in enumerate(self.units):
            x = unit(x)
            x = self.maxpool(x)

        x = self.avgpool(x)
        x = x.reshape((x.shape[0], x.shape[1]))
        x = self.dropout(x)
        return self.fc(x)
"""

    
class CBR(nn.Module):
    def __init__(self, in_channels, num_classes=5,
                 filters=None,
                 ksizes = 3,
                 group=None,
                 dense_classifier=False, group_pool=None):
        super(CBR, self).__init__()
        if filters is None or filters is False:
            filters = 32
        if isinstance(filters, int):
            filters = [[1*filters] * 2, [2*filters] * 2, [4*filters] * 2,
                       [8*filters] * 2, [16*filters] * 2]

        if group is None or group is False:
            group = [["z2" for fi in fo] for fo in filters]
            group_pool=False if group_pool is None else group_pool
        elif group is True:
            group = [["p4" for fi in fo] for fo in filters]
            group_pool=True if group_pool is None else group_pool
        elif isinstance(group, str):
            group = [[group for fi in fo] for fo in filters]
        if isinstance(ksizes, int):
            ksizes = [[ksizes for fi in fo] for fo in filters]

        in_f = in_channels
        first = True
        self.seq = nn.ModuleList()
        for f_outer, k_outer, g_outer in zip(filters, ksizes, group):
            for out_f, ks, g in zip(f_outer, k_outer, g_outer):
                self.seq.append(unit(in_f, out_f, ks, g, first))
                first = False
                in_f = out_f
            self.seq.append(nn.MaxPool2d(2, 2))
        
        if group_pool:
            out_f = out_f//4
            self.seq.append(GroupPooling2d())

        if dense_classifier:
            self.seq.append(nn.Flatten())
            self.seq.append(nn.Dropout(0.2))
            self.fc1 = nn.Linear(out_f * 7 * 7, out_f)
            self.fc1_actv = nn.ReLU()
            self.fc2 = nn.Linear(out_f, out_f)
            self.fc2_actv = nn.ReLU()
            self.fc3 = nn.Linear(out_f, num_classes)
            self.seq.append(self.fc1)
            self.seq.append(self.fc1_actv)
            self.seq.append(self.fc2)
            self.seq.append(self.fc2_actv)
            self.seq.append(self.fc3)

            torch.nn.init.xavier_normal_(self.fc1.weight, 1)
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.xavier_normal_(self.fc2.weight, 1)
            torch.nn.init.zeros_(self.fc2.bias)
            torch.nn.init.xavier_normal_(self.fc3.weight, 1)
            torch.nn.init.zeros_(self.fc3.bias)
        else:
            self.seq.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.seq.append(nn.Flatten())
            self.seq.append(nn.Dropout(0.2))
            self.fc = nn.Linear(out_f, num_classes)
            self.seq.append(self.fc)

            torch.nn.init.xavier_normal_(self.fc.weight, 1)
            torch.nn.init.zeros_(self.fc.bias)
            print(self)

    def forward(self, x):
        for module in self.seq:
            x = module(x)
        return x

class babyCBR(CBR):
    def __init__(self, in_channels, num_classes=5, filters=None,
                 ksizes = 3, group=None, dense_classifier=False):
        if filters is None or filters is False:
            filters = 16
        if isinstance(filters, int):
            filters = [[1*filters], [2*filters], [4*filters],
                       [8*filters], [16*filters]]
            super(babyCBR, self).__init__(in_channels, num_classes, filters, ksizes,
                 group, dense_classifier)
    
class unit(nn.Module):
    def __init__(self, in_f, out_f, ks=7, group="z2", first=False):
        super(unit, self).__init__()
        padding = (ks-1)//2
        if group=="z2":
            self.conv = nn.Conv2d(in_f, out_f, kernel_size=ks, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_f, momentum=0.1)
        elif group=="p4":
            self.conv = RotConv2D(in_f, out_f, kernel_size=ks, padding=padding, bias=False, first=first)
            self.bn = RotBatchNorm2d(out_f, momentum=0.1)

        self.relu = nn.ReLU(True)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")    

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

"""            
        super(CBR, self).__init__()
        if isinstance(filters, int):
            filters = [filters * int(i / 32) for i in [32, 64, 128, 256, 512]]
        in_filters = [in_channels, *filters[:-1]]
        out_filters = filters
        self.units = nn.ModuleList()
        for in_f, out_f in zip(in_filters, out_filters):
            self.units.append(unit(in_f, out_f))    
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_f, num_classes)
        torch.nn.init.xavier_normal_(self.fc.weight, 1)
        torch.nn.init.zeros_(self.fc.bias)
        print(self)


"""  
"""
class unit(nn.Module):
    def __init__(self, in_f, out_f, ks=7):
        super(unit, self).__init__()
        padding = (ks-1)//2
        self.conv = nn.Conv2d(in_f, out_f, kernel_size=ks, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_f)
        #self.bn = torch.nn.Identity()
        self.relu = nn.ReLU(True)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Group_CBR(nn.Module):
    def __init__(self, in_channels=1, filters=[32, 64, 128, 256, 512], num_classes=5):
        super(Group_CBR, self).__init__()
        if isinstance(filters, int):
            filters = [filters * int(i / 32) for i in [32, 64, 128, 256, 512]]
        in_filters = [in_channels, *filters[:-1]]
        out_filters = filters
        self.units = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        for i, (in_f, out_f) in enumerate(zip(in_filters, out_filters)):
            first = True if i == 0 else False
            #self.units.append(group_unit(in_f, out_f, first=first))
            self.units.append(group_unit2(in_f, out_f, first=first))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.fc = nn.Linear(out_f // 2, num_classes)
        self.fc = nn.Linear(out_f, num_classes)
        torch.nn.init.xavier_normal_(self.fc.weight, 1)
        torch.nn.init.zeros_(self.fc.bias)
        print(self)
        
    def forward(self, x):
        for i, unit in enumerate(self.units):
            x = unit(x)
            #x = plane_group_spatial_max_pooling(x, 2, 2)
            x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.reshape((x.shape[0], x.shape[1]))
        x = self.dropout(x)
        return self.fc(x)
    
class group_unit2(nn.Module):
    def __init__(self, in_f, out_f, ks=7, first=False, gain=1):
        super(group_unit2, self).__init__()
        padding = (ks-1)//2
        if first:
            self.conv = RotConv2D(in_f, out_f, kernel_size=ks, padding=padding, bias=False, first=True)
        else:
            self.conv = RotConv2D(in_f, out_f, kernel_size=ks, padding=padding, bias=False, first=False)
        self.bn = RotBatchNorm2d(out_f)
        #self.bn = torch.nn.Identity()
        self.relu = nn.ReLU(True)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class HalfGroup_CBR(nn.Module):
    def __init__(self, in_channels=1, filters=[32, 64, 128, 256, 512],
                 group_layers_until=4, num_classes=5):
        super(HalfGroup_CBR, self).__init__()
        if isinstance(filters, int):
            filters = [filters * int(i / 32) for i in [32, 64, 128, 256, 512]]
        in_filters = [in_channels, *filters[:-1]]
        out_filters = filters
        self.units = nn.ModuleList()
        self.group_layers_until = group_layers_until
        self.maxpool = nn.MaxPool2d(2, 2)
        for i, (in_f, out_f) in enumerate(zip(in_filters, out_filters)):
            if i < group_layers_until:
                if i==0:
                    #self.units.append(group_unit(in_f, out_f//2, first=True))
                    self.units.append(group_unit2(in_f, out_f, first=True))
                else:
                    #self.units.append(group_unit(in_f//2, out_f//2, first=False))
                    self.units.append(group_unit2(in_f, out_f, first=False))
            else:
                self.units.append(unit(in_f, out_f))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(out_f, num_classes)
        torch.nn.init.xavier_normal_(self.fc.weight, 1)
        torch.nn.init.zeros_(self.fc.bias)
        print(self)

    def forward(self, x):
        for i, unit in enumerate(self.units):
            if i == self.group_layers_until:
                s = x.shape
                #x = x.reshape([s[0], s[1]*s[2], s[3], s[4]])
            if i < self.group_layers_until:
                x = unit(x)
                #x = plane_group_spatial_max_pooling(x, 2, 2)
                x = self.maxpool(x)
            else:
                x = unit(x)
                x = self.maxpool(x)

        x = self.avgpool(x)
        x = x.reshape((x.shape[0], x.shape[1]))
        x = self.dropout(x)
        return self.fc(x)
"""