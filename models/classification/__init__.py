#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:03:21 2020

@author: emcastro
"""
import torch
from torch import nn
from models.classification import resnet, vgg, cbr

def get_model(name, num_classes, n_input_channels=3, pretrained=True,
              change_last_layer_if_necessary=True, **kwargs):
    if pretrained:
        assert n_input_channels in [1, 3]

    if name.startswith("resnet"):
        if not pretrained:
            model = getattr(resnet, name)(pretrained, num_classes=num_classes,
                                          input_channels=n_input_channels, **kwargs)

        else:
            model = getattr(resnet, name)(pretrained, num_classes=1000, input_channels=3, **kwargs)
            if num_classes!= 1000 and change_last_layer_if_necessary:
                model.fc = nn.Linear(model.fc.in_features, num_classes)

            if n_input_channels == 1:
                weight = model.conv1.weight.detach()
                weight = torch.sum(weight, dim=1).unsqueeze(1)
                model.conv1 = nn.Conv2d(1, weight.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
                model.conv1.weight.data = weight.detach()

    if name.startswith("vgg"):
        if pretrained:
            raise(NotImplementedError("This has not been checked"))
        model = getattr(vgg, name)(pretrained, in_channels=n_input_channels, num_classes=num_classes, **kwargs)

    if name == "CBR":
        if pretrained:
            raise(NotImplementedError("CBR does not have pretrained weights stored"))
        model = cbr.CBR(in_channels=n_input_channels, num_classes=num_classes, **kwargs)
    if name == "babyCBR":
        if pretrained:
            raise(NotImplementedError("babyCBR does not have pretrained weights stored"))
        model = cbr.babyCBR(in_channels=n_input_channels, num_classes=num_classes, **kwargs)
    return model
