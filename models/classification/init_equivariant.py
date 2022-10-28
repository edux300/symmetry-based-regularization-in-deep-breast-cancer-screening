import torch
from torch import nn
import torchvision.models
from models.classification.equivariance_converter import to_group_conv
from models.classification.equivariance_converter import to_group_layernorm2d
from models.classification.equivariance_converter import to_group_layernorm
from models.classification.equivariance_converter import to_group_bnorm
from models.classification.equivariance_converter import to_group_linear
from models.classification.equivariance_converter import to_group_CNBlock

from models.rot_equiv_layers.base import RotConv2D

def sum_layer_weights(layer):
    weight = layer.weight.detach()
    if isinstance(layer, nn.Conv2d):
        new_layer = nn.Conv2d(in_channels = 1,
                              out_channels = weight.size(0),
                              kernel_size = (weight.size(2), weight.size(3)),
                              stride = layer.stride,
                              padding = layer.padding,
                              dilation = layer.dilation,
                              groups = layer.groups,
                              bias = True if layer.bias is not None else False,
                              padding_mode = layer.padding_mode,
                              device = weight.device,
                              dtype = weight.dtype,)
    elif isinstance(layer, RotConv2D):
        n_filters, in_channels, k1, k2 = layer.weight.shape
        new_layer = RotConv2D(1,
                              layer.out_channels,
                              layer.kernel_size,
                              stride=layer.stride,
                              padding=layer.padding,
                              bias=True if layer.bias is not None else False,
                              groups=layer.groups,
                              first=layer.first)
    else:
        raise(ValueError("First layer should be a conv layer"))
    
    weight = torch.sum(weight, dim=1).unsqueeze(1)
    new_layer.weight.data = weight.detach()
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.detach()
    return new_layer


def aux_make_equiv_convnext(model, idx, break_sooner, init_copy):
    for i, block in enumerate(model.features[idx]):
        model.features[idx][i].block[0] = to_group_conv(model.features[idx][i].block[0], first=True, init_copy=init_copy)  # small trick here
        model.features[idx][i].block[2] = to_group_layernorm(model.features[idx][i].block[2], init_copy=init_copy)
        model.features[idx][i].block[3] = to_group_linear(model.features[idx][i].block[3], init_copy=init_copy)
        if i==len(model.features[idx])-1 and break_sooner:
            break
        model.features[idx][i].block[5] = to_group_linear(model.features[idx][i].block[5], init_copy=init_copy)
        model.features[idx][i] = to_group_CNBlock(model.features[idx][i])
    return model
    
def make_equivariant_convnext(model, z2_transition, init_copy=False,
                              group_equiv_expand=False):
    assert not group_equiv_expand
    assert z2_transition in [0, 3, 4, 5, 6]
    if z2_transition == 0:
        return model
    model.features[0][0] = to_group_conv(model.features[0][0], first=True)
    model.features[0][1] = to_group_layernorm2d(model.features[0][1])
    
    model = aux_make_equiv_convnext(model, 1, z2_transition == 3, init_copy)
    
    if z2_transition == 3:
        return model
    
    
    model.features[2][0] = to_group_layernorm2d(model.features[2][0])
    model.features[2][1] = to_group_conv(model.features[2][1])
    
    model = aux_make_equiv_convnext(model, 3, z2_transition == 4, init_copy)
    
    if z2_transition == 4:
        return model
    
    model.features[4][0] = to_group_layernorm2d(model.features[4][0])
    model.features[4][1] = to_group_conv(model.features[4][1])
    
    model = aux_make_equiv_convnext(model, 5, z2_transition == 5, init_copy)
    
    if z2_transition == 5:
        return model
    
    model.features[6][0] = to_group_layernorm2d(model.features[6][0])
    model.features[6][1] = to_group_conv(model.features[6][1])
    
    model = aux_make_equiv_convnext(model, 7, break_sooner=False, init_copy=init_copy)
    return model

import numpy as np
def make_equivariant_vgg(model, z2_transition, init_copy=False,
                              group_equiv_expand=False):
    assert z2_transition in [0, 1, 2, 3, 4, 5]
    if z2_transition == 0:
        return model

    group_conv = partial(to_group_conv, init_copy=init_copy, group_equiv_expand=group_equiv_expand)
    group_bnorm = partial(to_group_bnorm, init_copy=init_copy, group_equiv_expand=group_equiv_expand)

    layers = model.features
    change_ind = np.cumsum([isinstance(l, nn.MaxPool2d) for l in layers]) + 1
    for i, ind in enumerate(change_ind):
        if ind>z2_transition:
            break
        else:
            if isinstance(model.features[i], nn.Conv2d):
                model.features[i] = group_conv(model.features[i], first=(i==0))
                last = model.features[i]
                i_last = i
            elif isinstance(model.features[i], nn.BatchNorm2d):
                model.features[i] = group_bnorm(model.features[i])
                last = model.features[i]
                i_last = i
            else:
                assert isinstance(model.features[i], (nn.ReLU, nn.MaxPool2d))

    if group_equiv_expand:
        model.features[i_last] = transition(last, last.num_features)
    return model


def aux_make_equiv_resnet(model, layer, init_copy):
    for i, block in enumerate(getattr(model, layer)):
        block.conv1 = to_group_conv(block.conv1, init_copy=init_copy)
        block.conv2 = to_group_conv(block.conv2, init_copy=init_copy)
        block.conv3 = to_group_conv(block.conv3, init_copy=init_copy)
        block.bn1 = to_group_bnorm(block.bn1, init_copy=init_copy)
        block.bn2 = to_group_bnorm(block.bn2, init_copy=init_copy)
        block.bn3 = to_group_bnorm(block.bn3, init_copy=init_copy)
        
        if block.downsample is not None:
            block.downsample[0] = to_group_conv(block.downsample[0], init_copy=init_copy)
            block.downsample[1] = to_group_bnorm(block.downsample[1], init_copy=init_copy)
    return model

    

def make_equivariant_resnet(model, z2_transition, init_copy=False,
                            group_equiv_expand=False):
    assert not group_equiv_expand
    assert z2_transition in [0, 2, 3, 4, 5, 6]
    if z2_transition == 0:
        return model
    
    model.conv1 = to_group_conv(model.conv1, first=True, init_copy=init_copy)
    model.bn1 = to_group_bnorm(model.bn1, init_copy=init_copy)
    if z2_transition == 2:
        return model

    model = aux_make_equiv_resnet(model, "layer1", init_copy=init_copy)
    if z2_transition == 3:
        return model

    model = aux_make_equiv_resnet(model, "layer2", init_copy=init_copy)
    if z2_transition == 4:
        return model

    model = aux_make_equiv_resnet(model, "layer3", init_copy=init_copy)
    if z2_transition == 5:
        return model

    model = aux_make_equiv_resnet(model, "layer4", init_copy=init_copy)
    if z2_transition == 6:
        return model

def group_cat(features):
    b,_,w,h = features[0].shape
    features = [f.reshape([b, 4, -1, w, h]) for f in features]
    features = torch.cat(features, 2)
    features = features.reshape([b, -1, w, h])
    return features

def aux_make_equiv_block_densenet(block, init_copy, group_equiv_expand=False):
    for l in block:
        block[l].norm1 = to_group_bnorm(block[l].norm1, init_copy=init_copy, group_equiv_expand=group_equiv_expand)
        block[l].conv1 = to_group_conv(block[l].conv1, init_copy=init_copy, group_equiv_expand=group_equiv_expand)
        block[l].norm2 = to_group_bnorm(block[l].norm2, init_copy=init_copy, group_equiv_expand=group_equiv_expand)
        block[l].conv2 = to_group_conv(block[l].conv2, init_copy=init_copy, group_equiv_expand=group_equiv_expand)

    import types
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(group_cat(features))
            features.append(new_features)
        return group_cat(features)
    
    block.forward = types.MethodType(forward, block)
    return block

class transition(nn.Module):
    def __init__(self, layer, in_features):
        super().__init__()
        self.layer = layer
        self.linear = nn.Conv2d(in_features*4, in_features, 1)
        with torch.no_grad():
            self.linear.weight.fill_(0.)
            #init = torch.cat([torch.eye(in_features) for i in range(4)], dim=1)
            #init = (init /4).unsqueeze(-1).unsqueeze(-1)
            #self.linear.weight.copy_(init)
            torch.nn.init.kaiming_uniform_(self.linear.weight, a=np.sqrt(5))
            #self.linear.weight[0:in_features, 0:in_features].copy_(torch.eye(in_features).reshape([in_features, in_features, 1, 1]))
            self.linear.bias.fill_(0.)

    def forward(self, x):
        o = self.linear(self.layer(x))
        #s = o.shape
        #o = o.reshape([s[0], s[1]//4, 4, s[2], s[3]])
        #o = torch.mean(o, dim=2)
        return o

def make_equivariant_densenet(model, z2_transition, init_copy=False,
                              group_equiv_expand=False):
    assert z2_transition in [0, 2, 3, 4, 5, 6]
    if z2_transition == 0:
        return model
    
    model.features[0] = to_group_conv(model.features[0], first=True, init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    model.features[1] = to_group_bnorm(model.features[1], init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)

    if z2_transition == 2:
        if group_equiv_expand:
            model.features[1] = transition(model.features[1], model.features[1].num_features)
        return model

    model.features[4] = aux_make_equiv_block_densenet(model.features[4], init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    if z2_transition == 3:
        if group_equiv_expand:
            model.features[4] = transition(model.features[4], model.features[5].norm.num_features)
        return model

    model.features[5].norm = to_group_bnorm(model.features[5].norm, init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    model.features[5].conv = to_group_conv(model.features[5].conv, init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    model.features[6] = aux_make_equiv_block_densenet(model.features[6], init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    if z2_transition == 4:
        if group_equiv_expand:
            model.features[6] = transition(model.features[6], model.features[7].norm.num_features)
        return model

    model.features[7].norm = to_group_bnorm(model.features[7].norm, init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    model.features[7].conv = to_group_conv(model.features[7].conv, init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    model.features[8] = aux_make_equiv_block_densenet(model.features[8], init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    if z2_transition == 5:
        if group_equiv_expand:
            model.features[8] = transition(model.features[8], model.features[9].norm.num_features)
        return model

    model.features[9].norm = to_group_bnorm(model.features[9].norm, init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    model.features[9].conv = to_group_conv(model.features[9].conv, init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    model.features[10] = aux_make_equiv_block_densenet(model.features[10], init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    model.features[11] = to_group_bnorm(model.features[11], init_copy=init_copy,
                                      group_equiv_expand=group_equiv_expand)
    if z2_transition == 6:
        if group_equiv_expand:
            model.features[11] = transition(model.features[11], model.features[11].num_features)
        return model

from functools import partial
def make_equivariant_efficientnet(model, z2_transition, init_copy=False,
                                  group_equiv_expand=False):
    assert z2_transition in [0, 2, 3, 4, 5, 6]
    if z2_transition == 0:
        return model

    group_conv = partial(to_group_conv, init_copy=init_copy, group_equiv_expand=group_equiv_expand)
    group_bnorm = partial(to_group_bnorm, init_copy=init_copy, group_equiv_expand=group_equiv_expand)

    model.features[0][0] = group_conv(model.features[0][0], first=True)
    model.features[0][1] = group_bnorm(model.features[0][1])

    if z2_transition == 2:
        if group_equiv_expand:
            model.features[0] = transition(model.features[0], model.features[0][1].num_features * 4)
        return model

    for block in model.features[1]:
        block.block[0][0] = group_conv(block.block[0][0], first=True)   # groups
        block.block[0][1] = group_bnorm(block.block[0][1])
        block.block[1].fc1 = group_conv(block.block[1].fc1)
        block.block[1].fc2 = group_conv(block.block[1].fc2)
        block.block[2][0] = group_conv(block.block[2][0])
        block.block[2][1] = group_bnorm(block.block[2][1])

    if z2_transition == 3:
        if group_equiv_expand:
            model.features[1] = transition(model.features[1], model.features[1][-1].block[2][1].num_features * 4)
        return model
    
    return model

def unpack(l, modules=[]):
    if len(list(l.children())) == 0:
        return modules + [l]
    else:
        for x in l.children():
            modules = unpack(x, modules)
        return modules

def find_mod(l, to_find, modules=[]):
    if isinstance(l, to_find):
        return modules + [l]
    else:
        for x in l.children():
            modules = find_mod(x, to_find, modules)
        return modules
    
def initialize_mean_var(layer, statistics):
    if layer.weight is not None:
        assert layer.weight.shape == statistics.weight.shape
        mean, std = torch.mean(statistics.weight).item(), torch.std(statistics.weight).item()
        nn.init.normal_(layer.weight, mean, std)

    if layer.bias is not None:
        assert layer.bias.shape == statistics.bias.shape
        mean, std = torch.mean(statistics.bias).item(), torch.std(statistics.bias).item()
        nn.init.normal_(layer.bias, mean, std)
        
def initialize_model_mean_var(model, pretrained):
    #classes_w_weights = (nn.Conv2d, nn.Linear, nn.LayerNorm, RotConv2D,
    #                     nn.BatchNorm2d, nn.BatchNorm3d)
    model_layers = [x for x in unpack(model) if hasattr(x, "weight")]
    pretr_layers = [x for x in unpack(pretrained) if hasattr(x, "weight")]
    assert len(model_layers) == len(pretr_layers)
    for l1, l2 in zip(model_layers, pretr_layers):
        initialize_mean_var(l1, l2)
    
    from torchvision.models.convnext import CNBlock
    model_block = [x for x in find_mod(model, to_find=CNBlock) if isinstance(x, CNBlock)]
    pretr_block = [x for x in find_mod(pretrained, to_find=CNBlock) if isinstance(x, CNBlock)]
    assert len(model_block) == len(pretr_block)
    for l1, l2 in zip(model_block, pretr_block):
        mean = torch.mean(l2.layer_scale).item()
        nn.init.constant_(l1.layer_scale, mean)
    return model


def make_equivariant(model, z2_transition, init_copy=False, group_equiv_expand=False):
    kwargs = {"z2_transition": z2_transition,
              "init_copy": init_copy,
              "group_equiv_expand": group_equiv_expand}

    if isinstance(model, torchvision.models.resnet.ResNet):
        return make_equivariant_resnet(model, **kwargs)

    if isinstance(model, torchvision.models.vgg.VGG):
        return make_equivariant_vgg(model, **kwargs)

    if isinstance(model, torchvision.models.densenet.DenseNet):
        return make_equivariant_densenet(model, **kwargs)

    if isinstance(model, torchvision.models.efficientnet.EfficientNet):
        return make_equivariant_efficientnet(model, **kwargs)

    if isinstance(model, torchvision.models.convnext.ConvNeXt):
        return make_equivariant_convnext(model, **kwargs)


def adjust_layers_if_needed(model, temp_n_classes, num_classes, n_input_channels):
    if temp_n_classes != num_classes:
        if isinstance(model, torchvision.models.resnet.ResNet):
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        if isinstance(model, torchvision.models.vgg.VGG):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

        if isinstance(model, torchvision.models.densenet.DenseNet):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)

        if isinstance(model, torchvision.models.efficientnet.EfficientNet):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

        if isinstance(model, torchvision.models.convnext.ConvNeXt):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

    if n_input_channels!=3:
        assert n_input_channels==1
        if isinstance(model, torchvision.models.resnet.ResNet):
            model.conv1 = sum_layer_weights(model.conv1)

        if isinstance(model, torchvision.models.vgg.VGG):
            model.features[0] = sum_layer_weights(model.features[0])
    
        if isinstance(model, torchvision.models.densenet.DenseNet):
            model.features[0] = sum_layer_weights(model.features[0])
    
        if isinstance(model, torchvision.models.efficientnet.EfficientNet):
            model.features[0][0] = sum_layer_weights(model.features[0][0])
    
        if isinstance(model, torchvision.models.convnext.ConvNeXt):
            model.features[0][0] = sum_layer_weights(model.features[0][0])

    return model



def get_model(name, pretrained=True, num_classes=1000, n_input_channels=3,
              z2_transition=0, group_equiv=None, mean_var_init=True,
              group_equiv_expand=False, **model_kwargs):
    
    # initial validation of inputs
    if pretrained:
        assert n_input_channels in [1, 3]
    
    if pretrained or mean_var_init:
        pretrained_model = get_architecture(name, pretrained=True, **model_kwargs)
        if pretrained == True and z2_transition==0:
            return adjust_layers_if_needed(pretrained_model, 1000, num_classes, n_input_channels)
        
    model = get_architecture(name, pretrained=False, **model_kwargs)
    
    if mean_var_init:
        initialize_model_mean_var(model, pretrained_model)
    
    # todo (substitute for a general function)
    if z2_transition>0:
        if pretrained:
            if group_equiv_expand:
                model = make_equivariant(pretrained_model, z2_transition, group_equiv_expand=True)
            else:
                print("Warning initializing pretrained model with diff architecture")
                print("This is the best approximation...")
                model = make_equivariant(pretrained_model, z2_transition, init_copy=True)
        else:
            model = make_equivariant(model, z2_transition, init_copy=False)

    return adjust_layers_if_needed(model, 1000, num_classes, n_input_channels)


def get_architecture(name, pretrained=True, **model_kwargs):
    admitted_archs = ["resnet", "vgg", "densenet", "efficientnet", "convnext"]
    assert any(name.startswith(arch) for arch in admitted_archs)
    model = getattr(torchvision.models, name)(pretrained=pretrained, **model_kwargs)
    return model

from torch import Tensor
from torch.nn import functional as torchF
from torchvision.models import DenseNet, ConvNeXt
def add_forward_with_representations(model):
    if isinstance(model, DenseNet):
        def forward_with_representations(self, x: Tensor) -> Tensor:
            features = self.features(x)
            out = torchF.relu(features, inplace=True)
            out = torchF.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            return self.classifier(out), out
    elif isinstance(model, ConvNeXt):
        def forward_with_representations(self, x: Tensor) -> Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            return self.classifier(x), torch.flatten(x, 1)
    
    else:
        raise(NotImplementedError("Unsupported architecture"))
        
    import types
    model.forward_with_representations = types.MethodType(forward_with_representations, model)
    return model

    
"""
    elif name.startswith("vgg"):
        model = getattr(torchvision.models, name)(pretrained=pretrained, num_classes=num_classes)

    elif name.startswith("densenet"):
        model = getattr(torchvision.models, name)(pretrained=pretrained, num_classes=num_classes)

    elif name.startswith("efficientnet"):
        model = getattr(torchvision.models, name)(pretrained=pretrained, num_classes=num_classes)
        
    elif name.startswith("convnext"):
        model = getattr(torchvision.models, name)(pretrained=pretrained, num_classes=num_classes)
        
        make_equivariant_convnext(model, z2_transition)

    else:
        raise(ValueError(f"Unrecognized model {name}"))

    model = adjust_layers_if_needed(model, temp_n_classes, num_classes, n_input_channels)
    return model
"""

if __name__ == "__main__":
    model = get_model("densenet121", num_classes=3, n_input_channels=1, z2_transition=6)

    """
    model = get_model("resnet50", num_classes=3, n_input_channels=1)
    print(model)
    
    model = get_model("vgg16", num_classes=3, n_input_channels=1)
    print(model)
    
    model = get_model("efficientnet_b4", num_classes=3, n_input_channels=1)
    print(model)

    model = get_model("densenet121", num_classes=3, n_input_channels=1)
    print(model)
    
    model = get_model("convnext_tiny", num_classes=3, n_input_channels=1)
    print(model)
    """
    