import torch
from torch import nn

from models.rot_equiv_layers.base import RotConv2D, RotLayerNorm2d, RotLayerNorm, RotLinear, GroupBatchNorm2d

def custom_initialize_mean_var(layer, statistics):
    if layer.weight is not None:
        mean, std = torch.mean(statistics.weight).item(), torch.std(statistics.weight).item()
        nn.init.normal_(layer.weight, mean, std)

    if layer.bias is not None:
        mean, std = torch.mean(statistics.bias).item(), torch.std(statistics.bias).item()
        nn.init.normal_(layer.bias, mean, std)

def custom_initialize_copy(layer, statistics, first, group_equiv_expand=False):
    s_old = statistics.weight.size(0)
    s_new = layer.weight.size(0)
    if not group_equiv_expand:
        assert s_old//4 == s_new
    else:
        assert s_old == s_new
    
    if first:
        with torch.no_grad():
            layer.weight.copy_(statistics.weight[0:s_new])
            if layer.bias is not None:
                layer.bias.copy_(statistics.bias[0:s_new])
            if hasattr(layer, "running_mean"):
                layer.running_mean.copy_(statistics.running_mean[0:s_new])
                layer.running_var.copy_(statistics.running_var[0:s_new])
    else:
        s2 = statistics.weight.size(1)
        with torch.no_grad():
            layer.weight.fill_(0.)
            layer.weight[0:s_new, 0:s2].copy_(statistics.weight[0:s_new])
            if layer.bias is not None:
                layer.bias.copy_(statistics.bias[0:s_new])

def to_group_conv(layer, first=False, init_copy=True,
                  group_equiv_expand=False):
    n_filters, in_channels, k1, k2 = layer.weight.shape

    if group_equiv_expand:
        n_filters*=4
        if not first:
            in_channels *= 4
    new = RotConv2D(in_channels,
                    n_filters,
                    (k1, k2),
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=True if layer.bias is not None else False,
                    groups=layer.groups,
                    first=first)
    if init_copy or group_equiv_expand:
        custom_initialize_mean_var(new, layer)
        custom_initialize_copy(new, layer, first, group_equiv_expand=group_equiv_expand)
    else:
        custom_initialize_mean_var(new, layer)
    return new

def general_to_group_layernorm(l, class_name, init_copy=True):
    if l.weight is not None:
        device = l.weight.device
        dtype = l.weight.dtype
    else:
        device = dtype = None
    new = class_name(l.normalized_shape[0],
                     eps=l.eps,
                     elementwise_affine=l.elementwise_affine,
                     device=device, dtype=dtype, n_groups=4)
    if init_copy:
        custom_initialize_mean_var(new, l)
        custom_initialize_copy(new, l, True)
    else:
        custom_initialize_mean_var(new, l)
    return new

def to_group_layernorm2d(*args, **kwargs):
    return general_to_group_layernorm(*args, class_name=RotLayerNorm2d, **kwargs)

def to_group_layernorm(*args, **kwargs):
    return general_to_group_layernorm(*args, class_name=RotLayerNorm, **kwargs)


def to_group_bnorm(l, init_copy=True,
                  group_equiv_expand=False):
    if group_equiv_expand:
        n_feat = l.num_features * 4
    else:
        n_feat = l.num_features
    new = GroupBatchNorm2d(num_features=n_feat,
                           eps=l.eps,
                           momentum=l.momentum,
                           affine=l.affine,
                           track_running_stats=l.track_running_stats,
                           equiv_groups=4)
    if init_copy or group_equiv_expand:
        custom_initialize_mean_var(new, l)
        custom_initialize_copy(new, l, True, group_equiv_expand=group_equiv_expand)
    else:
        custom_initialize_mean_var(new, l)
    return new


def to_group_linear(l, init_copy=True):
    new = RotLinear(l.in_features,
                    l.out_features,
                    bias=l.bias is not None,
                    device=l.weight.device,
                    dtype=l.weight.dtype)
    if init_copy:
        custom_initialize_mean_var(new, l)
        custom_initialize_copy(new, l, False)
    else:
        custom_initialize_mean_var(new, l)
    return new

class custom_CNBlock(nn.Module):
    def __init__(self, dim, block, stochastic_depth):
        super().__init__()
        self.block = block
        self.stochastic_depth = stochastic_depth
        self.layer_scale = nn.Parameter(torch.ones(dim // 4, 1, 1))

    def transform_scale(self, scale):
        return torch.tile(scale, (4, 1, 1))

    def forward(self, input):
        result = self.transform_scale(self.layer_scale) * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result
    
    
def to_group_CNBlock(block, init_copy=True):
    new = custom_CNBlock(block.layer_scale.size(0),
                         block.block,
                         block.stochastic_depth)
    with torch.no_grad():
        new.layer_scale.copy_(block.layer_scale[0:block.layer_scale.size(0)//4])
    return new

def group_cat(features):
    b,_,w,h = features[0].shape
    features = [f.reshape([b, 4, -1, w, h]) for f in features]
    features = torch.cat(features, 2)
    features = features.reshape([b, -1, w, h])
    return features

def forward(self, init_features):
    features = [init_features]
    for name, layer in self.items():
        new_features = layer(group_cat(features))
        features.append(new_features)
    return group_cat(features)
