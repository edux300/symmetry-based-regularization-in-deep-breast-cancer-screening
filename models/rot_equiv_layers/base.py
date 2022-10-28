import torch
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from torch.nn import Module, init, Parameter
import math

"""
Batch normalization
"""
class GroupBatchNorm2d(torch.nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, equiv_groups=4):
        assert num_features%4==0
        super(GroupBatchNorm2d, self).__init__(
            num_features//equiv_groups, eps, momentum, affine, track_running_stats)
        self.equiv_groups = equiv_groups

    def forward(self, input):
        s = input.shape
        reshaped_input = input.view([s[0], self.equiv_groups, s[1]//self.equiv_groups, s[2], s[3]])
        reshaped_input = reshaped_input.transpose(1, 2)
        reshaped_input = super(GroupBatchNorm2d, self).forward(reshaped_input)
        reshaped_input = reshaped_input.transpose(1, 2)
        reshaped_input = reshaped_input.reshape([s[0], s[1], s[2], s[3]])
        return reshaped_input

class RotBatchNorm2d(GroupBatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(RotBatchNorm2d, self).__init__(num_features, eps, momentum,
                                             affine, track_running_stats,
                                             equiv_groups=4)

class P4MBatchNorm2d(GroupBatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(P4MBatchNorm2d, self).__init__(num_features, eps, momentum,
                                             affine, track_running_stats,
                                             equiv_groups=8)

class GroupPooling2d(nn.Module):
    def __init__(self, equiv_groups=4):
        super(GroupPooling2d, self).__init__()
        self.equiv_groups = equiv_groups

    def forward(self, input):
        s=input.shape
        input = input.view([s[0], self.equiv_groups, s[1]//self.equiv_groups, s[2], s[3]])
        return torch.max(input, dim=1)[0]

import numbers
class RotLayerNorm2d(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 device=None, dtype=None, n_groups=4):
        
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape//4,)  # type: ignore[assignment]
        else:
            raise(ValueError("This has not been tested"))
        super(RotLayerNorm2d, self).__init__(normalized_shape=normalized_shape,
                                             eps=eps,
                                             elementwise_affine=elementwise_affine,
                                             device=device,
                                             dtype=dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        original_shape = x.shape
        x = x.reshape(-1, 4, self.normalized_shape[0])
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.reshape(original_shape)
        x = x.permute(0, 3, 1, 2)
        return x

class RotLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 device=None, dtype=None, n_groups=4):
        
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape//4,)  # type: ignore[assignment]
        else:
            raise(ValueError("This has not been tested"))
        super(RotLayerNorm, self).__init__(normalized_shape=normalized_shape,
                                           eps=eps,
                                           elementwise_affine=elementwise_affine,
                                           device=device,
                                           dtype=dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        x = x.reshape(-1, 4, self.normalized_shape[0])
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.reshape(original_shape)
        return x


class RotLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(RotLinear, self).__init__(in_features,
                                        out_features // 4,
                                        bias=bias,
                                        device=device,
                                        dtype=dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        tw = self.transform_weights(self.weight)
        tb = self.bias.repeat(4) if self.bias is not None else None
        return F.linear(x, tw, tb)
    
    def transform_weights(self, w):
        s = self.in_features // 4
        return torch.cat([torch.roll(w, i * s, dims=1) for i in range(4)]).contiguous()        

def t_weights(w, first=False, equiv_groups=4):
    if equiv_groups == 4:
        if first:
            w = torch.cat([torch.rot90(w, i, dims=(2,3)).contiguous() for i in range(4)], 0).contiguous()
        else:
            assert w.shape[1]%4==0
            s = w.shape[1] // 4
            w = torch.cat([torch.roll(torch.rot90(w, i, dims=(2,3)).contiguous(), i*s, dims=1).contiguous() for i in range(4)]).contiguous()
        return w
    elif equiv_groups == 8:
        if first:
            l1 = [torch.rot90(w, i, dims=(2,3)).contiguous() for i in range(4)]
            l2 = [torch.flip(torch.rot90(w, i, dims=(2,3)),dims=(2,3)).contiguous() for i in range(4)]
            l = [w for pair in zip(l1, l2) for w in pair]
            w = torch.cat(l, 0).contiguous()
        else:
            assert w.shape[1]%8==0
            s = w.shape[1] // 8
            l1 = [torch.roll(torch.rot90(w, i, dims=(2,3)).contiguous(), i*s*2, dims=1).contiguous() for i in range(4)]
            l2 = [torch.roll(torch.rot90(torch.flip(w, dims=(2,3)), i, dims=(2,3)).contiguous(), i*s*2+1, dims=1).contiguous() for i in range(4)]
            l = [w for pair in zip(l1, l2) for w in pair]
            w = torch.cat(l, 0).contiguous()
        return w

class GroupConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, groups=1, first=False, equiv_groups=4):
        super(GroupConv2D, self).__init__()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.first = first
        self.groups=groups
        self.equiv_groups = equiv_groups

        self.weight = Parameter(torch.Tensor(
            out_channels // self.equiv_groups, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels // self.equiv_groups))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        tw = t_weights(self.weight, self.first, self.equiv_groups)
        if self.bias is not None:
            tb = self.bias.repeat(self.equiv_groups)
            y = F.conv2d(input, weight=tw, bias=tb, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            y = F.conv2d(input, weight=tw, bias=None, stride=self.stride, padding=self.padding, groups=self.groups)
        return y

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, first={first}, equiv_groups={equiv_groups}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class RotConv2D(GroupConv2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, groups=1, first=False):
        super(RotConv2D, self).__init__(in_channels, out_channels, kernel_size,
                                        stride, padding, bias, groups=groups, first=first, equiv_groups=4)

class P4MConv2D(GroupConv2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, first=False):
        raise("Implementation needs to be checked")
        super(P4MConv2D, self).__init__(in_channels, out_channels, kernel_size,
                                        stride, padding, bias, first, equiv_groups=8)

if __name__ == "__main__":
    n1 = 10
    n2 = 20
    img = torch.rand(1, 1, 50, 50)
    timg = torch.rot90(img, 1, (2, 3))
    w1 = torch.rand(n1, 1, 3, 3)
    tw1 = t_weights(w1, True, 4)
    out = torch.conv2d(img, tw1)
    tout = torch.conv2d(timg, tw1)
    print("P4 first layer:", ((out-torch.roll(torch.rot90(tout, -1, (2,3)), -n1, 1))**2).mean())

    w2 = torch.rand(n2, n1*4, 3, 3)
    tw2 = t_weights(w2, False, 4)
    out = torch.conv2d(out, tw2)
    tout = torch.conv2d(tout, tw2)
    print("P4 second layer:", ((out-torch.roll(torch.rot90(tout, -1, (2,3)), -n2, 1))**2).mean())

    n1 = 10
    n2 = 20
    img = torch.rand(1, 1, 50, 50)
    timg = torch.rot90(img, 1, (2, 3))
    w1 = torch.rand(n1, 1, 3, 3)
    tw1 = t_weights(w1, True, 8)
    out = torch.conv2d(img, tw1)
    tout = torch.conv2d(timg, tw1)
    print("P4M first layer (rotation):", ((out-torch.roll(torch.rot90(tout, -1, (2,3)), -n1*2, 1))**2).mean())

    n1 = 10
    n2 = 20
    img = torch.rand(1, 1, 50, 50)
    timg = torch.flip(img, (2, 3))
    w1 = torch.rand(n1, 1, 3, 3)
    tw1 = t_weights(w1, True, 8)
    out = torch.conv2d(img, tw1)
    tout = torch.conv2d(timg, tw1)
    aux = torch.flip(tout, (2,3))
    print("P4M first layer (mirror):", ((out-torch.roll(aux, -n1, 1))**2).mean())
    print(((out-torch.roll(aux, -n1, 1))**2).mean(axis=(0, 2,3))< 0.0001)







