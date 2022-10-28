import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
from models.rot_equiv_layers import RotConv2D, RotBatchNorm2d, GroupPooling2d

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True, 
        group_pooled_features = False,
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        nf = 512
        #nf = 512 if not group_pooled_features else 128
        self.classifier = nn.Sequential(
            nn.Linear(nf * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, RotConv2D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, RotBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, group: str = "z2", first_in_channels = 3) -> nn.Sequential:
    assert group in ["z2", "p4"]
    first = True
    layers: List[nn.Module] = []
    in_channels = first_in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            if group == "p4":
                conv2d = RotConv2D(in_channels, v, kernel_size=3, padding=1, first=first)
                first=False
                if batch_norm:
                    layers += [conv2d, RotBatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    #if group=="p4":
    #    layers.append(GroupPooling2d())
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, group: str, in_channels: int, **kwargs: Any) -> VGG:
    group_pooled_features = True if group=="p4" else False
    if pretrained:
        kwargs['init_weights'] = False
    layers = make_layers(cfgs[cfg], batch_norm=batch_norm, group=group, first_in_channels=in_channels)
    model = VGG(layers, group_pooled_features=group_pooled_features, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, group: str = "z2", in_channels: int = 3, **kwargs: Any) -> VGG:
    return _vgg('vgg11', 'A', False, pretrained, progress, group, in_channels, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, group: str = "z2", in_channels: int = 3, **kwargs: Any) -> VGG:
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, group, in_channels, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, group: str = "z2", in_channels: int = 3, **kwargs: Any) -> VGG:
    return _vgg('vgg13', 'B', False, pretrained, progress, group, in_channels, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, group: str = "z2", in_channels: int = 3, **kwargs: Any) -> VGG:
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, group, in_channels, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, group: str = "z2", in_channels: int = 3, **kwargs: Any) -> VGG:
    return _vgg('vgg16', 'D', False, pretrained, progress, group, in_channels, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, group: str = "z2", in_channels: int = 3, **kwargs: Any) -> VGG:
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, group, in_channels, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, group: str = "z2", in_channels: int = 3, **kwargs: Any) -> VGG:
    return _vgg('vgg19', 'E', False, pretrained, progress, group, in_channels, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, group: str = "z2", in_channels: int = 3, **kwargs: Any) -> VGG:
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, group, in_channels, **kwargs)
