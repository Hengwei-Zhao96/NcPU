import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_model.LaGAM.meta_layers import *

class PreActBlockMeta(MetaModule):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockMeta, self).__init__()
        self.bn1 = MetaBatchNorm2d(in_planes)
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MetaConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out
    
class PreActBottleneckMeta(MetaModule):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckMeta, self).__init__()
        self.bn1 = MetaBatchNorm2d(in_planes)
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MetaConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActResNetMeta(MetaModule):
    def __init__(self, block, num_blocks, nb_classes=10, dataset_name=None):
        super(PreActResNetMeta, self).__init__()
        self.in_planes = 64
        self.dataset_name = dataset_name

        if self.dataset_name == "abcd":
            in_channels = 6
        else:
            in_channels = 3

        self.conv1 = MetaConv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (MetaBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def ResNet18(nb_classes=2, dataset_name=None):
    return PreActResNetMeta(PreActBlockMeta, [2,2,2,2], nb_classes, dataset_name=dataset_name)

def ResNet34(nb_classes=2, dataset_name=None):
    return PreActResNetMeta(PreActBlockMeta, [3,4,6,3], nb_classes, dataset_name=dataset_name)

def ResNet50(nb_classes=2, dataset_name=None):
    return PreActResNetMeta(PreActBottleneckMeta, [3,4,6,3], nb_classes, dataset_name=dataset_name)

def ResNet101(nb_classes=2, dataset_name=None):
    return PreActResNetMeta(PreActBottleneckMeta, [3,4,23,3], nb_classes, dataset_name=dataset_name)

def ResNet152(nb_classes=2, dataset_name=None):
    return PreActResNetMeta(PreActBottleneckMeta, [3,8,36,3], nb_classes, dataset_name=dataset_name)


model_dict = {
    "resnet18": [ResNet18, 512],
    "resnet34": [ResNet34, 512],
    "resnet50": [ResNet50, 2048],
    "resnet101": [ResNet101, 2048],
}


class ResNetMeta(MetaModule):
    """backbone + projection head"""

    def __init__(self, name="resnet18", num_class=0, dataset_name=None):
        super(ResNetMeta, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(dataset_name=dataset_name)

        self.classifier = MetaLinear(dim_in, num_class)

        self.fc4 = MetaLinear(dim_in, dim_in)
        self.fc5 = MetaLinear(dim_in, 128)
        self.head = nn.Sequential(self.fc4, nn.ReLU(), self.fc5)

    def forward(self, x, flag_feature=False):
        out = x
        out = out + torch.zeros(1, dtype=out.dtype, device=out.device, requires_grad=True)
        out = self.encoder(out)
        logits = self.classifier(out)
        feat_cl = F.normalize(self.head(out), dim=1)
        if flag_feature:
            return logits, feat_cl
        else:
            return logits
