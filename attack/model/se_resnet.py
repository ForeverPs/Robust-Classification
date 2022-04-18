import torch.nn as nn
from model.resnet import ResNet
from model.robust_layer import *
from model.se_module import SELayer
from model.ml_decoder import MLDecoder


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Robust(nn.Module):
    def __init__(self, w=1e10, epsilon=1e-6, bit=8):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.bit = bit

    def forward(self, x):
        # scale = 2 ** b
        # out = torch.div(x, 1 / scale, rounding_mode='trunc') / scale
        # return torch.sin(x * self.w) * self.epsilon + torch.div(x, 1 / (2 ** self.bit), rounding_mode='trunc') / (2 ** self.bit)
        return torch.sin(x * self.w) * self.epsilon + x.detach()

class ResizedPadSeResNetML(nn.Module):
    def __init__(self, depth, num_classes, l=218, use_robust=False):
        super(ResizedPadSeResNetML, self).__init__()
        if depth == 18:
            model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        elif depth == 34:
            model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
        else:  # SeResNet50
            model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)

        channel_in = model.fc.in_features
        self.robust = ResizedPaddingLayer(l)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.ml_decoder_head = MLDecoder(num_classes, initial_num_features=channel_in)
        self.use_robust = use_robust
        self.robust_conf = Robust(bit=8)

    def forward(self, x):
        if self.use_robust:
            x = self.robust_conf(x)
        x = self.robust(x)
        feat = self.backbone(x)
        cls = self.ml_decoder_head(feat)
        return cls