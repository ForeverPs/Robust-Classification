import torch
import torch.nn as nn
from model.se_module import SELayer
from torchvision.models import ResNet


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


def se_resnet18(num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1000, pretrain_path=None):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrain_path is not None:
        model.load_state_dict(torch.load(pretrain_path))
    else:
        try:
            default_path = './pretrained_models/seresnet50-60a8950a85b2b.pkl'
            weight_dict = torch.load(default_path)  # pretrained weights
            weight_dict = {k: v for k, v in weight_dict.items() if 'fc' not in k}  # remove classification layer
            model_dict = model.state_dict()
            model_dict.update(weight_dict)
            model.load_state_dict(model_dict)
            print('Loading Default Weights on ImageNet...')
        except:
            print('Training From Scratch...')
    return model


def se_resnet101(num_classes=1000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class SeResNet(nn.Module):
    def __init__(self, depth, num_classes, dropout=0.2):
        super(SeResNet, self).__init__()
        if depth == 18:
            model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
            model.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
            model.avgpool = nn.AdaptiveAvgPool2d(1)

        channel_in = model.fc.in_features
        self.backbone = nn.Sequential(*list(model.children())[:-1])

        self.cls_head = nn.Sequential(
            self.cls_block(channel_in, 256, dropout),
            self.cls_block(256, 128, dropout),
            nn.Linear(128, num_classes))

    def cls_block(self, channel_in, channel_out, p):
        block = nn.Sequential(
            nn.Linear(channel_in, channel_out),
            nn.Dropout(p),
            nn.LeakyReLU(0.1),
        )
        return block

    def forward(self, x):
        feat = self.backbone(x).view(x.shape[0], -1)
        cls = self.cls_head(feat)
        return feat, cls


if __name__ == '__main__':
    model = se_resnet50(num_classes=20).eval()
    x = torch.randn(64, 3, 224, 224)
    y = model(x)
    print(y.shape)