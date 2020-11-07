import os

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, noBN=False):
        super(BasicBlock, self).__init__()
        self.noBN = noBN
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print('noBN in basicBlock = ', self.noBN)
        outBN = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        outBN = outBN + residual
        outBN = self.relu(outBN)
        if self.noBN is False:
            return outBN
        else:
            out = out + residual
            return outBN, out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, noBN=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.noBN = noBN
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
        outBN = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        outBN = outBN + residual
        outBN = self.relu(outBN)
        if self.noBN is False:
            return outBN
        else:
            out = out + residual
            return outBN, out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, noBN=False):
        self.inplanes = 64
        self.noBN = noBN
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, noBN=self.noBN)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, noBN=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # print('blocks = ', blocks)
        if noBN is False:
            # print('with BN')
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        else:
            # print('no BN')
            if blocks > 2:
                # print('blocks > 2')
                for i in range(1, blocks-1):
                    layers.append(block(self.inplanes, planes))
                layers.append(block(self.inplanes, planes, noBN=True))
            else:
                # print('blocks <= 2')
                layers.append(block(self.inplanes, planes, noBN=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.noBN:
            conv_layer4BN, conv_layer4NBN = self.layer4(x)
        else:
            conv_layer4BN = self.layer4(x)
        x = self.avgpool(conv_layer4BN)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.noBN:
            return x, conv_layer4BN, conv_layer4NBN
        else:
            return x, conv_layer4BN


def resnet18(pretrain="places", noBN=True, **kwargs):
    if pretrain == "imagenet":
        num_classes = 1000
        state_dict = model_zoo.load_url(model_urls["resnet18"])
    elif pretrain == "places":
        num_classes = 365
        filename = "resnet18_places365.pth.tar"
        if not os.path.exists(filename):
            os.system("wget http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar -P .")
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model = ResNet(BasicBlock, [2, 2, 2, 2], noBN=noBN, num_classes=num_classes)
    model.load_state_dict(state_dict)
    return model

