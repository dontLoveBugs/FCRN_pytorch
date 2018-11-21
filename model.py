# -*- coding: utf-8 -*-
# @Time    : 2018/11/19 18:00
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class UpSample(nn.Module):

    def __init__(self, in_channel):
        super(UpSample, self).__init__()

        self.conv1_ = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=3)),
            ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ('relu', nn.ReLU(inplace=True))
            ]))

        self.conv1_ = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(2, 3))),
            ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ('relu', nn.ReLU(inplace=True))
            ]))

        self.conv1_ = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(3, 2))),
            ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ('relu', nn.ReLU(inplace=True))
            ]))

        self.conv1_ = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=2)),
            ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ('relu', nn.ReLU(inplace=True))
            ]))

        self.ps = nn.PixelShuffle(4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x1 = self.conv1_(x)
        x2 = self.conv2_(x)
        x3 = self.conv3_(x)
        x4 = self.conv4_(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        output = self.ps(x)
        output = self.relu(output)
        return output


class ResNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels // 2)

        self.up1 = UpSample(num_channels // 2)
        self.up2 = UpSample(num_channels // (2 ** 2))
        self.up3 = UpSample(num_channels // (2 ** 3))
        self.up4 = UpSample(num_channels // (2 ** 4))

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels // 32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)

        self.up1.apply(weights_init)
        self.up2.apply(weights_init)
        self.up3.apply(weights_init)
        self.up4.apply(weights_init)

        self.conv3.apply(weights_init)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # 上采样
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)


        x = self.conv3(x)
        x = self.bilinear(x)

        return x
