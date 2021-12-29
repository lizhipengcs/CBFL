#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import math
import torch.nn as nn

__all__ = ['CNNCifar', 'CNNMnist', 'PreResNet', 'PreBasicBlock', 'ModerateCNN']


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, num_channels=1, num_classes=10):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResnetCifar(nn.Module):
    def __init__(self, args):
        super(ResnetCifar, self).__init__()
        self.extractor = models.resnet18(pretrained=False)
        self.fflayer = nn.Sequential(nn.Linear(1000, args.num_classes))
        # self.extractor.weight_keys = [['fc1.weight', 'fc1.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]
        # self.weight_keys = [['fc1.weight', 'fc1.bias'],
        #                               ['fc2.weight', 'fc2.bias'],
        #                               ['fc3.weight', 'fc3.bias'],
        #                               ['conv2.weight', 'conv2.bias'],
        #                               ['conv1.weight', 'conv1.bias'],
        #                               ]
        # # self.register_buffer("weight_keys", [])

    def forward(self, x):
        x = self.extractor(x)
        x = self.fflayer(x)
        return F.log_softmax(x, dim=1)


class ResnetCifar(nn.Module):
    def __init__(self, args):
        super(ResnetCifar, self).__init__()
        self.extractor = models.resnet18(pretrained=False)
        self.fflayer = nn.Sequential(nn.Linear(1000, args.num_classes))

    def forward(self, x):
        x = self.extractor(x)
        x = self.fflayer(x)
        return F.log_softmax(x, dim=1)


# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------

def conv1x1(in_plane, out_plane, stride=1):
    """
    1x1 convolutional layer
    """
    return nn.Conv2d(in_plane, out_plane,
                     kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)


# both-preact | half-preact

class PreBasicBlock(nn.Module):
    """
    base module for PreResNet on small data sets
    """

    def __init__(self, in_plane, out_plane, stride=1, downsample=None, block_type="both_preact"):
        """
        init module and weights
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param stride: stride of convolutional layers, default 1
        :param downsample: down sample type for expand dimension of input feature maps, default None
        :param block_type: type of blocks, decide position of short cut, both-preact: short cut start from beginning
        of the first segment, half-preact: short cut start from the position between the first segment and the second
        one. default: both-preact
        """
        super(PreBasicBlock, self).__init__()
        self.name = block_type
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(in_plane, out_plane, stride)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_plane, out_plane)
        self.block_index = 0

    def forward(self, x):
        """
        forward procedure of residual module
        :param x: input feature maps
        :return: output feature maps
        """
        if self.name == "half_preact":
            x = self.bn1(x)
            x = self.relu1(x)
            residual = x
            x = self.conv1(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv2(x)
        elif self.name == "both_preact":
            residual = x
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv1(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv2(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = x + residual
        return out


class PreResNet(nn.Module):
    """
    define PreResNet on small data sets
    """

    def __init__(self, depth, wide_factor=1, num_classes=10, log_softmax=False):
        """
        init model and weights
        :param depth: depth of network
        :param wide_factor: wide factor for deciding width of network, default is 1
        :param num_classes: number of classes, related to labels. default 10
        """
        super(PreResNet, self).__init__()

        self.log_softmax = log_softmax
        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16 * wide_factor)
        self.layer1 = self._make_layer(PreBasicBlock, 16 * wide_factor, n)
        self.layer2 = self._make_layer(
            PreBasicBlock, 32 * wide_factor, n, stride=2)
        self.layer3 = self._make_layer(
            PreBasicBlock, 64 * wide_factor, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = linear(64 * wide_factor, num_classes)
        self._init_weight()

    def _init_weight(self):
        # init layer parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

    def _make_layer(self, block, out_plane, n_blocks, stride=1):
        """
        make residual blocks, including short cut and residual function
        :param block: type of basic block to build network
        :param out_plane: size of output plane
        :param n_blocks: number of blocks on every segment
        :param stride: stride of convolutional neural network, default 1
        :return: residual blocks
        """
        downsample = None
        if stride != 1 or self.in_plane != out_plane:
            downsample = conv1x1(self.in_plane, out_plane, stride=stride)

        layers = []
        layers.append(block(self.in_plane, out_plane, stride,
                            downsample, block_type="half_preact"))
        self.in_plane = out_plane
        for i in range(1, int(n_blocks)):
            layers.append(block(self.in_plane, out_plane))
        return nn.Sequential(*layers)

    # def train(self, mode=True):
    #         """
    #         Override the default train() to freeze the BN parameters
    #         """
    #         super(PreResNet, self).train(mode)
    #         # if self.freeze_bn:
    #         #     print("Freezing Mean/Var of BatchNorm2D.")
    #         #     if self.freeze_bn_affine:
    #         #         print("Freezing Weight/Bias of BatchNorm2D.")
    #         # if self.freeze_bn:
    #         for m in self.modules():
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m.eval()
    #             # if self.freeze_bn_affine:
    #             # m.weight.requires_grad = False
    #             # m.bias.requires_grad = False

    def forward(self, x):
        """
        forward procedure of model
        :param x: input feature maps
        :return: output feature maps
        """
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # if self. log_softmax:
        #     return F.log_softmax(out, dim=1)
        # else:
        return out


class ModerateCNN(nn.Module):
    def __init__(self):
        super(ModerateCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


if __name__ == '__main__':
    model = PreResNet(20)
    print(model)
    # for name, layer in model.named_modules():
    #     print(name)
    for name, module in model.named_modules():
        print(name, module)
    model.cuda()
    print(model(torch.randn((16,3,32,32)).cuda()))