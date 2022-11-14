import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange
import torchvision
import random


""" *** Recognition Models *** """


class ResNet50(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=8,
                 pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        if in_channels == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                          padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNeXt50(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=8,
                 pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        if in_channels == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                          padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=8,
                 pretrained=True):
        super().__init__()
        self.densenet = torchvision.models.densenet121(pretrained=pretrained)
        if in_channels == 1:
            self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                          padding=(3, 3), bias=False)
        self.densenet.classifier = nn.Linear(1024, num_classes, bias=True)

    def forward(self, x):
        x = self.densenet(x)
        return x
