"""
Copyright (C) 2018  Axel Davy
Copyright (C) 2018  Yiqi Yan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


This is a modified version derived from
https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py
"""

import torch
import torch.nn as nn


class ModifiedDnCNN(nn.Module):
    def __init__(self, input_channels, output_channels, nlconv_features, nlconv_layers, dnnconv_features, dnnconv_layers):
        super(ModifiedDnCNN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.nlconv_features = nlconv_features
        self.nlconv_layers = nlconv_layers
        self.dnnconv_features = dnnconv_features
        self.dnnconv_layers = dnnconv_layers

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_channels,\
                                out_channels=self.nlconv_features,\
                                kernel_size=1,\
                                padding=0,\
                                bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.nlconv_layers-1):
            layers.append(nn.Conv2d(in_channels=self.nlconv_features,\
                                    out_channels=self.nlconv_features,\
                                    kernel_size=1,\
                                    padding=0,\
                                    bias=True))
            layers.append(nn.ReLU(inplace=True))
        # Shorter DnCNN
        layers.append(nn.Conv2d(in_channels=self.nlconv_features,\
                                out_channels=self.dnnconv_features,\
                                kernel_size=3,\
                                padding=1,\
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.dnnconv_layers-2):
            layers.append(nn.Conv2d(in_channels=self.dnnconv_features,\
                                    out_channels=self.dnnconv_features,\
                                    kernel_size=3,\
                                    padding=1,\
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.dnnconv_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.dnnconv_features,\
                                out_channels=self.output_channels,\
                                kernel_size=3,\
                                padding=1,\
                                bias=False))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        out = self.net(x)
        return out
