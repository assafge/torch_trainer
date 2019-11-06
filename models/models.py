from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class EdofNet(nn.Module):

    def hidden_block(self, dilation):
        return nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(32))

    def __init__(self, max_dilation):
        super(EdofNet, self).__init__()
        self._stride = 1
        self.input_layer = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32,momentum=0.05))
        self.output_layer = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1))

        self.encode = np.arange(2, max_dilation+1)
        self.decode = np.flip(np.arange(2,max_dilation))
        self.dilations = np.append(self.encode, self.decode)

        self.hidden_blocks = nn.ModuleList()
        for j,i in enumerate(self.dilations):
            self.hidden_blocks.append(self.hidden_block(i))

    def forward(self, x):
        #y = F.relu(self.input_layer(x))
        y = F.leaky_relu(self.input_layer(x))

        for block in self.hidden_blocks:
            #y = F.relu(block(y))
            y = F.leaky_relu(block(y))
        y = self.output_layer(y)
        y = y + x
        return y


class Net(nn.Module):
    def __init__(self, target_mode, num_class=16, mode='classification', skip_layer=True):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(16, momentum=0.99)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 5,padding=2)
        self.batch_norm2 = nn.BatchNorm2d(36, momentum=0.99)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(36, 64, 5, padding=2)
        self.batch_norm3 = nn.BatchNorm2d(64, momentum=0.99)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128, momentum=0.99)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128,256,3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(256, momentum=0.99)
        self.pool5 = nn.MaxPool2d(2,2)
        self.conv6 = nn.Conv2d(256, num_class, 1)
        self.dense = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, num_class)
        self.upsampling = nn.ConvTranspose2d(in_channels=num_class, out_channels=num_class,kernel_size=64, stride=32)
        self.conv7_from_pool4 = nn.Conv2d(128, num_class, kernel_size=1)
        self.upsampling16 = nn.ConvTranspose2d(in_channels=num_class, out_channels=num_class, kernel_size=32, stride=16)
        self.conv9_from_pool3 = nn.Conv2d(64, num_class, kernel_size=1)
        self.upsampling8 = nn.ConvTranspose2d(in_channels=num_class, out_channels=num_class, kernel_size=16, stride=8)
        self.conv8_reg = nn.Conv2d(16, 1, 1, 1, bias=False)     # TODO - Remove! changed name only for checkpoint to match
        self.conv9_reg = nn.Conv2d(num_class, 1, 1, 1, bias=False)
        self.conv9_reg.weight.data[0,:,0,0] = torch.arange(num_class).float()
        for param in self.conv9_reg.parameters():
            param.requires_grad = False
        # self.conv8_reg.weight.requires_grad = False
        self.softmax = nn.Softmax(dim=1)
        # self.conv8_reg.weight[0,:,0,0] = torch.arange(16).float()
        self.num_class = num_class
        self.mode = mode
        self.skip_layer = skip_layer
        self.target_mode = target_mode

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool3(x)
        skip8 = x
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool4(x)
        skip = x
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = self.pool5(x)
        x = self.conv6(x)
        if self.mode == 'segmentation':
            x = self.upsampling(x)
            if self.skip_layer:
                skip_upsampled = self.upsampling16(self.conv7_from_pool4(skip))[:,:,8:-8,8:-8]
                skip_upsampled8 = self.upsampling8(self.conv9_from_pool3(skip8))[:,:,4:-4,4:-4]
                x = x[:,:,16:-16,16:-16]
                x = x + skip_upsampled + skip_upsampled8
            if self.target_mode == 'cont':
                x = self.softmax(x)
                x = self.conv9_reg(x)
                # x = ((x-1)/2)+1 # Fix from range 1-29 to range 1-15
                x = torch.squeeze(x)
        else:
            x = x.view(-1, self.num_class)
        return x

