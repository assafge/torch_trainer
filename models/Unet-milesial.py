import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, dilated=False, padding_mode='reflect'):
        super().__init__()
        if dilated:
            dilation = 2
            padding = 3
        else:
            dilation = 1
            padding = 1
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation,
                      padding=padding, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation,
                      padding=padding, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Full assembly of the parts to form the complete network
class UNet(nn.Module):
    """
     dilated_input: use dilated convolution in input layer to match bayer pattern image
     bilinear: use normal convolutions to reduce the number of channels (up-sampling)
    """
    def __init__(self, n_channels, n_classes, scale_channels=64, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.scale_channels = scale_channels
        dilated_input = n_classes == 1

        self.inc = DoubleConv(n_channels, scale_channels, dilated_input)
        self.down1 = Down(scale_channels, scale_channels * 2)
        self.down2 = Down(scale_channels * 2, scale_channels * 4)
        self.down3 = Down(scale_channels * 4, scale_channels * 8)
        self.down4 = Down(scale_channels * 8, scale_channels * 8)
        self.up1 = Up(scale_channels * 16, scale_channels * 4, bilinear)
        self.up2 = Up(scale_channels * 8, scale_channels * 2, bilinear)
        self.up3 = Up(scale_channels * 4, scale_channels, bilinear)
        self.up4 = Up(scale_channels * 2, scale_channels, bilinear)
        self.outc = OutConv(scale_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.n_classes > 1:
            return logits
        else:
            return torch.squeeze(logits, dim=1)

        # if self.n_classes > 1:
        #     return F.softmax(x, dim=1)
        # else:
        #     return torch.sigmoid(x)

    def fine_tune(self, new_n_classes):
        print('changing n classes to', new_n_classes)
        self.n_classes = new_n_classes
        self.outc = OutConv(self.scale_channels, new_n_classes)
