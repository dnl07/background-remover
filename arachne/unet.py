import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        def forward(self, x):
            return self.conv(x)
        
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv1x1(x)

        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:])
        
        x = torch.cat([skip, x], dim=1)

        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 128)
        self.down4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up4 = UpConv(1024, 512)
        self.up3 = UpConv(512, 256)
        self.up2 = UpConv(256, 128)
        self.up1 = UpConv(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)

        c2 = self.down1(x)
        p2 = self.pool(c1)

        c3 = self.down1(x)
        p3 = self.pool(c1)

        c4 = self.down1(x)
        p4 = self.pool(c1)

        bn = self.bottle_neck(p4)

        u4 = self.up4(bn, c4)
        u3 = self.up3(u3, c3)
        u2 = self.up2(u2, c2)
        u1 = self.up1(u1, c1)

        out = self.final(u1)
        return out