import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        # Resize mask to match the dimensions of the feature map
        mask = torch.nn.functional.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        # Apply convolution and batch normalization
        mask = self.conv(mask)
        mask = self.bn(mask)
        mask = self.sigmoid(mask)
        return x * mask
