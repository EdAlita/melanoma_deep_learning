import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import * 

class Discriminator(nn.Module):
    def __init__(self, image_size=64, num_classes=2):
        super().__init__()

        ndf = 512  #   # feature dimension of first layer of discriminator.
        layers = []

        # from rgb block
        layers = conv(layers, 3, ndf, 1, 1, 0, pixel=False)
        # intermediate block
        layers = conv(layers, ndf, ndf, 3, 1, 1, pixel=False)
        layers = conv(layers, ndf, ndf, 3, 1, 1, pixel=False)
        layers.append(nn.AvgPool2d(kernel_size=2))       # scale down by factor of 2.0
        # last block
        layers.append(minibatch_std_concat_layer())
        layers = conv(layers, ndf+1, ndf, 3, 1, 1, pixel=False)
        layers = conv(layers, ndf, ndf, 4, 1, 0, pixel=False)
        layers = linear(layers, ndf, 1)

        self.model = nn.Sequential(
            *layers
        )

        self.num_classes = num_classes

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        labels_one_hot = F.one_hot(labels.long(), num_classes=self.num_classes).float().squeeze()
        x = torch.cat([x, labels_one_hot], 1)
        print(x.shape)
        out = self.model(x)
        return out.squeeze()

class Generator(nn.Module):
    def __init__(self, image_size=64, num_classes=2):
        super().__init__()

        self.flag_tanh = True

        nz = 512 # input dimension of noise
        ngf = 512 # feature dimension of final layer of generator.
        layers = []

        # first block
        layers = deconv(layers, nz, ngf, 4, 1, 3)
        layers = deconv(layers, ngf, ngf, 4, 1, 3)

        # intermediate block
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))     # scale up by factor of 2.0
        layers = deconv(layers, ngf, ngf, 3, 1, 1)
        layers = deconv(layers, ngf, ngf, 3, 1, 1)

        # to_rgb_block
        layers = deconv(layers, ngf, 3, 1, 1, 0, only=True)
        if self.flag_tanh:  layers.append(nn.Tanh())

        self.model = nn.Sequential(
            *layers
        )
        
        self.im_size = image_size
        self.num_classes = num_classes

    def forward(self, z, labels):
        z = z.view(z.size(0), -1)
        labels_one_hot = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        x = torch.cat([z, labels_one_hot], 1)
        out = self.model(x)
        return out.view(x.size(0), 3, self.im_size, self.im_size)
