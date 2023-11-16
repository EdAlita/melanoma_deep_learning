import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, image_size=64, num_classes=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(image_size * image_size * 3 + num_classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.num_classes = num_classes

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        labels_one_hot = F.one_hot(labels.long(), num_classes=self.num_classes).float().squeeze()
        x = torch.cat([x, labels_one_hot], 1)
        out = self.model(x)
        return out.squeeze()

class Generator(nn.Module):
    def __init__(self, image_size=64, num_classes=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(100 + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size * image_size*3),
            nn.Tanh()
        )
        
        self.im_size = image_size
        self.num_classes = num_classes

    def forward(self, z, labels):
        z = z.view(z.size(0), -1)
        labels_one_hot = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        x = torch.cat([z, labels_one_hot], 1)
        out = self.model(x)
        return out.view(x.size(0), 3, self.im_size, self.im_size)
