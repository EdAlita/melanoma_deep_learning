import torch
import torch.nn as nn
import torchvision.models as models
from attention_block import AttentionBlock

class EfficientAttentionNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientAttentionNet, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)  # Load pre-trained EfficientNetB0
        self.att1 = AttentionBlock(in_channels=32)  # The channel number depends on the EfficientNet structure
        self.att2 = AttentionBlock(in_channels=56)
        self.att3 = AttentionBlock(in_channels=1280)
        self.fc = nn.Linear(1280, num_classes)  # Adjust the number of classes as needed

    def forward(self, image, mask):
        # EfficientNet features
        x = self.efficientnet.features(image)
        
        # Apply attention blocks after specific layers
        x = self.att1(x, mask)  # After first layer
        # ... (intermediate layers of EfficientNet)
        x = self.att2(x, mask)  # After fourth MBConv block
        # ... (more layers)
        x = self.att3(x, mask)  # Before Global Average Pooling layer

        # Pooling and classifier
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    
