import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.attention_weights = nn.Parameter(torch.randn(feature_dim))

    def forward(self, x):
        # Applying attention weights
        weights = F.softmax(self.attention_weights, dim=0)
        # Weighted sum of input features
        out = torch.mul(x, weights)
        return out

class CustomClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1=1024, hidden_size2=512, num_classes=2):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        #self.attention = Attention(hidden_size1) # Attention layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.attention(out) # Apply attention
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out
