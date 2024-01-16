import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class CustomResNetModel(nn.Module):
    def __init__(self):
        super(CustomResNetModel, self).__init__()

        # Load the ResNet18 model with pretrained weights
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features

        # Replace the fully connected layers
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # Output layer for regression
        )

    def forward(self, x):
        return self.resnet(x)