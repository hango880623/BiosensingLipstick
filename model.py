import torch.nn as nn
import torch
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
    
# Define a small CNN model
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(32 * 32 * 32, num_classes)  # Adjusted input size based on your image dimensions

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x