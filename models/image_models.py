import torch
import torch.nn as nn
import torchvision.models as models
from typing import List

from config import IMAGE_MODELS

class CNNImageProcessor(nn.Module):
    def __init__(
        self,
        channels: List[int] = IMAGE_MODELS["cnn"]["channels"],
        kernel_size: int = IMAGE_MODELS["cnn"]["kernel_size"],
        dropout: float = IMAGE_MODELS["cnn"]["dropout"]
    ):
        super().__init__()
        
        layers = []
        in_channels = 3  # RGB input
        
        # Create convolutional blocks
        for out_channels in channels:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout)
            )
            layers.append(conv_block)
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output size after convolutions
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)  # Sample input
            x = self.conv_layers(x)
            self.feature_size = x.view(1, -1).size(1)
            
        # Add final fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
        Returns:
            Tensor of shape [batch_size, 256]
        """
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x

class ResNetImageProcessor(nn.Module):
    def __init__(
        self,
        model_name: str = IMAGE_MODELS["resnet"]["model_name"],
        pretrained: bool = IMAGE_MODELS["resnet"]["pretrained"]
    ):
        super().__init__()
        
        # Load pre-trained ResNet
        self.resnet = getattr(models, model_name)(pretrained=pretrained)
        
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add custom layers
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
        Returns:
            Tensor of shape [batch_size, 256]
        """
        # Get ResNet features
        x = self.resnet(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply final layers
        x = self.fc(x)
        
        return x 