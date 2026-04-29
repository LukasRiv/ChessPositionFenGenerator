import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ChessBoardResNet(nn.Module):
    """ResNet50-based model to predict the 64 squares of a chessboard.

    Uses a pre-trained ResNet50 backbone (trained on ImageNet) and replaces the final layers
    to adapt to the chessboard classification task (64 squares × 13 classes).

    Args:
        num_classes (int, optional): Number of classes per square (default: 13 for 12 pieces + empty).
        pretrained (bool, optional): Whether to use pre-trained weights on ImageNet (default: True).
    """

    def __init__(self, num_classes=13, pretrained=True):
        super(ChessBoardResNet, self).__init__()

        # --- Backbone: ResNet50 ---
        # Load ResNet50 with pre-trained weights from ImageNet (if pretrained=True)
        # This allows the model to leverage features learned from millions of images
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # Remove the original fully connected layer (2048 -> 1000 for ImageNet)
        # We will replace it with our custom layers for chessboard classification
        self.resnet.fc = nn.Identity()  # Replace fc with Identity to remove the original output layer

        # --- Custom Head ---
        # Global Average Pooling: Reduces spatial dimensions to 1x1 (2048 features)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom fully connected layers to adapt to 64×13 output
        # ResNet50's last convolutional layer outputs 2048 features
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),  # 2048 features → 1024
            nn.BatchNorm1d(1024),    # Batch Normalization for stability
            nn.ReLU(),              # Activation function
            nn.Dropout(0.5),        # Regularization to prevent overfitting
            nn.Linear(1024, 64 * num_classes)  # 1024 → 832 (64 squares × 13 classes)
        )

    def forward(self, x):
        """Forward pass of the ResNet50-based model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 400, 400).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 64, num_classes).
        """

        # --- Feature Extraction: ResNet50 Backbone ---
        # Initial convolution block (7x7 conv, 64 channels, stride=2, padding=3)
        x = self.resnet.conv1(x)      # (batch_size, 64, 200, 200) [400/2=200]
        x = self.resnet.bn1(x)        # Batch Normalization
        x = self.resnet.relu(x)       # ReLU activation
        x = self.resnet.maxpool(x)    # Max Pooling (3x3, stride=2) → (batch_size, 64, 100, 100) [200/2=100]

        # Residual blocks (each layer reduces spatial dimensions by 2)
        x = self.resnet.layer1(x)      # (batch_size, 256, 50, 50) [100/2=50]
        x = self.resnet.layer2(x)      # (batch_size, 512, 25, 25) [50/2=25]
        x = self.resnet.layer3(x)      # (batch_size, 1024, 13, 13) [25/2≈13]
        x = self.resnet.layer4(x)      # (batch_size, 2048, 13, 13) [keeps 13x13 spatial dims]

        # --- Custom Head ---
        # Global Average Pooling: Reduces (batch_size, 2048, 13, 13) → (batch_size, 2048, 1, 1)
        x = self.global_pool(x)

        # Flatten the features for the fully connected layers
        x = torch.flatten(x, 1)        # (batch_size, 2048)

        # Apply custom fully connected layers
        x = self.fc(x)                # (batch_size, 832)

        # Reshape to match the chessboard structure: (batch_size, 64, 13)
        return x.view(-1, 64, 13)