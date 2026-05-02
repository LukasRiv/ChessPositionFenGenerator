import torch
import torch.nn as nn

class ChessBoardCNN(nn.Module):
    """CNN to predict the 64 squares of a chessboard.

    Args:
        num_classes (int, optional): Number of classes per square (default: 13).
    """

    def __init__(self, num_classes=13):
        super(ChessBoardCNN, self).__init__()

        # Convolution Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Pooling Layers
        self.pool = nn.MaxPool2d(2, 2)

        # Deep Layers
            # After 3 pooling layers : 400 -> 200 -> 100 -> 50
        self.fc1 = nn.Linear(in_features=64*50*50, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=64*num_classes)


    def forward(self, x):
        """Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 400, 400).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 64, num_classes).
        """

        # Convolution Layers + ReLU + Pooling
        x = torch.relu(self.conv1(x))   # (batch_size, 16, 400, 400)
        x = self.pool(x)                # (batch_size, 16, 200, 200)
        x = torch.relu(self.conv2(x))   # (batch_size, 32, 200, 200)
        x = self.pool(x)                # (batch_size, 32, 100, 100)
        x = torch.relu(self.conv3(x))   # (batch_size, 64, 100, 100)
        x = self.pool(x)                # (batch_size, 64, 50, 50)

        # Flattening
        x = torch.flatten(x, 1)         # (batch_size, 160000)

        # Deep Layers
        x = torch.relu(self.fc1(x))     # (batch_size, 1024)
        x = self.fc2(x)                 # (batch_size, 832)

        # Reshape for (batch_size, 64, 13)
        return x.view(-1, 64, 13)
