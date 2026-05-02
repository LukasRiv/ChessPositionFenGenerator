import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import ChessDataset
from classification_old.model_old import ChessBoardCNN
from src.utils.fen_utils import encode_fen_to_indices

def train_classification_model(train_loader, test_loader, num_epochs=10):
    """Trains the classification model to predict 64 squares of the chessboard.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.

    Returns:
        ChessBoardCNN: The trained model.
    """
    # --- 1. Initialisation ---
    # Check if CUDA (GPU) is available; otherwise, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")  # Affiche le device utilisé
    # Initialize the model and move it to the selected device (GPU/CPU)
    model = ChessBoardCNN().to(device)
    # Define the loss function: CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer: Adam with learning rate 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- 2. Training Loop ---
    for epoch in range(num_epochs):
        # Set the model to training mode (enables dropout, batch norm, etc.)
        model.train()

        # Iterate over batches of training data
        for images, fens in train_loader:
            # Move images to the selected device (GPU/CPU)
            images = images.to(device)
            # Convert FEN strings to tensor labels (batch_size, 64)
            labels = torch.stack([torch.tensor(encode_fen_to_indices(fen)) for fen in fens]).to(device)
            # Réinitialize gradients
            optimizer.zero_grad()
            # Forward pass: compute predictions for the current batch
            outputs = model(images)  # (batch_size, 64, 13)
            # Compute the loss: flatten outputs and labels for CrossEntropyLoss
            loss = criterion(outputs.view(-1, 13), labels.view(-1))
            # Backward pass: compute gradients of the loss w.r.t. model parameters
            loss.backward()
            # Update the model weights using the optimizer
            optimizer.step()

        # --- 3. Evaluation Loop ---
        model.eval()  # Mode évaluation (active le dropout, désactive BatchNorm)
        correct = 0   # Counter for correct predictions
        total = 0     # Counter for total number of squares (64 per image)

        # Disable gradient computation for evaluation (saves memory and time)
        with torch.no_grad():
            for images, fens in test_loader:
                images = images.to(device)
                labels = torch.stack([torch.tensor(encode_fen_to_indices(fen)) for fen in fens]).to(device)

                # Forward Pass: compute predictions
                outputs = model(images)
                # Get the predicted class for each square (argmax over the 13 classes)
                _, predicted = torch.max(outputs, 2)  # (batch_size, 64)
                ## Update counters: total squares and correct predictions
                total += labels.size(0) * 64
                correct += (predicted == labels).sum().item()

        # Calculate accuracy as a percentage
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.2f}%")

    # --- 4. Return the trained model ---
    return model

# --- 5. Execute script (if launched directly) ---
if __name__ == "__main__":
    from torchvision import transforms

    ## Define image transformations: (resizing + normalization)
    transform = transforms.Compose([
        transforms.Resize((400, 400)),          # Resize to 400x400
        transforms.ToTensor(),                  # Convert to PyTorch tensor (values in [0, 1])
        transforms.Normalize(                   # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Initialize training and test datasets with transformations
    train_dataset = ChessDataset(
        csv_path="../../data/train/labels.csv",
        img_dir="../../data/train/images",
        transform=transform
    )
    test_dataset = ChessDataset(
        csv_path="../../data/test/labels.csv",
        img_dir="../../data/test/images",
        transform=transform
    )

    # Create DataLoaders for batching and shuffling:
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,      # Shuffle data every epoch to avoid order bias
        pin_memory=True    # Enable faster data transfer to CUDA-enabled GPUs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True
    )

    # Train the model
    model = train_classification_model(train_loader, test_loader)

    # Save model weights
    torch.save(model.state_dict(), "chess_cnn.pth")