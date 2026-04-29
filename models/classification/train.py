import torch
import torch.nn as nn
import math
from collections import Counter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.dataset import ChessDataset
from classification.model import ChessBoardCNN
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


    all_train_labels = []
    for _, fens in train_loader:
        for fen in fens:
            all_train_labels.extend(encode_fen_to_indices(fen))
    class_counts = [0] * 13
    for label in all_train_labels:
        class_counts[label] += 1

    # Strategical Importance
    importance = {
        0: 0.75,  # Empty (index 0)
        1: 1.2,  # White Pawn (P, index 1)
        2: 1.2,  # Black Pawn (p, index 2)
        3: 1.2,  # White Knight (N, index 3)
        4: 1.5,  # Black Knight (n, index 4)
        5: 1.5,  # White Bishop (B, index 5)
        6: 1.2,  # Black Bishop (b, index 6)
        7: 1.5,  # White Rook (R, index 7)
        8: 1.2,  # Black Rook (r, index 8)
        9: 4.0,  # White Queen (Q, index 9)
        10: 4.0,  # Black Queen (q, index 10)
        11: 2.5,  # White King (K, index 11)
        12: 2.5  # Black King (k, index 12)
    }

    # Logarithmic ponderation + importance
    class_weights = []
    for i in range(13):
        if class_counts[i] == 0:
            class_weights.append(1.0)
        else:
            rarity = math.log(sum(class_counts) / class_counts[i])
            class_weights.append(rarity * importance[i])

    # Normalization
    mean_weight = sum(class_weights) / 13
    # Class Weight
    class_weights = [w / mean_weight for w in class_weights]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # Define the loss function: CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Define the optimizer: Adam with learning rate 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # Define the scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  # Reduce lr

    # --- 2. Training Loop ---
    for epoch in range(num_epochs):
        # Set the model to training mode (enables dropout, batch norm, etc.)
        model.train()
        epoch_loss = 0.0

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
            epoch_loss += loss.item()
            # Backward pass: compute gradients of the loss w.r.t. model parameters
            loss.backward()
            # Update the model weights using the optimizer
            optimizer.step()

        # Update scheduler
        scheduler.step(epoch_loss / len(train_loader))

        # --- 3. Evaluation Loop ---
        model.eval()  # Evaluation mode (activate dropout, de-activate BatchNorm)
        correct = 0   # Counter for correct predictions
        total = 0     # Counter for total number of squares (64 per image)
        all_labels = []
        all_preds = []

        # Disable gradient computation for evaluation (saves memory and time)
        with torch.no_grad():
            for images, fens in test_loader:
                images = images.to(device)
                labels = torch.stack([torch.tensor(encode_fen_to_indices(fen)) for fen in fens]).to(device)

                # Forward Pass: compute predictions
                outputs = model(images)
                # Get the predicted class for each square (argmax over the 13 classes)
                _, predicted = torch.max(outputs, 2)  # (batch_size, 64)
                # Update counters: total squares and correct predictions
                total += labels.size(0) * 64
                correct += (predicted == labels).sum().item()
                # Store labels and predicitons for class metrics
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(predicted.cpu().numpy().flatten())

        # Calculate accuracy as a percentage
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss/len(train_loader):.4f} | Test Accuracy: {accuracy:.2f}%")

        # Display Precision per Class
        piece_names = ['empty', 'P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k']
        label_counts = Counter(all_labels)
        for i, name in enumerate(piece_names):
            if label_counts[i] > 0:
                acc = 100 * sum(1 for l, p in zip(all_labels, all_preds) if l == p == i) / label_counts[i]
                print(f"{name:5s} (count={label_counts[i]:6d}): {acc:.2f}%")

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
        pin_memory=True,    # Enable faster data transfer to CUDA-enabled GPUs
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    # Train the model
    model = train_classification_model(train_loader, test_loader)

    # Save model weights
    torch.save(model.state_dict(), "chess_cnn.pth")