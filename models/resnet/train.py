import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from model import ChessBoardResNet
from src.dataset import ChessDataset
from src.utils.fen_utils import encode_fen_to_indices

# --- Data Augmentation: Training Transformations ---
# Define image transformations for training data
# Note: No RandomResizedCrop or RandomRotation to avoid misaligning pieces with FEN labels
train_transform = transforms.Compose([
    transforms.Resize((400, 400)),          # Resize to fixed size (400x400)
    transforms.RandomHorizontalFlip(p=0.5), # Randomly flip horizontally (50% chance)
    transforms.RandomVerticalFlip(p=0.5),   # Randomly flip vertically (50% chance)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variations
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),  # Slight blur for robustness
    transforms.ToTensor(),                  # Convert PIL Image to PyTorch tensor (values in [0, 1])
    transforms.Normalize(                   # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],         # Mean for RGB channels
        std=[0.229, 0.224, 0.225]           # Std for RGB channels
    )
])

# --- Test Transformations ---
# Simpler transformations for evaluation (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((400, 400)),          # Resize to fixed size
    transforms.ToTensor(),                  # Convert to tensor
    transforms.Normalize(                   # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Data Loading ---
# Initialize training dataset with transformations
train_dataset = ChessDataset(
    csv_path="../../data/train/labels.csv",  # Path to training labels CSV
    img_dir="../../data/train/images",      # Path to training images
    transform=train_transform               # Apply training transformations
)

# Initialize test dataset with transformations
test_dataset = ChessDataset(
    csv_path="../../data/test/labels.csv",   # Path to test labels CSV
    img_dir="../../data/test/images",       # Path to test images
    transform=test_transform                # Apply test transformations
)

# Create DataLoaders for batching and shuffling
train_loader = DataLoader(
    train_dataset,
    batch_size=64,           # Number of images per batch
    shuffle=True,            # Shuffle data every epoch to avoid order bias
    pin_memory=True,         # Enable faster data transfer to CUDA-enabled GPUs
    num_workers=4,           # Number of subprocesses for data loading
    drop_last=True           # Drop last incomplete batch (if any)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,           # Number of images per batch
    shuffle=False,           # No shuffling for evaluation
    pin_memory=True,
    num_workers=4
)

# --- Post-processing Function ---
# Corrects invalid FEN strings (e.g., multiple kings of the same color)
def postprocess_predictions(predicted):
    """Post-processes model predictions to ensure valid FEN strings.

    Args:
        predicted (torch.Tensor): Model predictions of shape (batch_size, 64).

    Returns:
        torch.Tensor: Corrected predictions with valid FEN constraints.
    """
    predicted = predicted.clone()
    batch_size = predicted.shape[0]

    # Mapping from class indices to piece characters
    class_to_piece = {
        0: ' ', 1: 'P', 2: 'p', 3: 'N', 4: 'n', 5: 'B', 6: 'b',
        7: 'R', 8: 'r', 9: 'Q', 10: 'q', 11: 'K', 12: 'k'
    }

    for b in range(batch_size):
        board = predicted[b].reshape(8, 8)  # Reshape to 8x8 chessboard

        # Find positions of white and black kings
        white_king_pos = [(i, j) for i in range(8) for j in range(8) if board[i, j] == 11]
        black_king_pos = [(i, j) for i in range(8) for j in range(8) if board[i, j] == 12]

        # If multiple white kings, keep the first and set others to empty (0)
        if len(white_king_pos) > 1:
            for i, j in white_king_pos[1:]:
                board[i, j] = 0

        # If multiple black kings, keep the first and set others to empty (0)
        if len(black_king_pos) > 1:
            for i, j in black_king_pos[1:]:
                board[i, j] = 0

        predicted[b] = board.reshape(64)  # Flatten back to 64-length list

    return predicted

# --- Helper Function: Calculate Class Counts ---
def get_class_counts(train_loader):
    """Calculates the frequency of each class (piece type) in the training set.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.

    Returns:
        list: List of 13 integers representing counts for each class (0-12).
    """
    all_train_labels = []
    for _, fens in train_loader:
        for fen in fens:
            all_train_labels.extend(encode_fen_to_indices(fen))
    class_counts = [0] * 13
    for label in all_train_labels:
        class_counts[label] += 1
    return class_counts

# --- Training Function ---
def train_classification_model(train_loader, test_loader, num_epochs=20, class_counts=None):
    """Trains the ResNet50-based model to predict 64 squares of the chessboard.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        num_epochs (int, optional): Number of training epochs. Defaults to 20.
        class_counts (list, optional): Precomputed class counts. If None, they will be calculated.

    Returns:
        ChessBoardResNet: The trained model.
    """
    # --- 1. Initialization ---
    # Check if CUDA (GPU) is available; otherwise, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize the model and move it to the selected device (GPU/CPU)
    model = ChessBoardResNet().to(device)

    # Use precomputed class counts or calculate them
    if class_counts is None:
        class_counts = get_class_counts(train_loader)


    # --- 2. Model Setup: Freeze Layers ---
    # Freeze all layers of the ResNet50 backbone to retain pre-trained features
    for param in model.resnet.parameters():
        param.requires_grad = False  # Freeze all parameters

    # Unfreeze only the last convolutional layer (layer4) and custom head
    # This allows fine-tuning of high-level features while keeping low-level features fixed
    for param in model.resnet.layer4.parameters():
        param.requires_grad = True  # Unfreeze layer4 parameters
    for param in model.fc.parameters():
        param.requires_grad = True   # Unfreeze custom head parameters


    # --- 3. Class Weights: Handle Imbalanced Dataset ---
    # Define strategic importance for each piece type
    # Higher values = more penalty for misclassification of that piece
    importance = {
        0: 0.9,   # Empty square (less critical)
        1: 1.0,   # White Pawn
        2: 1.0,   # Black Pawn
        3: 1.1,   # White Knight
        4: 1.1,   # Black Knight
        5: 1.1,   # White Bishop
        6: 1.1,   # Black Bishop
        7: 1.1,   # White Rook
        8: 1.1,   # Black Rook
        9: 1.5,   # White Queen (more important)
        10: 1.5,  # Black Queen
        11: 2.0,  # White King (most important)
        12: 2.0   # Black King
    }

    # Calculate class weights: (log(1/frequency) * importance) for each class
    class_weights = []
    for i in range(13):
        if class_counts[i] == 0:
            class_weights.append(1.0)  # Default weight if class has no samples
        else:
            # Rarity factor: log of inverse frequency
            rarity = torch.log(torch.tensor(sum(class_counts) / class_counts[i], dtype=torch.float32))
            class_weights.append(rarity * importance[i])

    # Normalize weights so their mean is ~1.0 (for numerical stability)
    mean_weight = sum(class_weights) / 13
    class_weights = [w / mean_weight for w in class_weights]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # Define the loss function: CrossEntropyLoss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Define the optimizer: AdamW with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Learning rate scheduler: Cosine annealing with warm restarts
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    # Mixed precision training for faster computation and lower memory usage
    scaler = torch.amp.GradScaler(device='cuda')


    # --- 4. Training State ---
    best_accuracy = 0
    patience = 3  # Number of epochs to wait before early stopping if no improvement
    no_improvement = 0


    # --- 5. Training Loop ---
    for epoch in range(num_epochs):
        # Set the model to training mode (enables dropout, batch norm, etc.)
        model.train()
        epoch_loss = 0.0  # Track loss for the current epoch

        # Iterate over batches of training data
        for images, fens in train_loader:
            # Move images to the selected device (GPU/CPU)
            images = images.to(device)
            # Convert FEN strings to tensor labels (batch_size, 64)
            labels = torch.stack([torch.tensor(encode_fen_to_indices(fen)) for fen in fens]).to(device)
            # Reset gradients to zero
            optimizer.zero_grad()

            # Mixed precision training
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Forward pass: compute predictions for the current batch
                outputs = model(images)  # (batch_size, 64, 13)

                # Compute the loss: flatten outputs and labels for CrossEntropyLoss
                loss = criterion(outputs.view(-1, 13), labels.view(-1))

            # Scale loss for gradient scaling in mixed precision
            scaler.scale(loss).backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Update model weights
            scaler.step(optimizer)
            scaler.update()
            # Accumulate loss for the epoch
            epoch_loss += loss.item()

        # Update learning rate scheduler
        scheduler.step()


        # --- 6. Evaluation Loop ---
        # Set the model to evaluation mode (disables dropout, etc.)
        model.eval()
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
                outputs = model(images)  # (batch_size, 64, 13)
                # Get the predicted class for each square (argmax over the 13 classes)
                predicted = torch.max(outputs, 2)[1]  # (batch_size, 64)
                # Apply post-processing to correct invalid FENs
                predicted = postprocess_predictions(predicted)
                # Update counters: total squares and correct predictions
                total += labels.size(0) * 64
                correct += (predicted == labels).sum().item()
                # Store labels and predicitons for class metrics
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(predicted.cpu().numpy().flatten())

        # Calculate accuracy as a percentage
        accuracy = 100 * correct / total if total > 0 else 0
        # Calculate accuracy pr class
        piece_names = ['empty', 'P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k']
        label_counts = [0] * 13
        class_correct = [0] * 13
        for l, p in zip(all_labels, all_preds):
            label_counts[l] += 1
            if l == p:
                class_correct[l] += 1

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Test Accuracy: {accuracy:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Print class statistics
        print("--- Accuracy per Class ---")
        for i, name in enumerate(piece_names):
            if label_counts[i] > 0:
                acc = 100 * class_correct[i] / label_counts[i]
                print(f"{name:5s} (count={label_counts[i]:6d}): {acc:.2f}%")
        print()

        # Save best model if accuracy improved
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement = 0
            torch.save(model.state_dict(), "chess_resnet50.pth")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} (best accuracy: {best_accuracy:.2f}%)")
                break


    # --- 7. Return the trained model ---
    return model


# --- 8. Execute script (if launched directly) ---
if __name__ == "__main__":
    # Calculate class counts from training data
    class_counts = get_class_counts(train_loader)

    # Train the model
    model = train_classification_model(train_loader, test_loader, class_counts=class_counts)

    # Save final model weights
    torch.save(model.state_dict(), "chess_resnet50.pth")