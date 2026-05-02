import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter
from model import ChessBoardResNet
from src.dataset import ChessDataset
from src.utils.fen_utils import encode_fen_to_indices

def evaluate_model(test_loader, model_path, device):
    """
    Evaluate the ResNet model on the test set and display metrics.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        model_path (str): Path to the saved model (.pth file).
        device (torch.device): Device to use (cuda or cpu).

    Returns:
        float: Global accuracy (percentage of correctly predicted squares).
        dict: Accuracy per piece class.
        np.ndarray: Confusion matrix (pieces only).
    """
    # --- 1. Load Model ---
    model = ChessBoardResNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode

    # --- 2. Initialize Metrics ---
    all_labels = []
    all_preds = []
    correct = 0
    total = 0

    # --- 3. Evaluation Loop ---
    with torch.no_grad():
        for images, fens in test_loader:
            images = images.to(device)
            labels = torch.stack([torch.tensor(encode_fen_to_indices(fen)) for fen in fens]).to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 2)  # (batch_size, 64)

            total += labels.size(0) * 64
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(predicted.cpu().numpy().flatten())

    # --- 4. Calculate Metrics ---
    # Global accuracy
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\n--- Global Accuracy: {accuracy:.2f}% ---\n")

    # Accuracy per piece
    piece_names = ['empty', 'P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k']
    label_counts = Counter(all_labels)
    print("--- Accuracy per Piece ---")
    for i, name in enumerate(piece_names):
        if label_counts[i] > 0:
            piece_correct = sum(1 for l, p in zip(all_labels, all_preds) if l == p == i)
            piece_accuracy = 100 * piece_correct / label_counts[i]
            print(f"{name:5s} (count={label_counts[i]:6d}): {piece_accuracy:.2f}%")

    # --- 5. Confusion Matrix (Pieces Only) ---
    # Exclude empty squares (index 0) - Filtre les paires (label, pred) ensemble
    filtered_pairs = [(l, p) for l, p in zip(all_labels, all_preds) if l != 0]
    if filtered_pairs:
        filtered_labels, filtered_preds = zip(*filtered_pairs)
        filtered_labels = list(filtered_labels)
        filtered_preds = list(filtered_preds)
    else:
        filtered_labels, filtered_preds = [], []

    piece_indices = list(range(1, 13))  # Exclude empty squares (index 0)
    piece_labels = [piece_names[i] for i in piece_indices]

    cm = confusion_matrix(filtered_labels, filtered_preds, labels=piece_indices)

    # --- 6. Plot Confusion Matrix Heatmap ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=piece_labels,
        yticklabels=piece_labels,
        cmap="Blues",
        linewidths=0.5
    )
    plt.title("Confusion Matrix (Pieces Only - No Empty Squares)", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()

    return accuracy, label_counts, cm

if __name__ == "__main__":
    # --- Set Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Define Paths ---
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "test"
    MODEL_DIR = BASE_DIR / "models" / "resnet"

    # --- Test Transformations ---
    test_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # --- Load Test Dataset ---
    test_dataset = ChessDataset(
        csv_path=str(DATA_DIR / "labels.csv"),
        img_dir=str(DATA_DIR / "images"),
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True
    )

    # --- Model Path ---
    model_path = str(MODEL_DIR / "chess_resnet50.pth")

    # --- Run Evaluation ---
    evaluate_model(test_loader, model_path, device)