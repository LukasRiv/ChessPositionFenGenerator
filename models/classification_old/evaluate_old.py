import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from classification_old.model_old import ChessBoardCNN
from src.dataset import ChessDataset
from src.utils.fen_utils import encode_fen_to_indices, decode_indices_to_fen
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(test_loader, model_path, device, num_samples=None):
    """Evaluate the model on the entire test set.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        model_path (str): Path to the saved model (.pth file).
        device (torch.device): Device to use (cuda or cpu).
        num_samples (int, optional): Number of samples to evaluate. If None, evaluate all. Defaults to None.

    Returns:
        float: Global accuracy (percentage of correctly predicted squares).
        list: List of (image_path, real_fen, predicted_fen) for visualization.
    """
    # Load model
    model = ChessBoardCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    results = []  # Pour stocker les résultats pour visualisation

    with torch.no_grad():
        for idx, (images, fens) in enumerate(test_loader):
            images = images.to(device)
            labels = torch.stack([torch.tensor(encode_fen_to_indices(fen)) for fen in fens]).to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 2)  # (batch_size, 64)

            # Calculate accuracy
            total += labels.size(0) * 64
            correct += (predicted == labels).sum().item()

            # Store results for visualization (optional)
            if num_samples and idx * test_loader.batch_size >= num_samples:
                break
            for i in range(len(images)):
                real_fen = fens[i]
                predicted_fen = decode_indices_to_fen(predicted[i].cpu().tolist())
                results.append((test_loader.dataset.labels_df.iloc[test_loader.batch_size * idx + i]["image_path"],
                                real_fen, predicted_fen))

    accuracy = 100 * correct / total
    print(f"Global accuracy: {accuracy:.2f}%")

    piece_names = ['empty', 'P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k']
    piece_correct = [0] * 13
    piece_total = [0] * 13

    with torch.no_grad():
        for images, fens in test_loader:
            # ... (code existant)
            for i in range(len(images)):
                for j in range(64):  # Pour chaque case
                    real_idx = labels[i, j].item()
                    pred_idx = predicted[i, j].item()
                    piece_total[real_idx] += 1
                    if real_idx == pred_idx:
                        piece_correct[real_idx] += 1

    # Afficher la précision par pièce
    for i, name in enumerate(piece_names):
        acc = 100 * piece_correct[i] / piece_total[i] if piece_total[i] > 0 else 0
        print(f"Accuracy per piece:")
        print(f"{name}: {acc:.2f}%")

    cm = confusion_matrix(labels.view(-1).cpu().numpy(), predicted.view(-1).cpu().numpy())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=piece_names, yticklabels=piece_names)
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy, results

def visualize_results(results, num_examples=5):
    """Visualize a few examples of predictions.

    Args:
        results (list): List of (image_path, real_fen, predicted_fen).
        num_examples (int): Number of examples to display.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    from src.utils.visualization import imshow

    for i in range(min(num_examples, len(results))):
        image_path, real_fen, predicted_fen = results[i]
        image = Image.open(image_path).convert("RGB")
        imshow(image, fen=f"Real: {real_fen}\nPredicted: {predicted_fen}")
        plt.show()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations (must match training!)
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = ChessDataset(
        csv_path="../../data/test/labels.csv",
        img_dir="../../data/test/images",
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    # Path to saved model
    model_path = "chess_cnn.pth"

    # Evaluate
    accuracy, results = evaluate_model(test_loader, model_path, device, num_samples=None)  # None = all samples

    # Visualize some examples
    visualize_results(results, num_examples=3)