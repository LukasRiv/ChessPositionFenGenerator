import torch
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from model import ChessBoardCNN
from src.utils.fen_utils import decode_indices_to_fen

def test_model(image_path, model_path, device, fen_real=None):
    """Test the trained model on a single image.

    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the saved model (.pth file).
        device (torch.device): Device to use (cuda or cpu).
        fen_real (str, optional): Real FEN of the image for comparison. Defaults to None.
    """
    # Load the trained model
    model = ChessBoardCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode (disables dropout, etc.)

    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format

    # Define transformations (MUST match training!)
    transform = transforms.Compose([
        transforms.Resize((400, 400)),          # Resize to model input size
        transforms.ToTensor(),                  # Convert to tensor (values in [0, 1])
        transforms.Normalize(                   # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Add batch dimension (model expects (batch_size, 3, 400, 400))
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)  # (1, 64, 13)
        _, predicted_indices = torch.max(outputs, 2)  # (1, 64)
        predicted_indices = predicted_indices.squeeze().cpu().tolist()  # Convert to list of 64 integers
        predicted_fen = decode_indices_to_fen(predicted_indices)

    print(f"Predicted FEN: {predicted_fen}")

    # Display the image with real and predicted FEN
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    title = f"Predicted: {predicted_fen}"
    if fen_real:
        title += f"\nReal: {fen_real}"
    plt.title(title)
    plt.axis("off")
    plt.show()

    return predicted_fen

if __name__ == "__main__":
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define base directories
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "test" / "images"
    LABELS_CSV = BASE_DIR / "data" / "test" / "labels.csv"
    MODEL_DIR = BASE_DIR / "models" / "classification"

    # Example usage
    image_path = str(DATA_DIR / "test.jpeg")  # Replace with an actual image from your dataset
    model_path = str(MODEL_DIR / "chess_cnn.pth")

    # Optionally, retrieve the real FEN from the CSV
    # Assumes CSV has columns: "filename" and "fen"
    fen_real = None
    try:
        labels_df = pd.read_csv(str(BASE_DIR / "data" / "test" / "labels.csv"))
        image_name = Path(image_path).name
        if "filename" in labels_df.columns:
            fen_row = labels_df[labels_df["filename"] == image_name]
        elif "image_path" in labels_df.columns:
            fen_row = labels_df[labels_df["image_path"] == image_path]
        else:
            fen_row = None

        if fen_row is not None and not fen_row.empty:
            fen_real = fen_row["fen"].iloc[0]
    except Exception as e:
        print(f"Could not retrieve real FEN: {e}")

    test_model(image_path, model_path, device, fen_real)