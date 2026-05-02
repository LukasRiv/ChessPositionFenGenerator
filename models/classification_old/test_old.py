import torch
from torchvision import transforms
from PIL import Image
from classification_old.model_old import ChessBoardCNN
from src.utils.fen_utils import decode_indices_to_fen
from src.utils.visualization import imshow

def test_model(image_path, model_path, fen_real=None):
    """Test the trained model on a single image.

    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the saved model (.pth file).
        fen_real (str, optional): Real FEN of the image for comparison. Defaults to None.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ChessBoardCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode

    # Define transformations (must match training!)
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_indices = torch.max(outputs, 2)  # (1, 64)
        predicted_indices = predicted_indices.squeeze().cpu().tolist()  # Convert to list

    # Decode to FEN
    predicted_fen = decode_indices_to_fen(predicted_indices)
    print(f"Predicted FEN: {predicted_fen}")

    # Display image with real and predicted FEN
    if fen_real:
        transform_visu = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform_visu(image)
        imshow(image_tensor, fen=f"Real: {fen_real}\nPredicted: {predicted_fen}")
    else:
        transform_visu = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform_visu(image)
        imshow(image_tensor, fen=f"Predicted: {predicted_fen}")

if __name__ == "__main__":
    # Paths (adjust according to your project structure)
    image_path = "../../data/test/images/test.jpeg"  # Remplace par une image de ton dataset
    model_path = "chess_cnn.pth"  # Chemin relatif depuis models/classification/

    # Optionnel: Récupère le FEN réel depuis le CSV
    # (Si tu veux comparer avec la vérité terrain)
    import pandas as pd
    labels_df = pd.read_csv("../../data/test/labels.csv")
    # Trouve le FEN correspondant à l'image (suppose que le CSV a une colonne "filename" ou "image_path")
    # Exemple: fen_real = labels_df[labels_df["filename"] == "0001.png"]["fen"].iloc[0]
    fen_real = None  # à adapter selon ton CSV

    test_model(image_path, model_path, fen_real)