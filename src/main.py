import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import ChessDataset
from src.utils.visualization import visualize_sample


print(f"PyTorch version: {torch.__version__}")          # Doit afficher 2.11.0+cu130
print(f"CUDA available: {torch.cuda.is_available()}")  # Doit afficher True
print(f"CUDA version: {torch.version.cuda}")            # Doit afficher 13.0
print(f"GPU: {torch.cuda.get_device_name(0)}")



# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Optionnel : redimensionne si besoin
    transforms.ToTensor(),          # Convertit en tenseur [0, 1]
    transforms.Normalize(          # Normalise avec ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Create datasets
train_dataset = ChessDataset(
    csv_path="../data/train/labels.csv",
    img_dir="../data/train/images",
    transform=transform
)
test_dataset = ChessDataset(
    csv_path="../data/test/labels.csv",
    img_dir="../data/test/images",
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Exemple d'utilisation
if __name__ == "__main__":
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    visualize_sample(train_dataset)
    # Tu peux maintenant itérer sur train_loader et test_loader pour l'entraînement