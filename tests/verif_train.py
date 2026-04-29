import pandas as pd
from src.dataset import ChessDataset
from src.utils.fen_utils import encode_fen_to_indices

train_df = pd.read_csv("../data/train/labels.csv")
test_df = pd.read_csv("../data/test/labels.csv")

# Vérifie si des noms de fichiers sont communs
train_files = set(train_df["image_path"])  # Supposons que la colonne s'appelle "filename"
test_files = set(test_df["image_path"])
overlap = train_files & test_files

print(f"Nombre d'images en commun entre train et test: {len(overlap)}\n")
if overlap:
    print(f"Exemples d'images en double: {list(overlap)[:5]}")


train_dataset = ChessDataset(
    csv_path="../data/train/labels.csv",
    img_dir="../data/train/images",
    transform=None  # Pas de transformation pour le debug
)

# Affiche les 3 premiers échantillons
for i in range(3):
    image, fen = train_dataset[i]
    print(f"Image {i}: FEN = {fen}")



fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
indices = encode_fen_to_indices(fen)
print(f"\nFEN: {fen}")
print(f"Encoded indices: {indices}")
print(f"Unique values: {set(indices)}")  # Doit avoir au moins 3-4 valeurs différentes