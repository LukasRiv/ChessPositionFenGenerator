import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ChessDataset(Dataset):
    """PyTorch Dataset for loading chess images and their FEN codes.

    Args:
        csv_path (str): Path to the CSV file containing image paths and FEN codes.
        img_dir (str): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, csv_path, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_path)
        self.transform = transform

        # Verify CSV has required columns
        if not all(col in self.labels_df.columns for col in ["image_path", "fen"]):
            raise ValueError("CSV must contain 'image_path' and 'fen' columns.")

        # Update image paths to be absolute if they are not already
        self.labels_df["image_path"] = self.labels_df["image_path"].apply(
            lambda x: os.path.join(img_dir, os.path.basename(x)) if not os.path.isabs(x) else x
        )

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels_df)

    def __getitem__(self, idx):
        """Loads an image and its FEN code given an index.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            tuple: (image, fen) where image is a PyTorch tensor and fen is a string.
        """
        # Get image path and FEN
        img_path = self.labels_df.iloc[idx]["image_path"]
        fen = self.labels_df.iloc[idx]["fen"]

        # Load image with PIL
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, fen

