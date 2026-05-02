import os
import shutil
import random
import pandas as pd

def split_dataset(input_dir, train_dir, test_dir, train_ratio=0.8, seed=42):
    """Split the dataset into train and test directories.

    Args:
        input_dir (str): Directory containing all images.
        train_dir (str): Directory to save training images.
        test_dir (str): Directory to save test images.
        train_ratio (float): Ratio of images to use for training. Defaults to 0.8.
        seed (int): Random seed for reproducibility. Defaults to 42.
    """
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(seed)
    random.shuffle(image_files)

    # Split into train and test
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]

    # Copy files to train and test directories
    for filename in train_files:
        shutil.copy(
            os.path.join(input_dir, filename),
            os.path.join(train_dir, filename)
        )
    for filename in test_files:
        shutil.copy(
            os.path.join(input_dir, filename),
            os.path.join(test_dir, filename)
        )

    print(f"Train: {len(train_files)} images -> {train_dir}")
    print(f"Test: {len(test_files)} images -> {test_dir}")

if __name__ == "__main__":
    input_dir = "../../data/raw/images"
    train_dir = "../../data/train/images"
    test_dir = "../../data/test/images"
    split_dataset(input_dir, train_dir, test_dir, train_ratio=0.8)