import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from torchvision import transforms

def imshow(img, fen=None, title=None):
    """Displays a chess image with its FEN code (if provided).

    Args:
        img (PIL.Image or np.ndarray or torch.Tensor): The image to display. Can be a PIL Image,
            a numpy array, or a PyTorch tensor.
        fen (str, optional): The FEN string associated with the image. If provided, it will be
            displayed as the title.
        title (str, optional): Custom title for the plot. If not provided and fen is given, fen will be used.

    Returns:
        None: Displays the image using matplotlib.
    """
    # Convert PIL Image to numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Convert PyTorch tensor to numpy array
    if hasattr(img, 'numpy'):
        img = img.numpy()

    # Convert (C, H, W) to (H, W, C) for matplotlib
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # Denormalize if the image was normalized with ImageNet stats
    if img.min() < 0 or img.max() > 1:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean  # Reverse normalization: (img * std) + mean
        img = np.clip(img, 0, 1)  # Ensure values are in the valid range [0, 1]

    # Display the image
    plt.imshow(img)
    if fen:
        plt.title(fen if title is None else title)
    elif title:
        plt.title(title)
    plt.axis('off')  # Hide axes
    plt.show()

def visualize_sample(dataset, idx=None):
    """Visualizes a random sample from the dataset (image + FEN).

    Args:
        dataset (ChessDataset): An instance of ChessDataset.
        idx (int, optional): Index of the sample to visualize. If None, a random sample is chosen.
    """
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)  # Pick a random index

    image, fen = dataset[idx]
    print(f"FEN: {fen}")  # Print FEN to console
    imshow(image, fen=fen)