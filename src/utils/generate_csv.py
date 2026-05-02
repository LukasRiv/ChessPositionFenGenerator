import os
import pandas as pd
from fen_extractor import extract_fen_from_filename

def generate_fen_csv(image_dir, output_csv_path):
    """Generates a CSV file mapping image paths to their FEN codes.

    Args:
        image_dir (str): Directory containing chess position images.
        output_csv_path (str): Path where the CSV file will be saved.

    Raises:
        FileNotFoundError: If no images are found in the specified directory.
    """
    # List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    data = []
    for filename in image_files:
        fen = extract_fen_from_filename(filename)
        image_path = os.path.join(image_dir, filename)
        data.append({"image_path": image_path, "fen": fen})
        print(f"Extracted FEN for {filename}: {fen}")  # Debug print

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"CSV generated with {len(df)} entries: {output_csv_path}")

if __name__ == "__main__":
    # Relative paths from src/utils/ to data directories
    image_dir = "../../data/raw/images"
    output_csv_path = "../../data/raw/labels.csv"
    generate_fen_csv(image_dir, output_csv_path)