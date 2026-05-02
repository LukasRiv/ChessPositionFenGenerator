import pandas as pd
import os

def split_labels(input_csv, train_csv, test_csv, train_files, test_files):
    """Splits the labels CSV into train and test CSVs based on provided file lists.

    Args:
        input_csv (str): Path to the input CSV file containing all labels.
        train_csv (str): Path where the training labels CSV will be saved.
        test_csv (str): Path where the test labels CSV will be saved.
        train_files (list): List of filenames for the training set.
        test_files (list): List of filenames for the test set.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
    """
    # Load the input CSV file
    df = pd.read_csv(input_csv)

    # Filter rows based on train/test files
    train_df = df[df['image_path'].apply(lambda x: os.path.basename(x) in train_files)]
    test_df = df[df['image_path'].apply(lambda x: os.path.basename(x) in test_files)]

    # Save to new CSV files
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

if __name__ == "__main__":
    # Paths to input and output CSV files
    input_csv = "../../data/raw/labels.csv"
    train_csv = "../../data/train/labels.csv"
    test_csv = "../../data/test/labels.csv"

    # Get list of files in train and test directories
    train_files = os.listdir("../../data/train/images")
    test_files = os.listdir("../../data/test/images")

    # Split the labels
    split_labels(input_csv, train_csv, test_csv, train_files, test_files)