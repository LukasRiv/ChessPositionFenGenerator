import os

def extract_fen_from_filename(filename):
    """Extracts FEN from a filename by replacing hyphens with slashes and appends metadata.

    Args:
        filename (str): The name of the image file, e.g., "1b1B1b2-2pK2q1-4p1rB-7k-8-8-3B4-3rb3.png".

    Returns:
        str: The full FEN string with metadata, e.g., "1b1B1b2/2pK2q1/4p1rB/7k/8/8/3B4/3rb3 w - - 0 1".
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    # Replace hyphens with slashes to form the board part of FEN
    board_fen = name_without_ext.replace("-", "/")
    # Append standard metadata for a valid FEN
    full_fen = f"{board_fen} w - - 0 1"
    return full_fen