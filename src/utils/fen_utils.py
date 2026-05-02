# Mapping for classification model (13 classes)
PIECE_TO_INDEX = {
    ' ': 0,    # Empty square
    'P': 1, 'p': 2,  # Pawns
    'N': 3, 'n': 4,  # Knights
    'B': 5, 'b': 6,  # Bishops
    'R': 7, 'r': 8,  # Rooks
    'Q': 9, 'q': 10, # Queens
    'K': 11, 'k': 12 # Kings
}
INDEX_TO_PIECE = {v: k for k, v in PIECE_TO_INDEX.items()}

# Special tokens for Seq2Seq
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
PAD_INDEX = 0

def encode_fen_to_indices(fen):
    """Encodes a FEN string into a list of 64 indices for the classification model.

    Args:
        fen (str): The FEN string to encode, e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1".

    Returns:
        list: A list of 64 integers representing the chess board squares.
    """
    # Split FEN into board part (ignore metadata for now)
    board_part = fen.split()[0]
    rows = board_part.split('/')
    indices = []
    for row in rows:
        for char in row:
            if char.isdigit():
                indices.extend([0] * int(char))  # Empty squares
            else:
                indices.append(PIECE_TO_INDEX[char])
    assert len(indices) == 64, f"Expected 64 squares, got {len(indices)}"
    return indices

def decode_indices_to_fen(indices):
    """Decodes a list of 64 indices into a FEN string for the classification model.

    Args:
        indices (list): A list of 64 integers representing the chess board squares.

    Returns:
        str: The FEN string with default metadata, e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1".
    """
    pieces = [INDEX_TO_PIECE[i] for i in indices]
    fen_parts = []
    for i in range(8):
        row = pieces[i*8 : (i+1)*8]
        fen_row = []
        empty_count = 0
        for piece in row:
            if piece == ' ':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row.append(str(empty_count))
                    empty_count = 0
                fen_row.append(piece)
        if empty_count > 0:
            fen_row.append(str(empty_count))
        fen_parts.append(''.join(fen_row))
    board_fen = '/'.join(fen_parts)
    return f"{board_fen} w - - 0 1"

def fen_to_indices(fen, vocab):
    """Encode a full FEN string into a list of indices for the Seq2Seq model.

    Args:
        fen (str): The FEN string to encode.
        vocab (str): The vocabulary string containing all possible characters.

    Returns:
        list: A list of integer indices, starting with SOS_TOKEN and ending with EOS_TOKEN.
    """
    # Add SOS and EOS tokens
    fen_with_tokens = f"{SOS_TOKEN} {fen} {EOS_TOKEN}"
    return [vocab.index(char) for char in fen_with_tokens]

def indices_to_fen(indices, vocab):
    """Decode a list of indices into a FEN string for the Seq2Seq model.

    Args:
        indices (list): A list of integer indices.
        vocab (str): The vocabulary string used for encoding.

    Returns:
        str: The decoded FEN string (without SOS/EOS tokens).
    """
    fen_chars = [vocab[i] for i in indices]
    fen_str = ''.join(fen_chars).replace(SOS_TOKEN, '').replace(EOS_TOKEN, '').replace(PAD_TOKEN, '').strip()
    return fen_str