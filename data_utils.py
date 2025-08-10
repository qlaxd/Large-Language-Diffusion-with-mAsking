"""
Data processing utilities for the LLaDA project.

Includes functions for downloading the dataset, a character-level tokenizer,
and a PyTorch Dataset class for creating sequences.
"""

import torch
from torch.utils.data import Dataset
import requests

# Importing config might seem unusual here, but it centralizes paths.
# A more advanced setup might use dependency injection.
from configs import base_config

def download_tinyshakespeare(url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", dest_path=base_config.TINYSHAKESPEARE_PATH):
    """
    Downloads the TinyShakespeare dataset from the given URL.

    Args:
        url (str): The URL of the dataset.
        dest_path (str): The local path to save the file.

    Returns:
        str: The raw text content of the dataset, or None if download fails.
    """
    try:
        print("Downloading TinyShakespeare dataset...")
        response = requests.get(url)
        response.raise_for_status()
        with open(dest_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Download complete! Saved to {dest_path}")
        return response.text
    except Exception as e:
        print(f"Download failed: {e}")
        return None

class SimpleTokenizer:
    """
    A simple character-level tokenizer.

    Handles encoding text to token IDs and decoding token IDs back to text.
    Manages special tokens like [PAD], [MASK], [START], [END].
    """
    def __init__(self, text):
        """
        Initializes the tokenizer based on the provided text corpus.

        Args:
            text (str): The entire text corpus to build the vocabulary from.
        """
        self.special_tokens = {'[PAD]': 0, '[MASK]': 1, '[START]': 2, '[END]': 3}
        unique_chars = sorted(list(set(text)))
        self.char_to_id = self.special_tokens.copy()
        for i, char in enumerate(unique_chars):
            self.char_to_id[char] = len(self.special_tokens) + i
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        self.pad_token_id = self.special_tokens['[PAD]']
        self.mask_token_id = self.special_tokens['[MASK]']

    def encode(self, text):
        """Converts a string of text into a list of token IDs."""
        return [self.char_to_id.get(c, self.mask_token_id) for c in text]

    def decode(self, ids):
        """Converts a list of token IDs back into a string."""
        return "".join([self.id_to_char.get(i, '?') for i in ids])

class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for creating fixed-length sequences from the TinyShakespeare corpus.
    """
    def __init__(self, text, tokenizer, seq_length):
        """
        Args:
            text (str): The text to create sequences from.
            tokenizer (SimpleTokenizer): The tokenizer instance to use.
            seq_length (int): The fixed length of each sequence.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.token_ids = tokenizer.encode(text)
        self.sequences = []
        # Create sequences by sliding a window over the token IDs
        for i in range(0, len(self.token_ids) - seq_length, seq_length): # Non-overlapping sequences
            self.sequences.append(self.token_ids[i:i + seq_length])

    def __len__(self):
        """Returns the total number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Returns a sequence at a given index as a LongTensor."""
        return torch.LongTensor(self.sequences[idx])
