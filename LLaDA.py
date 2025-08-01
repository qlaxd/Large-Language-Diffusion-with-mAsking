
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
import numpy as np

# --- 0. Szekció: Projekt Előkészítés és Környezet ---

def setup_environment():
    """
    Ensures the environment is ready.
    (For this script, it's mostly about confirming imports work)
    """
    print("PyTorch version:", torch.__version__)
    print("Environment setup complete.")

# --- 1. Szekció: Adatfeldolgozó Modul ---

def download_tinyshakespeare(url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"):
    """
    Download the TinyShakespeare dataset.
    """
    try:
        print("Downloading TinyShakespeare dataset...")
        response = requests.get(url)
        response.raise_for_status()
        print("Download complete!")
        return response.text
    except Exception as e:
        print(f"Download failed: {e}")
        return None

class SimpleTokenizer:
    """
    Character-level tokenizer for TinyShakespeare.
    """
    def __init__(self, text):
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
        return [self.char_to_id.get(c, self.mask_token_id) for c in text]

    def decode(self, ids):
        return "".join([self.id_to_char.get(i, '?') for i in ids])

class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for TinyShakespeare sequences.
    """
    def __init__(self, text, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.token_ids = tokenizer.encode(text)
        self.sequences = []
        for i in range(0, len(self.token_ids) - seq_length + 1, seq_length):
            self.sequences.append(self.token_ids[i:i + seq_length])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx])

def create_data_loaders(raw_text, tokenizer, seq_length, batch_size):
    """
    Creates training and validation DataLoaders.
    """
    # Split text (80/20)
    split_idx = int(0.8 * len(raw_text))
    train_text = raw_text[:split_idx]
    val_text = raw_text[split_idx:]

    # Create datasets
    train_dataset = ShakespeareDataset(train_text, tokenizer, seq_length=seq_length)
    val_dataset = ShakespeareDataset(val_text, tokenizer, seq_length=seq_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")
    
    return train_loader, val_loader

# --- 2. Szekció: Diffúziós Folyamat (Forward Process) ---

def forward_process(sequences, t, mask_token_id):
    """
    Apply forward masking process to a batch of sequences.
    """
    random_values = torch.rand_like(sequences, dtype=torch.float32)
    binary_mask = random_values < t
    masked_sequences = torch.where(binary_mask, mask_token_id, sequences)
    return masked_sequences, binary_mask


if __name__ == '__main__':
    # --- Main execution ---
    setup_environment()

    # 1. Adatfeldolgozás
    raw_text = download_tinyshakespeare()
    if raw_text:
        tokenizer = SimpleTokenizer(raw_text)
        
        # Ellenőrzés (1. szekció)
        print(f"\n--- Section 1: Data Processing Check ---")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        sample_text = "Hello, world!"
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        print(f"Sample: '{sample_text}' -> Encoded: {encoded} -> Decoded: '{decoded}'")

        train_loader, val_loader = create_data_loaders(
            raw_text, 
            tokenizer, 
            seq_length=128, 
            batch_size=32
        )
        
        sample_batch = next(iter(train_loader))
        print("\nChecking a batch from DataLoader:")
        print("Batch shape:", sample_batch.shape)
        print("Original sequence (decoded):")
        print(tokenizer.decode(sample_batch[0].tolist()))

        # Ellenőrzés (2. szekció)
        print(f"\n--- Section 2: Forward Process Check ---")
        for t_val in [0.1, 0.5, 0.9]:
            masked_seq, mask = forward_process(sample_batch, t_val, tokenizer.mask_token_id)
            print(f"\nTesting with t = {t_val}:")
            print("Masked sequence (decoded):")
            print(tokenizer.decode(masked_seq[0].tolist()))
            masked_percentage = (mask.sum() / mask.numel()).item() * 100
            print(f"Actual masked percentage: {masked_percentage:.2f}%")
