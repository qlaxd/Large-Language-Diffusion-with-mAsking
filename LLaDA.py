
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
import numpy as np
import math

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

class ShakespeareDataset:
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

# --- 3. Szekció: Modell Architektúra (Mask Predictor) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class LLaDAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ff_dim, max_seq_len):
        super(LLaDAModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoding(src)
        output = self.transformer_encoder(src) # No causal mask needed
        logits = self.output_layer(output)
        return logits

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
        masked_seq, _ = forward_process(sample_batch, 0.5, tokenizer.mask_token_id)

        # Ellenőrzés (3. szekció)
        print(f"\n--- Section 3: Model Architecture Check ---")
        model = LLaDAModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=256,
            num_heads=4,
            num_layers=4,
            ff_dim=1024,
            max_seq_len=128
        )
        logits = model(masked_seq)
        print(f"Model output logits shape: {logits.shape}")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameter count: {num_params:,}")
