
"""
LLaDA: Large Language Diffusion with mAsking
Implementation of a character-level diffusion model for text generation
and a baseline autoregressive model, as per the TASK.md specification.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
import numpy as np
import math
import torch.optim as optim
import os
import matplotlib.pyplot as plt

# --- 0. Szekció: Projekt Előkészítés és Környezet ---

def setup_environment(seed=42):
    """
    Sets up the environment, including random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print("PyTorch version:", torch.__version__)
    print("Environment setup complete.")

# --- 1. Szekció: Adatfeldolgozó Modul ---

def download_tinyshakespeare(url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"):
    """
    Downloads the TinyShakespeare dataset from the given URL.

    Args:
        url (str): The URL of the dataset.

    Returns:
        str: The raw text content of the dataset, or None if download fails.
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
    def __init__(self, text, tokenizer, seq_length=128):
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
        for i in range(0, len(self.token_ids) - seq_length + 1, seq_length):
            self.sequences.append(self.token_ids[i:i + seq_length])

    def __len__(self):
        """Returns the total number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Returns a sequence at a given index as a LongTensor."""
        return torch.LongTensor(self.sequences[idx])

def create_data_loaders(raw_text, tokenizer, config):
    """
    Creates training and validation DataLoaders.

    Args:
        raw_text (str): The complete raw text of the dataset.
        tokenizer (SimpleTokenizer): An initialized tokenizer.
        config (dict): A configuration dictionary with keys like 'SEQ_LENGTH' and 'BATCH_SIZE'.

    Returns:
        tuple: A tuple containing the training DataLoader and validation DataLoader.
    """
    split_idx = int(config['TRAIN_SPLIT'] * len(raw_text))
    train_text = raw_text[:split_idx]
    val_text = raw_text[split_idx:]

    train_dataset = ShakespeareDataset(train_text, tokenizer, seq_length=config['SEQ_LENGTH'])
    val_dataset = ShakespeareDataset(val_text, tokenizer, seq_length=config['SEQ_LENGTH'])

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")
    
    return train_loader, val_loader

# --- 2. Szekció: Diffúziós Folyamat (Forward Process) ---

def forward_process(sequences, t, mask_token_id):
    """
    Applies the forward masking process to a batch of sequences.

    Each token in the sequence is replaced by a [MASK] token with probability 't'.

    Args:
        sequences (torch.Tensor): The input batch of sequences (shape: [batch, seq_len]).
        t (float): The probability of masking a token.
        mask_token_id (int): The ID of the [MASK] token.

    Returns:
        tuple: A tuple containing:
            - masked_sequences (torch.Tensor): The sequences after masking.
            - binary_mask (torch.Tensor): A boolean tensor indicating which positions were masked.
    """
    random_values = torch.rand_like(sequences, dtype=torch.float32)
    binary_mask = random_values < t
    masked_sequences = torch.where(binary_mask, mask_token_id, sequences)
    return masked_sequences, binary_mask

# --- 3. Szekció: Modell Architektúrák ---

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformer models.
    """
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
        """Adds positional encoding to the input tensor."""
        x = x + self.pe[:, :x.size(1)]
        return x

class LLaDAModel(nn.Module):
    """
    LLaDA: A Transformer Encoder-based model for masked language modeling.
    This model uses bidirectional attention to predict masked tokens.
    """
    def __init__(self, config):
        super(LLaDAModel, self).__init__()
        self.embedding_dim = config['EMBEDDING_DIM']
        self.embedding = nn.Embedding(config['VOCAB_SIZE'], config['EMBEDDING_DIM'])
        self.pos_encoding = PositionalEncoding(config['EMBEDDING_DIM'], config['MAX_CONTEXT_LENGTH'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['EMBEDDING_DIM'], 
            nhead=config['NUM_HEADS'], 
            dim_feedforward=config['FF_DIM'], 
            batch_first=True,
            dropout=config['DROPOUT']
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['NUM_LAYERS'])
        self.output_layer = nn.Linear(config['EMBEDDING_DIM'], config['VOCAB_SIZE'])

    def forward(self, src):
        """
        Forward pass for the LLaDA model.

        Args:
            src (torch.Tensor): Input tensor of token IDs (shape: [batch, seq_len]).

        Returns:
            torch.Tensor: Logits over the vocabulary for each position (shape: [batch, seq_len, vocab_size]).
        """
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoding(src)
        output = self.transformer_encoder(src)
        logits = self.output_layer(output)
        return logits

class AutoregressiveModel(nn.Module):
    """
    A baseline Transformer Decoder-based model for autoregressive language modeling.
    This model uses causal (masked) attention.
    """
    def __init__(self, config):
        super(AutoregressiveModel, self).__init__()
        self.embedding_dim = config['EMBEDDING_DIM']
        self.embedding = nn.Embedding(config['VOCAB_SIZE'], config['EMBEDDING_DIM'])
        self.pos_encoding = PositionalEncoding(config['EMBEDDING_DIM'], config['MAX_CONTEXT_LENGTH'])
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config['EMBEDDING_DIM'],
            nhead=config['NUM_HEADS'],
            dim_feedforward=config['FF_DIM'],
            batch_first=True,
            dropout=config['DROPOUT']
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['NUM_LAYERS'])
        self.output_layer = nn.Linear(config['EMBEDDING_DIM'], config['VOCAB_SIZE'])

    def forward(self, tgt, tgt_mask):
        """
        Forward pass for the autoregressive model.

        Args:
            tgt (torch.Tensor): Target tensor of token IDs (shape: [batch, seq_len]).
            tgt_mask (torch.Tensor): Causal mask to prevent attending to future tokens.

        Returns:
            torch.Tensor: Logits over the vocabulary for each position (shape: [batch, seq_len, vocab_size]).
        """
        tgt = self.embedding(tgt) * math.sqrt(self.embedding_dim)
        tgt = self.pos_encoding(tgt)
        # The decoder layer expects a memory tensor, but for generation we can pass the target itself.
        output = self.transformer_decoder(tgt, tgt, tgt_mask=tgt_mask)
        logits = self.output_layer(output)
        return logits

# --- 4. Szekció: Tanítási Ciklus és Veszteségfüggvény ---

def get_causal_mask(size, device):
    """
    Generates a causal mask for autoregressive models.
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

def compute_llada_loss(logits, targets, mask, t):
    """
    Computes the LLaDA loss, weighted by 1/t.
    The loss is calculated only on the positions that were masked.
    """
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    
    full_loss = loss_fn(logits_flat, targets_flat)
    
    mask_flat = mask.view(-1)
    masked_loss = full_loss[mask_flat]
    
    if masked_loss.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
        
    # Weighting by 1/t gives more importance to harder tasks (higher t)
    return masked_loss.mean() / t

def compute_autoregressive_loss(logits, targets):
    """
    Computes the standard cross-entropy loss for an autoregressive model.
    """
    loss_fn = nn.CrossEntropyLoss()
    # Input: (Batch * SeqLen, VocabSize), Target: (Batch * SeqLen)
    # Using .reshape() instead of .view() to handle potential non-contiguous tensors.
    return loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

def train_model(model, train_loader, val_loader, optimizer, config, model_type='llada'):
    """
    Main training and validation loop. Handles both LLaDA and autoregressive models.
    Includes checkpointing to save the best model based on validation loss.
    """
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    device = config['DEVICE']
    model.to(device)

    print(f"\n--- Starting Training for {model_type.upper()} Model ---")
    for epoch in range(config['EPOCHS']):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if model_type == 'llada':
                t = torch.rand(1).item() * (1.0 - 1e-3) + 1e-3
                masked_seq, mask = forward_process(batch, t, config['TOKENIZER'].mask_token_id)
                logits = model(masked_seq)
                loss = compute_llada_loss(logits, batch, mask, t)
            else: # autoregressive
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                causal_mask = get_causal_mask(inputs.size(1), device)
                logits = model(inputs, causal_mask)
                loss = compute_autoregressive_loss(logits, targets)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if model_type == 'llada':
                    t = torch.rand(1).item() * (1.0 - 1e-3) + 1e-3
                    masked_seq, mask = forward_process(batch, t, config['TOKENIZER'].mask_token_id)
                    logits = model(masked_seq)
                    loss = compute_llada_loss(logits, batch, mask, t)
                else: # autoregressive
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]
                    causal_mask = get_causal_mask(inputs.size(1), device)
                    logits = model(inputs, causal_mask)
                    loss = compute_autoregressive_loss(logits, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{config['EPOCHS']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{model_type}_model_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model improved. Saved to {save_path}")

    return history

# --- 5. Szekció: Inferenciamotor (Reverse Process) ---

def generate_llada_text(model, tokenizer, prompt, config, num_steps=20):
    """
    Generates text using the LLaDA model's reverse diffusion process.
    """
    model.eval()
    device = config['DEVICE']
    model.to(device)
    max_length = config.get('GENERATE_LENGTH', 100)

    prompt_tokens = tokenizer.encode(prompt)
    seq = prompt_tokens + [tokenizer.mask_token_id] * (max_length - len(prompt_tokens))
    seq = torch.LongTensor(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        for i in range(num_steps):
            t = 1.0 - (i / num_steps)
            logits = model(seq)
            # Apply temperature and sample
            scaled_logits = logits / config.get('TEMPERATURE', 0.8)
            probs = torch.softmax(scaled_logits, dim=-1)
            predicted_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(seq.shape)
            
            mask_positions = (seq == tokenizer.mask_token_id)
            seq[mask_positions] = predicted_ids[mask_positions]

            if i < num_steps - 1:
                s = 1.0 - ((i + 1) / num_steps)
                remask_prob = s / t if t > 0 else 0
                
                remask_filter = torch.rand_like(seq, dtype=torch.float32) < remask_prob
                remask_filter[:, :len(prompt_tokens)] = False
                
                seq[remask_filter] = tokenizer.mask_token_id

    return tokenizer.decode(seq.squeeze(0).tolist())

def generate_autoregressive_text(model, tokenizer, prompt, config):
    """
    Generates text using the autoregressive baseline model.
    """
    model.eval()
    device = config['DEVICE']
    model.to(device)
    max_length = config.get('GENERATE_LENGTH', 100)
    
    tokens = tokenizer.encode(prompt)
    seq = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            causal_mask = get_causal_mask(seq.size(1), device)
            logits = model(seq, causal_mask)
            # Apply temperature to the logits for the last token
            next_token_logits = logits[:, -1, :] / config.get('TEMPERATURE', 0.8)
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)

    return tokenizer.decode(seq.squeeze(0).tolist())

# --- 6. Szekció: Kiértékelés és Analízis ---

def plot_loss_curves(history, model_type):
    """
    Plots and saves training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type.upper()} Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_path = f"{model_type}_loss_curves.png"
    plt.savefig(save_path)
    print(f"\nLoss curves saved to {save_path}")
    plt.close()

def calculate_perplexity(model, data_loader, config, model_type='llada'):
    """
    Calculates perplexity on a given dataset.
    """
    model.eval()
    device = config['DEVICE']
    model.to(device)
    total_loss = 0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=config['TOKENIZER'].pad_token_id)

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            if model_type == 'llada':
                # Perplexity for LLaDA is less standard. We evaluate its ability
                # to predict a small number of masked tokens.
                t = 0.1 # Low masking rate for evaluation
                masked_seq, mask = forward_process(batch, t, config['TOKENIZER'].mask_token_id)
                logits = model(masked_seq)
                
                # We only care about the loss on the masked tokens
                loss = compute_llada_loss(logits, batch, mask, t) * t # Un-weight the loss
                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()
            else: # autoregressive
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                causal_mask = get_causal_mask(inputs.size(1), device)
                logits = model(inputs, causal_mask)
                loss = loss_fn(logits.reshape(-1, config['VOCAB_SIZE']), targets.reshape(-1))
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()

    if total_tokens == 0: return float('inf')
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def main():
    """
    Main execution function to run the entire pipeline.
    """
    # --- Configuration ---
    config = {
        'SEQ_LENGTH': 128,
        'MAX_CONTEXT_LENGTH': 256,
        'BATCH_SIZE': 128,
        'EMBEDDING_DIM': 256,
        'NUM_HEADS': 4,
        'NUM_LAYERS': 4,
        'FF_DIM': 1024,
        'DROPOUT': 0.1,
        'EPOCHS': 20, # Increased for better results
        'LEARNING_RATE': 1e-4,
        'TRAIN_SPLIT': 0.9,
        'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'GENERATE_LENGTH': 200,
        'TEMPERATURE': 0.8, # Added for sampling
    }

    setup_environment()
    print(f"Using device: {config['DEVICE']}")

    # --- Data Processing ---
    raw_text = download_tinyshakespeare()
    if not raw_text:
        return
    
    tokenizer = SimpleTokenizer(raw_text)
    config['TOKENIZER'] = tokenizer
    config['VOCAB_SIZE'] = tokenizer.vocab_size
    
    train_loader, val_loader = create_data_loaders(raw_text, tokenizer, config)

    # --- LLaDA Model Training & Evaluation ---
    llada_model = LLaDAModel(config)
    optimizer = optim.AdamW(llada_model.parameters(), lr=config['LEARNING_RATE'])
    llada_history = train_model(llada_model, train_loader, val_loader, optimizer, config, model_type='llada')
    plot_loss_curves(llada_history, 'llada')
    
    # Load best model for generation and perplexity
    llada_model.load_state_dict(torch.load("llada_model_best.pth"))
    
    print("\n--- Generating Text with LLaDA ---")
    prompt = "JULIET:"
    llada_text = generate_llada_text(llada_model, tokenizer, prompt, config, num_steps=20)
    print(f"Prompt: '{prompt}'")
    print(f"Generated Text: {llada_text}")
    
    llada_ppl = calculate_perplexity(llada_model, val_loader, config, model_type='llada')
    print(f"\nLLaDA Perplexity on Validation Set: {llada_ppl:.2f}")

    # --- Autoregressive Model Training & Evaluation ---
    autoregressive_model = AutoregressiveModel(config)
    optimizer = optim.AdamW(autoregressive_model.parameters(), lr=config['LEARNING_RATE'])
    autoregressive_history = train_model(autoregressive_model, train_loader, val_loader, optimizer, config, model_type='autoregressive')
    plot_loss_curves(autoregressive_history, 'autoregressive')

    # Load best model for generation and perplexity
    autoregressive_model.load_state_dict(torch.load("autoregressive_model_best.pth"))

    print("\n--- Generating Text with Autoregressive Model ---")
    prompt = "JULIET:"
    autoregressive_text = generate_autoregressive_text(autoregressive_model, tokenizer, prompt, config)
    print(f"Prompt: '{prompt}'")
    print(f"Generated Text: {autoregressive_text}")

    autoregressive_ppl = calculate_perplexity(autoregressive_model, val_loader, config, model_type='autoregressive')
    print(f"\nAutoregressive Perplexity on Validation Set: {autoregressive_ppl:.2f}")

    # --- Final Report Update ---
    print("\nUpdating EVALUATION_REPORT.md...")
    # This part would ideally be automated to fill the report,
    # but for now, we print the results to be manually copied.
    print("\n--- Results Summary ---")
    print(f"LLaDA Perplexity: {llada_ppl:.2f}")
    print(f"Autoregressive Perplexity: {autoregressive_ppl:.2f}")
    print("\nLLaDA Generated Text:")
    print(llada_text)
    print("\nAutoregressive Generated Text:")
    print(autoregressive_text)
    print("\nPlease copy these results into EVALUATION_REPORT.md")


if __name__ == '__main__':
    main()
