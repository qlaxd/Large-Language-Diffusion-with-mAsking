"""
Training script for the LLaDA project.

This script contains the main training loop, loss functions, and data loader creation
for both the LLaDA and the autoregressive models.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import time
import wandb

# Import from our new modular structure
from configs import base_config
from data_utils import SimpleTokenizer, ShakespeareDataset, download_tinyshakespeare
from model import LLaDAModel, AutoregressiveModel

# --- Data Loading ---
def create_data_loaders(tokenizer):
    """
    Creates and returns the training and validation DataLoaders.

    Args:
        tokenizer (SimpleTokenizer): An initialized tokenizer.

    Returns:
        tuple: A tuple containing (train_loader, val_loader).
    """
    with open(base_config.TINYSHAKESPEARE_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    split_idx = int(base_config.TRAIN_SPLIT * len(raw_text))
    train_text = raw_text[:split_idx]
    val_text = raw_text[split_idx:]

    train_dataset = ShakespeareDataset(train_text, tokenizer, seq_length=base_config.SEQ_LENGTH)
    val_dataset = ShakespeareDataset(val_text, tokenizer, seq_length=base_config.SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=base_config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=base_config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")
    
    return train_loader, val_loader

# --- Diffusion Forward Process ---
def forward_process(sequences, t, mask_token_id):
    """
    Applies the forward masking process to a batch of sequences.

    Args:
        sequences (torch.Tensor): The input batch of sequences [batch, seq_len].
        t (float): The probability of masking a token.
        mask_token_id (int): The ID of the [MASK] token.

    Returns:
        tuple: (masked_sequences, binary_mask)
    """
    random_values = torch.rand_like(sequences, dtype=torch.float32)
    binary_mask = random_values < t
    masked_sequences = torch.where(binary_mask, mask_token_id, sequences)
    return masked_sequences, binary_mask

# --- Loss Functions ---
def compute_llada_loss(logits, targets, mask, t):
    """
    Computes the LLaDA loss, weighted by 1/t, only on masked positions.
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    mask_flat = mask.view(-1)
    masked_loss = loss[mask_flat]
    
    if masked_loss.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
        
    return masked_loss.mean() / t

def compute_autoregressive_loss(logits, targets):
    """
    Computes the standard cross-entropy loss for an autoregressive model.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

# --- Plotting ---
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
    
    os.makedirs(base_config.PLOT_DIR, exist_ok=True)
    save_path = os.path.join(base_config.PLOT_DIR, f"{model_type}_loss_curves.png")
    
    plt.savefig(save_path)
    print(f"Loss curves saved to {save_path}")
    plt.close()

# --- Main Training Function ---
def run_training(model_type):
    """
    Main training and validation loop with wandb integration.
    """
    # --- Setup ---
    device = torch.device(base_config.DEVICE)
    config_dict = {k: v for k, v in base_config.__dict__.items() if k.isupper()}

    run_name = f"{model_type}_{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project="LLaDA-Take-Home", name=run_name, config=config_dict)
    print(f"Using device: {device}")
    print(f"Wandb run initialized: {run_name}")

    # --- Data ---
    if not os.path.exists(base_config.TINYSHAKESPEARE_PATH):
        raw_text = download_tinyshakespeare()
    else:
        with open(base_config.TINYSHAKESPEARE_PATH, 'r', encoding='utf-8') as f:
            raw_text = f.read()

    tokenizer = SimpleTokenizer(raw_text)
    wandb.config.update({"vocab_size": tokenizer.vocab_size})
    train_loader, val_loader = create_data_loaders(tokenizer)

    # --- Model ---
    if model_type == 'llada':
        model = LLaDAModel(wandb.config)
    else:
        model = AutoregressiveModel(wandb.config)
    model.to(device)
    wandb.watch(model, log='all', log_freq=100)
    
    optimizer = optim.AdamW(model.parameters(), lr=wandb.config.LEARNING_RATE)
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    print(f"\n--- Starting Training for {model_type.upper()} Model ---")
    for epoch in range(wandb.config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if model_type == 'llada':
                t = torch.rand(1).item() * (1.0 - 1e-3) + 1e-3
                masked_seq, mask = forward_process(batch, t, tokenizer.mask_token_id)
                logits = model(masked_seq)
                loss = compute_llada_loss(logits, batch, mask, t)
            else: # autoregressive
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                logits = model(inputs)
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
                    t = 0.5
                    masked_seq, mask = forward_process(batch, t, tokenizer.mask_token_id)
                    logits = model(masked_seq)
                    loss = compute_llada_loss(logits, batch, mask, t)
                else: # autoregressive
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]
                    logits = model(inputs)
                    loss = compute_autoregressive_loss(logits, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1:02d}/{wandb.config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch + 1})

        # --- Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(base_config.MODEL_DIR, exist_ok=True)
            save_path = os.path.join(base_config.MODEL_DIR, f"{model_type}_model_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model improved. Saved to {save_path}")
            wandb.run.summary["best_val_loss"] = best_val_loss

    # --- Finalization ---
    plot_loss_curves(history, model_type)
    # Log the plot to wandb
    wandb.log({"loss_curves": wandb.Image(os.path.join(base_config.PLOT_DIR, f"{model_type}_loss_curves.png"))})
    print(f"--- Finished Training for {model_type.upper()} Model ---")
    wandb.finish()