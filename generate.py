"""
Inference script for text generation.

This script loads a trained model (either LLaDA or autoregressive) and
uses it to generate text based on a given prompt.
"""

import torch
import os

from configs import base_config
from model import LLaDAModel, AutoregressiveModel
from data_utils import SimpleTokenizer

def generate_llada_text(model, tokenizer, prompt, device):
    """
    Generates text using the LLaDA model's reverse diffusion process.

    Args:
        model (LLaDAModel): The trained LLaDA model.
        tokenizer (SimpleTokenizer): The tokenizer instance.
        prompt (str): The initial text to start generation from.
        device (torch.device): The device to run inference on.

    Returns:
        str: The generated text.
    """
    model.eval()
    prompt_tokens = tokenizer.encode(prompt)
    
    # Ensure the sequence is at least as long as the generation length
    if len(prompt_tokens) >= base_config.GENERATION_MAX_LENGTH:
        seq_tokens = prompt_tokens[:base_config.GENERATION_MAX_LENGTH]
    else:
        seq_tokens = prompt_tokens + [tokenizer.mask_token_id] * (base_config.GENERATION_MAX_LENGTH - len(prompt_tokens))

    seq = torch.LongTensor(seq_tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        for i in range(base_config.GENERATION_NUM_STEPS):
            # In the reverse process, we predict the original tokens from the masked ones
            logits = model(seq)
            
            # Use argmax for deterministic generation, or sampling for diversity
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Only update the tokens that were masked
            mask_positions = (seq == tokenizer.mask_token_id)
            seq[mask_positions] = predicted_ids[mask_positions]

            # For all but the last step, re-mask a fraction of the tokens
            if i < base_config.GENERATION_NUM_STEPS - 1:
                # This is a simple re-masking schedule. More complex ones can be used.
                remask_prob = 1.0 - ((i + 1) / base_config.GENERATION_NUM_STEPS)
                
                # Create a random mask, but don't re-mask the original prompt
                remask_filter = torch.rand_like(seq, dtype=torch.float32) < remask_prob
                remask_filter[:, :len(prompt_tokens)] = False
                
                seq[remask_filter] = tokenizer.mask_token_id

    return tokenizer.decode(seq.squeeze(0).tolist())

def generate_autoregressive_text(model, tokenizer, prompt, device):
    """
    Generates text using the autoregressive baseline model.

    Args:
        model (AutoregressiveModel): The trained autoregressive model.
        tokenizer (SimpleTokenizer): The tokenizer instance.
        prompt (str): The initial text to start generation from.
        device (torch.device): The device to run inference on.

    Returns:
        str: The generated text.
    """
    model.eval()
    tokens = tokenizer.encode(prompt)
    seq = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(base_config.GENERATION_MAX_LENGTH - len(tokens)):
            logits = model(seq)
            # Get the logits for the very last token
            next_token_logits = logits[:, -1, :]
            # Use argmax for deterministic output
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            seq = torch.cat([seq, next_token], dim=1)

    return tokenizer.decode(seq.squeeze(0).tolist())

def run_generation(model_type, prompt):
    """
    Main function to load a model and run text generation.

    Args:
        model_type (str): The type of model to use ('llada' or 'autoregressive').
        prompt (str): The prompt to start generation with.
    """
    device = torch.device(base_config.DEVICE)
    print(f"Using device: {device}")

    # --- Load Tokenizer (re-created from the dataset file) ---
    # In a production system, the tokenizer would be saved and loaded directly.
    with open(base_config.TINYSHAKESPEARE_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    tokenizer = SimpleTokenizer(raw_text)
    base_config.VOCAB_SIZE = tokenizer.vocab_size # Ensure vocab size is correct

    # --- Load Model ---
    if model_type == 'llada':
        model = LLaDAModel(base_config)
        generation_func = generate_llada_text
    else:
        model = AutoregressiveModel(base_config)
        generation_func = generate_autoregressive_text
    
    model_path = os.path.join(base_config.MODEL_DIR, f"{model_type}_model_best.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        print("Please run training first using: python main.py --mode train --model_type", model_type)
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"Loaded {model_type.upper()} model from {model_path}")

    # --- Generate Text ---
    print(f"\n--- Generating text with prompt: '{prompt}' ---")
    generated_text = generation_func(model, tokenizer, prompt, device)
    print("\n--- Generated Text ---")
    print(generated_text)
    print("\n----------------------")
