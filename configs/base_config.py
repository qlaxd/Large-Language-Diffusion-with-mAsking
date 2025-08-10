"""
Base configuration file for the LLaDA project.
All hyperparameters and paths are centralized here.
"""
import torch

# --- File Paths ---
DATA_DIR = "data"
TINYSHAKESPEARE_PATH = f"{DATA_DIR}/tinyshakespeare.txt"
OUTPUT_DIR = "outputs"
MODEL_DIR = f"{OUTPUT_DIR}/models"
PLOT_DIR = f"{OUTPUT_DIR}/plots"

# --- Data Parameters ---
# Vocabulary size is determined by the SimpleTokenizer on the TinyShakespeare dataset.
# It includes 65 unique characters + 4 special tokens -> 69.
# Let's make it dynamic later, but for now, this is a placeholder.
VOCAB_SIZE = 69
TRAIN_SPLIT = 0.8

# --- Model Hyperparameters ---
SEQ_LENGTH = 128
EMBEDDING_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT = 0.1

# --- Training Hyperparameters ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10 # For a full run, might need more
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Generation Parameters ---
GENERATION_MAX_LENGTH = 500
GENERATION_NUM_STEPS = 20 # Number of denoising steps
