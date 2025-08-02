EG_TakeHome_LLaDA.ipynb
EG_TakeHome_LLaDA.ipynb_
LLaDA: Large Language Diffusion with mAsking
Take-Home Exercise: Implementing Diffusion Models for Language Generation
Welcome to your take-home exercise!

In this notebook, you'll implement LLaDA (Large Language Diffusion with mAsking), a novel approach to language modeling that challenges the dominance of autoregressive models.

Framework Freedom: Although we'd prefer you use PyTorch, you're free to use any deep learning framework you prefer and structure your code however you like. This notebook provides guidance on what to implement, not how to implement it. You can freely delete the placeholder code for the sections below. You can freely install and use new packages.

Goals: the key criteria we look for in your solution are the following:

Thoughtful design choices
Thorough evaluation
Clean and detailes experiment tracking
Note: We encourage you to submit your solution even in the case you don't have a fully working prototype at the deadline. At this point, your thought process and workflow matters more than the benchmark results.

Let's dive in!

⚠️ Important: We recommend you read the whole notebook before starting your implementation. See the implementation checklist for a comprehensive list of deliverables.

[ ]

# Setup and Imports

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import requests
import random

# Set random seeds for reproducibility

random.seed(42)
np.random.seed(42)

PART 1: Understanding Diffusion Models for Language
What are Diffusion Models?
Traditional language models (like GPT) generate text autoregressively - one token at a time, left to right. But what if we could generate text differently?

Diffusion models work by:

Forward Process: Gradually corrupt data by adding noise
Reverse Process: Learn to remove noise and recover original data
For language, instead of adding Gaussian noise (like in image diffusion), we can use MASKING:

Forward: Randomly mask tokens with increasing probability
Reverse: Predict what the masked tokens should be
The LLaDA Approach
LLaDA uses a "masked diffusion model" where:

At time t=0: Clean text (no masks)
At time t∈(0,1]: Partially masked text (each token masked with probability t)
At time t=1: Fully masked text
The model learns to predict original tokens given partially masked sequences.

Key Insight: Unlike autoregressive models, LLaDA can see the entire sequence context (bidirectional) when making predictions!

[ ]

# Let's visualize the masking process

def visualize_masking_process():
    """Demonstrate how masking probability increases with time t."""
    sample_text = "To be or not to be, that is the question"
    tokens = sample_text.split()

    print("LLaDA Forward Process: Gradual Masking")
    print("=" * 50)
    
    for t in [0.0, 0.3, 0.6, 0.9]:
        # Simulate masking for visualization
        masked_tokens = []
        for token in tokens:
            if random.random() < t:
                masked_tokens.append("[MASK]")
            else:
                masked_tokens.append(token)
    
        masked_text = " ".join(masked_tokens)
        print(f"t = {t:.1f} (≈{int(t*100):2}% masked): {masked_text}")

# Run visualization

visualize_masking_process()
PART 2: Data Preparation
Your Task: Set up the TinyShakespeare Dataset
What you need to implement:

Data Loading

Download TinyShakespeare from: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
Load and preprocess the text
Tokenization

Choose your tokenization approach (character-level recommended for simplicity)
Create vocabulary with special tokens: [MASK], [PAD], [START], [END]
Implement encode/decode functions
Dataset Creation

Split into train/validation (suggest 80/20)
Create sequences of fixed length (128-512 tokens recommended)
Handle padding appropriately
Expected Outputs:

Vocabulary size
Number of training/validation sequences
Sample of tokenized text

[ ]
def download_tinyshakespeare():
    """Download the TinyShakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    try:
        print("Downloading TinyShakespeare dataset...")
        response = requests.get(url)
        response.raise_for_status()
    
        with open('tinyshakespeare.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)
    
        print("Download complete!")
        return response.text
    
    except Exception as e:
        print(f"Download failed: {e}")
        print("Try downloading manually from:", url)
        return None

# Download the dataset

raw_text = download_tinyshakespeare()

if raw_text:
    print(f"Dataset size: {len(raw_text):,} characters")
    print(f"Sample text:\n{raw_text[:200]}...")
else:
    print("Please download the dataset manually and load it here")
    # raw_text = open('tinyshakespeare.txt', 'r').read()
Tokenization Implementation

[ ]

# @title Tokenization Implementation

class SimpleTokenizer:
    """
    Character-level tokenizer for TinyShakespeare.

    Creates a vocabulary from unique characters in the text and adds special tokens.
    Feel free to replace this implemetnation with your own.
    """
    
    def __init__(self, text):
        """Initialize tokenizer with text corpus."""
        # Special tokens
        self.special_tokens = {
            '[PAD]': 0,
            '[MASK]': 1,
            '[START]': 2,
            '[END]': 3
        }
    
        # Get unique characters from text
        unique_chars = sorted(list(set(text)))
    
        # Create vocabulary: special tokens + characters
        self.char_to_id = self.special_tokens.copy()
        for i, char in enumerate(unique_chars):
            self.char_to_id[char] = len(self.special_tokens) + i
    
        # Create reverse mapping
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
    
        # Store vocabulary info
        self.vocab_size = len(self.char_to_id)
        self.pad_token_id = self.special_tokens['[PAD]']
        self.mask_token_id = self.special_tokens['[MASK]']
        self.start_token_id = self.special_tokens['[START]']
        self.end_token_id = self.special_tokens['[END]']
    
        print(f"Vocabulary created: {self.vocab_size} tokens")
        print(f"Special tokens: {list(self.special_tokens.keys())}")
        print(f"Character range: '{min(unique_chars)}' to '{max(unique_chars)}'")
    
    def encode(self, text):
        """Convert text to list of token IDs."""
        return [self.char_to_id.get(char, self.mask_token_id) for char in text]
    
    def decode(self, token_ids):
        """Convert list of token IDs back to text."""
        return ''.join([self.id_to_char.get(token_id, '[UNK]') for token_id in token_ids])
    
    def encode_with_special_tokens(self, text, add_start=True, add_end=True):
        """Encode text with start/end tokens."""
        tokens = []
        if add_start:
            tokens.append(self.start_token_id)
        tokens.extend(self.encode(text))
        if add_end:
            tokens.append(self.end_token_id)
        return tokens

# Initialize tokenizer and create datasets

if raw_text:
    tokenizer = SimpleTokenizer(raw_text)

    # Test tokenization
    sample_text = "Hello world!"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTokenization test:")
    print(f"Original: '{sample_text}'")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  '{decoded}'")
    
    # Test special tokens
    special_test = tokenizer.encode_with_special_tokens("Hi!")
    special_decoded = tokenizer.decode(special_test)
    print(f"With special tokens: {special_test} -> '{special_decoded}'")

Dataset Creation

[ ]

# @title Dataset Creation

class ShakespeareDataset:
    """
    Dataset class for TinyShakespeare sequences.

    Creates fixed-length sequences from the text corpus with padding.
    """
    
    def __init__(self, text, tokenizer, seq_length=128, stride=None):
        """
        Initialize dataset.
    
        Args:
            text: Raw text string
            tokenizer: SimpleTokenizer instance
            seq_length: Fixed length for all sequences
            stride: Step size between sequences (defaults to seq_length for non-overlapping)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride or seq_length
    
        # Tokenize the entire text
        self.token_ids = tokenizer.encode(text)
    
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.token_ids) - seq_length + 1, self.stride):
            sequence = self.token_ids[i:i + seq_length]
            self.sequences.append(sequence)
    
        print(f"Created {len(self.sequences):,} sequences of length {seq_length}")
    
        # Store some statistics
        self.num_tokens = len(self.token_ids)
        self.num_sequences = len(self.sequences)
    
    def __len__(self):
        """Return number of sequences in dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get sequence by index."""
        return self.sequences[idx]
    
    def get_batch(self, indices):
        """Get multiple sequences as a batch."""
        return [self.sequences[i] for i in indices]
    
    def sample_batch(self, batch_size):
        """Sample random batch of sequences."""
        indices = random.sample(range(len(self.sequences)), batch_size)
        return self.get_batch(indices)
    
    def get_stats(self):
        """Return dataset statistics."""
        return {
            'num_sequences': self.num_sequences,
            'seq_length': self.seq_length,
            'vocab_size': self.tokenizer.vocab_size,
            'total_tokens': self.num_tokens,
            'coverage': self.num_sequences * self.seq_length / self.num_tokens
        }

# Create train/validation split

if raw_text:
    # Split text (80/20)
    split_idx = int(0.8 * len(raw_text))
    train_text = raw_text[:split_idx]
    val_text = raw_text[split_idx:]

    # Create datasets with some overlap for validation
    train_dataset = ShakespeareDataset(train_text, tokenizer, seq_length=128, stride=128)
    val_dataset = ShakespeareDataset(val_text, tokenizer, seq_length=128, stride=64)
    
    print(f"\nDataset Statistics:")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")
    
    # Display detailed stats
    train_stats = train_dataset.get_stats()
    val_stats = val_dataset.get_stats()
    
    print(f"\nTraining set coverage: {train_stats['coverage']:.1%}")
    print(f"Validation set coverage: {val_stats['coverage']:.1%}")
    
    # Show sample sequences
    print(f"\nSample training sequence:")
    sample_seq = train_dataset[0]
    sample_text = tokenizer.decode(sample_seq)
    print(f"Length: {len(sample_seq)}")
    print(f"Tokens: {sample_seq[:20]}...")
    print(f"Text: '{sample_text[:100]}...'")
    
    # Show batch sampling
    print(f"\nBatch sampling test:")
    batch = train_dataset.sample_batch(3)
    print(f"Batch shape: {len(batch)} sequences x {len(batch[0])} tokens")
    for i, seq in enumerate(batch):
        text_preview = tokenizer.decode(seq[:50])
        print(f"  Sequence {i}: '{text_preview}...'")

PART 3: The Forward Process - Gradually Masking Text
Understanding the Forward Process
In LLaDA, the forward process gradually masks tokens:

At t=0: Original text (no masking)
At t=0.5: ~50% of tokens are masked randomly
At t=1: All tokens are masked
Each token is independently masked with probability t.

Your Task: Implement Forward Masking
What you need to implement:

forward_process(clean_sequence, t, mask_token_id):
    For each token in clean_sequence:
        With probability t: replace with [MASK]
        With probability (1-t): keep original token
    Return (masked_sequence, boolean_mask_indicating_what_was_masked)
Key Points:

Each token is masked independently
Same masking probability t for all tokens in a sequence
Return both the masked sequence and a mask indicating which positions were masked
Forward Process Implementation

[ ]

# @title Forward Process Implementation

def forward_process(sequences, t, mask_token_id):
    """
    Apply forward masking process to sequences.

    YOUR TASK: Complete this implementation, or create your own.
    """
    
    # TODO: Implement the forward masking process
    # Hint: For each token, mask with probability t
    
    pass

PART 4: Model Architecture - The Mask Predictor
Understanding the Architecture
The core of LLaDA is a Transformer that predicts masked tokens.

Key differences from standard language models:

NO causal masking - can see the entire sequence (bidirectional attention)
Predicts ALL masked tokens simultaneously
Only computes loss on masked positions
Your Task: Implement the Mask Predictor
Architecture Requirements:

Transformer-based architecture
Bidirectional attention (no causal masking)
Input: sequence with some tokens masked
Output: probability distribution over vocabulary for each position
Suggested Architecture:

Embedding layer (token + positional)
4-8 Transformer blocks with multi-head attention
Output projection to vocabulary size
Model size: ~1-10M parameters (keep it manageable)
Critical Implementation Note: Make sure your attention mechanism can see the full sequence context, not just previous tokens!

Model Implementation

[ ]

# @title Model Implementation

"""
YOUR TASK: Implement the Mask Predictor architecture.

Key requirements:

- Bidirectional attention (no causal masking!)
- Input: sequences with masked tokens
- Output: predictions for all positions
- Only compute loss on masked positions

Choose your framework and implement below.
"""

PART 5: Training Objective and Loss Function
Understanding the LLaDA Loss
The training objective is:

L(θ) = -E[1/t * Σ 1[x_i_t = MASK] * log p_θ(x_i_0 | x_t)]
This means:

Only compute loss on masked tokens (ignore unmasked positions)
Weight the loss by 1/t (more weight for higher masking rates)
Use cross-entropy loss for token prediction
Your Task: Implement the Training Loss
Pseudocode:

compute_loss(model, clean_sequences, t, mask_token_id):
    # Apply forward process
    masked_sequences, mask = forward_process(clean_sequences, t, mask_token_id)

    # Get model predictions
    logits = model(masked_sequences)
    
    # Compute cross-entropy loss only on masked positions
    loss = cross_entropy(logits[mask], clean_sequences[mask])
    
    # Weight by 1/t
    weighted_loss = loss / t
    
    return weighted_loss

Loss Function Implementation

[ ]

# @title Loss Function Implementation

def compute_llada_loss(model, sequences, t, tokenizer):
    """
    Compute the LLaDA training loss.

    Loss formula: L(θ) = -E[1/t * Σ 1[x_i_t = MASK] * log p_θ(x_i_0 | x_t)]
    
    YOUR TASK: Complete this implementation for your chosen framework.
    """
    
    # TODO: Implement loss computation
    
    pass

PART 6: Training Loop
Your Task: Train the LLaDA Model
Training Procedure:

For each batch of clean sequences:
Sample random t from uniform distribution [0, 1]
Apply forward process to get masked sequences
Compute loss (as defined above)
Update model parameters
Key Training Details:

Sample t uniformly from [0, 1] for each batch
Use appropriate optimizer (Adam with lr ~1e-4 is a good start)
Track both training and validation loss
Train for sufficient epochs to see convergence
Expected Outputs:

Training loss curves
Model convergence metrics
Final trained model
Training Implementation

[ ]

# @title Training Implementation

def train_llada_model(model, train_dataset, val_dataset, tokenizer, num_epochs=10, batch_size=32):
    """
    Train the LLaDA model.

    YOUR TASK: Complete this training loop for your framework or replace it completely.
    """
    
    print(f"Starting training: {num_epochs} epochs, batch size {batch_size}")
    
    # TODO: Set up optimizer, training state, etc...
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
    
        # TODO: forward, backward, update, logs, checkpointing

PART 7: Inference - The Reverse Process
Understanding Text Generation
To generate text with LLaDA:

Start with a fully masked sequence (except for prompt if provided)
Iteratively unmask tokens by predicting what they should be
Use remasking strategies for better quality
The key insight: we discretize the continuous reverse process into discrete steps.

Your Task: Implement Text Generation
Reverse Process Algorithm:

generate_text(model, prompt, max_length, num_steps):
    # Initialize sequence: prompt + masked tokens
    sequence = [prompt_tokens] + [MASK] * (max_length - len(prompt))

    for step in range(num_steps):
        # Compute time steps
        t = (num_steps - step) / num_steps  # Goes from 1 to 0
        s = (num_steps - step - 1) / num_steps
    
        # Predict masked tokens
        logits = model(sequence)
        predicted_tokens = sample_from_logits(logits)  # Use sampling or argmax
    
        # Update sequence with predictions
        sequence[masked_positions] = predicted_tokens[masked_positions]
    
        # Remask some tokens for next iteration
        fraction_to_remask = s / t
        randomly_remask(sequence, fraction_to_remask)
    
    return sequence

Implementation Notes:

You can use greedy decoding (argmax) or sampling
Try different remasking strategies (random vs. low-confidence)
Experiment with different numbers of sampling steps
Reverse Process Implementation

[ ]

# @title Reverse Process Implementation

def reverse_process_generate(model, prompt, max_length, num_steps, tokenizer):
    """
    Generate text using the reverse process.

    YOUR TASK: Implement the reverse sampling process.
    """
    
    print(f"Generating text: prompt='{prompt}', length={max_length}, steps={num_steps}")
    
    # TODO: Implement reverse process sampling
    
    pass

PART 8: Experiments and Evaluation
Your Task: Analyze model performance
Quantitative Metrics:

Perplexity on validation set
Training convergence (loss vs epochs)
Generation speed (tokens/second)
Qualitative Analysis:

Sample Generation: Generate text with various prompts
Coherence: How coherent are the generated samples?
Diversity: How diverse are multiple generations from same prompt?
Note: These are only suggestions. Feel free to extend your analysis with different methods.

Implementation Checklist
Core Implementation:
 Data loading and tokenization for TinyShakespeare
 Forward process (gradual masking) implementation
 Mask predictor architecture with bidirectional attention
 LLaDA training loss computation (with 1/t weighting)
 Training loop with random t sampling
 Reverse process sampling for text generation
 Autoregressive baseline model
 Model evaluation and comparison
Experimental Results:
 Training curves for both models
 Perplexity comparison on validation set
 Generated text samples
Analysis & Discussion:
 2-3 page analysis report (separate PDF)
 Comparison of LLaDA vs autoregressive approaches
 Discussion of strengths, weaknesses, and failure modes
DELIVERABLES TO SUBMIT:
This Notebook (Primary): Complete implementation with all code
Documentatoin and Analysis Report (PDF/Markdown): 2-3 pages of detailed analysis
Model Weights: Save checkpoints for later analysis (don't submit)
FINAL TIPS:
Focus on correctness over performance - we want to see understanding
Document your implementation choices and trade-offs
Be honest about limitations and areas for improvement
Include ablation studies if time permits
Test with small models first, then scale up if computational resources allow
Monitor training closely - diffusion models can be sensitive to hyperparameters
Questions?
If you have questions about the assignment or need clarification on any requirements, feel free to reach out. We're looking for thoughtful implementation and analysis rather than perfect performance.

Good luck with your implementation!

Colab paid products - Cancel contracts here
