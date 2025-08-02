# LLaDA Model Evaluation Report

This report details the implementation, training, and evaluation of the LLaDA (Large Language Diffusion with mAsking) model, as well as a baseline autoregressive model, on the TinyShakespeare dataset.

## 1. Executive Summary

*(This section will be filled in after the experiments are complete.)*

## 2. Model Architectures

### 2.1. LLaDA (Masked Diffusion Model)

- **Objective:** Predict masked tokens in a sequence.
- **Architecture:** Transformer Encoder (bidirectional attention).
- **Key Feature:** Non-autoregressive, sees the full context to predict masked tokens.

### 2.2. Autoregressive Baseline (GPT-like)

- **Objective:** Predict the next token in a sequence.
- **Architecture:** Transformer Decoder (causal/masked attention).
- **Key Feature:** Generates text token-by-token, left-to-right.

## 3. Experimental Setup

- **Dataset:** TinyShakespeare
- **Hardware:** *(To be filled in, e.g., CPU, GPU model)*
- **Key Hyperparameters:**
  - `SEQ_LENGTH`:
  - `BATCH_SIZE`:
  - `EMBEDDING_DIM`:
  - `NUM_HEADS`:
  - `NUM_LAYERS`:
  - `FF_DIM`:
  - `EPOCHS`:
  - `LEARNING_RATE`:

## 4. Quantitative Results

### 4.1. Training & Validation Loss

*(Loss curves for both models will be embedded here.)*

### 4.2. Perplexity

| Model                | Perplexity on Validation Set |
|----------------------|------------------------------|
| LLaDA                | *(To be filled in)*          |
| Autoregressive       | *(To be filled in)*          |

## 5. Qualitative Analysis

### 5.1. Generated Samples

#### LLaDA Model

**Prompt:** "To be, or not to be"
> *(Generated text will be inserted here.)*

**Prompt:** "JULIET:"
> *(Generated text will be inserted here.)*

#### Autoregressive Model

**Prompt:** "To be, or not to be"
> *(Generated text will be inserted here.)*

**Prompt:** "JULIET:"
> *(Generated text will be inserted here.)*

### 5.2. Discussion

*(This section will contain a detailed comparison of the models based on the generated samples, discussing coherence, diversity, and failure modes.)*

## 6. Conclusion & Future Work

*(This section will summarize the findings and suggest potential areas for improvement.)*
