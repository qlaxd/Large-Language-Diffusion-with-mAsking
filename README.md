# LLaDA: Large Language Diffusion with mAsking

This project provides a PyTorch implementation of **LLaDA (Large Language Diffusion with mAsking)**, a non-autoregressive language model that uses a diffusion-like masking process to generate text. The project also includes a standard **autoregressive (GPT-like) baseline model** for comparison. Both models are trained on the TinyShakespeare dataset.

The implementation follows the detailed plan outlined in `TASK.md` and has been refactored into a modular, script-based structure for clarity and ease of use.

## Features

- **LLaDA Model**: A bidirectional Transformer that learns to predict masked tokens in a sequence.
- **Autoregressive Model**: A standard causal Transformer baseline.
- **Modular Codebase**: The logic is separated into distinct modules for data handling, model architecture, training, and generation.
- **Centralized Configuration**: All hyperparameters and file paths are managed in a single configuration file.
- **Training & Validation**: Complete training and validation loops for both models.
- **Checkpointing**: Automatically saves the best performing model based on validation loss.
- **Result Visualization**: Generates and saves loss curves for each training run.

## Project Structure

```
/
├── configs/
│   └── base_config.py      # Central configuration for hyperparameters and paths
├── data/
│   └── tinyshakespeare.txt # The dataset (downloaded automatically)
├── outputs/
│   ├── models/             # Saved model checkpoints (.pth)
│   └── plots/              # Saved loss curves (.png)
├── tests/
│   ├── test_data_utils.py  # Tests for data utilities
│   └── test_model.py       # Tests for model architectures
├── .gitignore
├── data_utils.py           # Data loading and tokenization logic
├── model.py                # LLaDA and Autoregressive model architectures
├── train.py                # Training and validation loops
├── generate.py             # Text generation logic (inference)
├── main.py                 # Main entry point to run training or generation
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The first time you run the training, the script will automatically download the TinyShakespeare dataset into the `data/` directory.

## Usage

The project is controlled via the `main.py` script, which takes command-line arguments to specify the desired mode and model type.

### Training

To train a model, use the `--mode train` argument.

-   **Train the LLaDA model:**
    ```bash
    python main.py --mode train --model_type llada
    ```

-   **Train the Autoregressive baseline model:**
    ```bash
    python main.py --mode train --model_type autoregressive
    ```

Training progress will be displayed in the console. The best model checkpoints will be saved to `outputs/models/`, and loss curves will be saved to `outputs/plots/`.

### Text Generation

To generate text from a trained model, use the `--mode generate` argument.

-   **Generate text with the LLaDA model:**
    ```bash
    python main.py --mode generate --model_type llada --prompt "To be, or not to be"
    ```

-   **Generate text with the Autoregressive model:**
    ```bash
    python main.py --mode generate --model_type autoregressive --prompt "JULIET:"
    ```
    *(Note: The generation script `generate.py` is not yet fully implemented.)*

### Configuration

All hyperparameters (e.g., `BATCH_SIZE`, `LEARNING_RATE`, `EMBEDDING_DIM`), file paths, and training settings can be easily modified in the `configs/base_config.py` file.
