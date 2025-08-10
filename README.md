# LLaDA: Large Language Diffusion with mAsking

This project is an implementation of a character-level diffusion model (LLaDA) for text generation, based on the principles outlined in the take-home exercise. It also includes a standard autoregressive Transformer model as a baseline for comparison.

The entire project has been refactored from a monolithic script into a modular, clean, and testable structure that supports training and inference from the command line and tracks experiments using Weights & Biases.

## Project Structure

```
/
├── configs/            # Centralized configuration files
├── data/               # Raw data files (e.g., tinyshakespeare.txt)
├── outputs/            # Saved models (.pth) and plots (.png)
├── tests/              # Unit tests for the project
├── .gitignore
├── data_utils.py       # Tokenizer and PyTorch Dataset classes
├── model.py            # Model architectures (LLaDA and Autoregressive)
├── train.py            # Training script with wandb integration
├── generate.py         # Inference script for text generation
├── main.py             # Main entry point for the CLI
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── EVALUATION_REPORT.md # Analysis of the model performance
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional but Recommended) Login to Weights & Biases:**
    To enable experiment tracking, log in to your W&B account. You will be prompted for your API key.
    ```bash
    wandb login
    ```

## Usage

The project is controlled via the `main.py` script with command-line arguments.

### Training

To train a model, use the `--mode train` argument and specify the model type.
The script will download the dataset, train the model, save the best version to `outputs/models/`, and log the experiment to Weights & Biases.

**Train the LLaDA model:**
```bash
python3 main.py --mode train --model_type llada
```

**Train the Autoregressive model:**
```bash
python3 main.py --mode train --model_type autoregressive
```

### Generation

To generate text with a trained model, use the `--mode generate` argument. The script will automatically load the best saved model weights.

**Generate with the LLaDA model:**
```bash
python3 main.py --mode generate --model_type llada --prompt "O Romeo, Romeo!"
```

**Generate with the Autoregressive model:**
```bash
python3 main.py --mode generate --model_type autoregressive --prompt "O Romeo, Romeo!"
```

### Running Tests

To ensure all components are working correctly, run the unit tests:
```bash
python3 -m unittest discover tests
```