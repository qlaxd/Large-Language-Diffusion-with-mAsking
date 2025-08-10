"""
Main entry point for the LLaDA project.

This script handles command-line argument parsing to run different modes of the
project, such as training models or generating text.
"""

import argparse

from train import run_training
from generate import run_generation

def main():
    """
    Parses command-line arguments and executes the corresponding project mode.
    """
    parser = argparse.ArgumentParser(description="LLaDA: Large Language Diffusion with mAsking")
    
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['train', 'generate'],
        help='The mode to run the script in: `train` a model or `generate` text.'
    )
    parser.add_argument(
        '--model_type', 
        type=str, 
        required=True, 
        choices=['llada', 'autoregressive'],
        help='The type of model to use.'
    )
    parser.add_argument(
        '--prompt', 
        type=str, 
        default='O, Romeo, Romeo! wherefore art thou Romeo?',
        help='The prompt to use for text generation.'
    )
    # In a real project, you might add an argument for a custom config file
    # parser.add_argument('--config', type=str, default='base_config', help='The configuration file to use.')

    args = parser.parse_args()

    print(f"--- Running in {args.mode.upper()} mode for {args.model_type.upper()} model ---")

    if args.mode == 'train':
        run_training(model_type=args.model_type)
    elif args.mode == 'generate':
        run_generation(model_type=args.model_type, prompt=args.prompt)
        pass

if __name__ == '__main__':
    main()
