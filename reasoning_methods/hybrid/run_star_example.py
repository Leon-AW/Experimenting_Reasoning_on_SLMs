#!/usr/bin/env python3
"""
Example script demonstrating how to use the modular STaR implementation.

This script shows different ways to run the STaR process:
1. Complete STaR process
2. Individual phases
3. Custom configurations
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and print its description."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    # For demonstration, we'll just print the commands
    # Uncomment the next line to actually run them
    # subprocess.run(cmd, shell=True, check=True)
    print("(Command not executed - remove comments to run)")

def main():
    print("STaR Implementation Examples")
    print("=" * 60)
    
    # Example 1: Complete STaR process with small dataset for testing
    run_command(
        "python star_main.py --dataset cqa --num_iterations 2 --max_samples 50 --debug",
        "Complete STaR process on CommonsenseQA (2 iterations, 50 samples, debug mode)"
    )
    
    # Example 2: Run only rationale collection
    run_command(
        "python star_main.py --collect_only --dataset cqa --iteration 1 --model_path meta-llama/Llama-3.2-1B --max_samples 20",
        "Collect rationales only for iteration 1 using base model"
    )
    
    # Example 3: Run only fine-tuning (after collection)
    run_command(
        "python star_main.py --finetune_only --dataset cqa --iteration 1",
        "Fine-tune model using collected rationales from iteration 1"
    )
    
    # Example 4: Evaluate a specific model
    run_command(
        "python star_main.py --eval_only --dataset cqa --model_path ./star_models/iteration_1/model --max_samples 100",
        "Evaluate the model from iteration 1"
    )
    
    # Example 5: Full STaR on GSM8K
    run_command(
        "python star_main.py --dataset gsm8k --num_iterations 3 --no_eval",
        "Full STaR process on GSM8K (3 iterations, no evaluation)"
    )
    
    # Example 6: Direct module usage
    run_command(
        "python collect_rationales.py --model_path meta-llama/Llama-3.2-1B --dataset arithmetic --iteration 1 --output_dir ./my_rationales --max_samples 100",
        "Direct rationale collection using the collect_rationales module"
    )
    
    run_command(
        "python finetune.py --rationales_dir ./my_rationales --dataset arithmetic --iteration 1 --output_dir ./my_models --num_train_steps 20",
        "Direct fine-tuning using the finetune module"
    )
    
    print(f"\n{'='*60}")
    print("Example Usage Summary")
    print(f"{'='*60}")
    print("""
Key Usage Patterns:

1. Quick Test:
   python star_main.py --dataset cqa --max_samples 10 --num_iterations 1 --debug

2. Full STaR Process:
   python star_main.py --dataset gsm8k --num_iterations 5

3. Step-by-step:
   # Collect rationales
   python star_main.py --collect_only --dataset cqa --iteration 1 --model_path meta-llama/Llama-3.2-1B
   
   # Fine-tune
   python star_main.py --finetune_only --dataset cqa --iteration 1
   
   # Evaluate
   python star_main.py --eval_only --dataset cqa --model_path ./star_models/iteration_1/model

4. Custom Directories:
   python star_main.py --dataset arithmetic --rationales_dir ./custom_rationales --models_dir ./custom_models

Important Notes:
- Use --max_samples for testing with smaller datasets
- Use --debug to see detailed generation and rationalization process
- Rationales are saved to files and can be inspected
- Each iteration's model is saved separately
- Fine-tuning always starts from the base model (as per STaR paper)
""")

if __name__ == "__main__":
    main() 