# reasoning_methods/hybrid/finetune.py

import argparse
import os
import json
import torch
import gc
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

# Fine-tuning hyperparameters from STaR paper
INITIAL_TRAIN_STEPS = 40
STEP_INCREASE_FACTOR = 1.2
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
LOGGING_STEPS = 10

def load_model_and_tokenizer(model_id_or_path):
    """Loads the model and tokenizer for fine-tuning."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

def load_collected_rationales(rationales_dir, dataset_type, iteration):
    """
    Load collected rationales from the specified directory.
    
    Args:
        rationales_dir: Directory containing collected rationales
        dataset_type: Type of dataset ('cqa', 'gsm8k', 'arithmetic')
        iteration: Iteration number
        
    Returns:
        List of formatted text instances for fine-tuning
    """
    iter_dir = os.path.join(rationales_dir, f"iteration_{iteration}")
    
    # Load from the combined rationales file
    rationales_file = os.path.join(iter_dir, f"{dataset_type}_rationales.jsonl")
    
    finetuning_data = []
    generated_count = 0
    rationalized_count = 0
    
    if os.path.exists(rationales_file):
        with open(rationales_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                finetuning_data.append(item['formatted_text'])
                
                # Count by source for statistics
                if item.get('source') == 'generated':
                    generated_count += 1
                elif item.get('source') == 'rationalized':
                    rationalized_count += 1
        
        print(f"Loaded {len(finetuning_data)} total rationales from {rationales_file}")
        print(f"  Generated rationales: {generated_count}")
        print(f"  Rationalized rationales: {rationalized_count}")
    else:
        raise ValueError(f"Rationales file not found: {rationales_file}")
    
    if len(finetuning_data) == 0:
        raise ValueError(f"No rationales found for iteration {iteration} in {iter_dir}")
    
    return finetuning_data

def finetune_model(
    rationales_dir: str,
    dataset_type: str,
    iteration: int,
    output_dir: str,
    num_train_steps: int = None,
    debug: bool = False
):
    """
    Fine-tune the model on collected rationales for a single STaR iteration.
    
    This implements Step 7 from the STaR paper:
    - Load the original pretrained model M_0 (not the previous iteration's model)
    - Fine-tune on D_n^gen ∪ D_n^rat (generated + rationalized rationales)
    - Save the resulting model M_n
    
    Args:
        rationales_dir: Directory containing collected rationales
        dataset_type: Type of dataset ('cqa', 'gsm8k', 'arithmetic')
        iteration: Current iteration number
        output_dir: Directory to save the fine-tuned model
        num_train_steps: Number of training steps (if None, calculated from iteration)
        debug: Enable debug printing
        
    Returns:
        Path to the saved fine-tuned model
    """
    print(f"\n=== Fine-tuning Model for Iteration {iteration} ===")
    print(f"Dataset: {dataset_type}")
    print(f"Base model: {BASE_MODEL_ID}")
    
    # Calculate training steps if not provided
    if num_train_steps is None:
        # STaR paper: start with 40 steps, increase by 20% each iteration
        num_train_steps = int(INITIAL_TRAIN_STEPS * (STEP_INCREASE_FACTOR ** (iteration - 1)))
    
    print(f"Training steps: {num_train_steps}")
    
    # Load collected rationales
    print("Loading collected rationales...")
    finetuning_data = load_collected_rationales(rationales_dir, dataset_type, iteration)
    
    if not finetuning_data:
        raise ValueError(f"No rationales found for iteration {iteration}")
    
    # Create dataset for SFTTrainer
    finetuning_dataset = Dataset.from_dict({"text": finetuning_data})
    
    # Load model for fine-tuning
    # IMPORTANT: Always start from the original BASE_MODEL_ID as per STaR paper
    print(f"Loading base model for fine-tuning: {BASE_MODEL_ID}")
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_ID)
    
    # Create output directory for this iteration
    iter_output_dir = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(iter_output_dir, exist_ok=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=iter_output_dir,
        max_steps=num_train_steps,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{iter_output_dir}/logs",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=max(num_train_steps // 2, 1) if num_train_steps > 10 else num_train_steps,
        save_total_limit=1,
        report_to="none",  # Disable wandb/tensorboard unless configured
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_drop_last=False,  # Don't drop incomplete batches
        remove_unused_columns=False,  # Keep all columns for SFTTrainer
    )
    
    if debug:
        print(f"Training arguments:")
        print(f"  Max steps: {num_train_steps}")
        print(f"  Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
        print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Output dir: {iter_output_dir}")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=finetuning_dataset,
        args=training_args,
        processing_class=tokenizer,
        # No PEFT config for full fine-tuning
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed.")
    
    # Save the fine-tuned model
    model_save_path = os.path.join(iter_output_dir, "model")
    os.makedirs(model_save_path, exist_ok=True)
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"Fine-tuned model saved to: {model_save_path}")
    
    # Save training metadata
    metadata = {
        'iteration': iteration,
        'dataset_type': dataset_type,
        'base_model': BASE_MODEL_ID,
        'num_train_steps': num_train_steps,
        'num_training_examples': len(finetuning_data),
        'training_args': training_args.to_dict(),
    }
    
    metadata_file = os.path.join(iter_output_dir, "training_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training metadata saved to: {metadata_file}")
    
    # Clean up
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return model_save_path

def load_training_stats(rationales_dir, dataset_type, iteration):
    """Load training statistics from the collection phase."""
    iter_dir = os.path.join(rationales_dir, f"iteration_{iteration}")
    stats_file = os.path.join(iter_dir, f"{dataset_type}_collection_stats.json")
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune model on collected STaR rationales.")
    parser.add_argument("--rationales_dir", type=str, required=True, help="Directory containing collected rationales")
    parser.add_argument("--dataset", type=str, choices=['cqa', 'gsm8k', 'arithmetic'], required=True, help="Dataset type")
    parser.add_argument("--iteration", type=int, required=True, help="Current iteration number")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fine-tuned model")
    parser.add_argument("--num_train_steps", type=int, default=None, help="Number of training steps (default: calculated from iteration)")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    
    args = parser.parse_args()
    
    # Load and print collection statistics if available
    stats = load_training_stats(args.rationales_dir, args.dataset, args.iteration)
    if stats:
        print(f"\n=== Collection Statistics for Iteration {args.iteration} ===")
        print(f"Total examples processed: {stats.get('total_processed', 'N/A')}")
        print(f"Generated rationales: {stats.get('generated_rationales_count', 'N/A')}")
        print(f"Rationalized rationales: {stats.get('rationalized_rationales_count', 'N/A')}")
        print(f"Total for fine-tuning: {stats.get('generated_rationales_count', 0) + stats.get('rationalized_rationales_count', 0)}")
        print("=" * 50)
    
    model_path = finetune_model(
        rationales_dir=args.rationales_dir,
        dataset_type=args.dataset,
        iteration=args.iteration,
        output_dir=args.output_dir,
        num_train_steps=args.num_train_steps,
        debug=args.debug
    )
    
    print(f"\nFine-tuning completed for iteration {args.iteration}")
    print(f"Model saved to: {model_path}") 