# reasoning_methods/hybrid/finetune_mixed.py

import argparse
import os
import json
import torch
import gc
import random
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

# Fine-tuning hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
LOGGING_STEPS = 10
NUM_TRAIN_EPOCHS = 1.0 # Train for one full pass over the data

def load_model_and_tokenizer(model_id_or_path):
    """Loads the model and tokenizer for fine-tuning."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Move model to GPU (SFTTrainer will handle multi-GPU training)
    if torch.cuda.is_available():
        print(f"CUDA available with {torch.cuda.device_count()} GPUs.")
        model.to(torch.device("cuda"))
    else:
        print("CUDA not available. Using CPU.")
        model.to(torch.device("cpu"))

    return model, tokenizer

def load_and_mix_rationales(rationales_dir, dataset_types, iteration):
    """
    Load and mix collected rationales from multiple datasets.
    
    Args:
        rationales_dir: Directory containing collected rationales
        dataset_types: List of dataset types to include
        iteration: Iteration number
        
    Returns:
        A shuffled list of formatted text instances for fine-tuning
    """
    iter_dir = os.path.join(rationales_dir, f"iteration_{iteration}")
    
    finetuning_data = []
    
    print(f"Loading rationales from: {iter_dir}")
    for dataset_type in dataset_types:
        rationales_file = os.path.join(iter_dir, f"{dataset_type}_rationales.jsonl")
        count = 0
        if os.path.exists(rationales_file):
            with open(rationales_file, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    finetuning_data.append(item['formatted_text'])
                    count += 1
            print(f"  - Loaded {count} rationales from {dataset_type}")
        else:
            print(f"  - Warning: Rationales file not found for {dataset_type}: {rationales_file}")
    
    if not finetuning_data:
        raise ValueError(f"No rationales found for any specified dataset in {iter_dir}")
        
    print(f"\nTotal rationales loaded: {len(finetuning_data)}")
    
    # Shuffle the combined dataset
    random.shuffle(finetuning_data)
    print("Combined dataset has been shuffled.")
    
    return finetuning_data

def finetune_mixed_model(
    rationales_dir: str,
    iteration: int,
    output_dir: str,
    debug: bool = False
):
    """
    Fine-tune the base model on a mixed dataset of collected rationales.
    
    Args:
        rationales_dir: Directory containing collected rationales
        iteration: Current iteration number to load rationales from
        output_dir: Directory to save the fine-tuned model
        debug: Enable debug printing
        
    Returns:
        Path to the saved fine-tuned model
    """
    print(f"\n=== Fine-tuning Model on Mixed Rationales from Iteration {iteration} ===")
    print(f"Base model: {BASE_MODEL_ID}")
    
    # Load and mix rationales from all datasets
    dataset_types = ['cqa', 'gsm8k', 'arithmetic']
    finetuning_data = load_and_mix_rationales(rationales_dir, dataset_types, iteration)
    
    # Create dataset for SFTTrainer
    finetuning_dataset = Dataset.from_dict({"text": finetuning_data})
    
    # Load model for fine-tuning
    print(f"Loading base model for fine-tuning: {BASE_MODEL_ID}")
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_ID)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up training arguments using SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{output_dir}/logs",
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_drop_last=True, # Drop last incomplete batch to have stable training dynamics
        max_seq_length=1024, # Max sequence length for the model
    )
    
    if debug:
        print(f"Training arguments:")
        print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
        print(f"  Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
        print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Output dir: {output_dir}")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=finetuning_dataset,
        args=training_args,
    )
    
    print("Starting fine-tuning on mixed dataset...")
    trainer.train()
    print("Fine-tuning completed.")
    
    # Save the fine-tuned model
    model_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(model_save_path)
    
    print(f"Fine-tuned mixed-task model saved to: {model_save_path}")
    
    # Save training metadata
    metadata = {
        'iteration_source': iteration,
        'dataset_sources': dataset_types,
        'base_model': BASE_MODEL_ID,
        'num_training_examples': len(finetuning_data),
        'training_args': training_args.to_dict(),
    }
    
    metadata_file = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training metadata saved to: {metadata_file}")
    
    # Clean up
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return model_save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on a mixed set of STaR rationales.")
    parser.add_argument("--rationales_dir", type=str, default="reasoning_methods/hybrid/collected_rationales",
                        help="Directory containing collected rationales")
    parser.add_argument("--iteration", type=int, default=1,
                        help="Iteration number to load rationales from")
    parser.add_argument("--output_dir", type=str, default="star_models/mixed_model_iter1",
                        help="Directory to save the fine-tuned mixed-task model")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug printing")
    
    args = parser.parse_args()
    
    finetune_mixed_model(
        rationales_dir=args.rationales_dir,
        iteration=args.iteration,
        output_dir=args.output_dir,
        debug=args.debug
    )
    
    print(f"\nFine-tuning on mixed dataset completed.")
    print(f"Final model saved in: {args.output_dir}") 