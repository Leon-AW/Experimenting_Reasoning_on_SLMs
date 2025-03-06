# main.py

import argparse
import os
import csv
import sys
import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import random
import numpy as np

# Change relative imports to absolute imports
from reasoning_methods.prompting.config import (
    SEED, 
    DATASET_CONFIGS, 
    SELF_CONSISTENCY_PATHS, 
    PROMPT_TEMPLATES,
    MIN_NEW_TOKENS,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    DO_SAMPLE,
    NUM_RETURN_SEQUENCES
)
from reasoning_methods.prompting.dataset_utils import configure_hardware, load_custom_dataset
from reasoning_methods.prompting.process_dataset_batch import process_dataset_batch

def run_experiment(pipe, dataset, dataset_key, template_name, current_args, batch_size, model_name, device, num_gpus, max_memory):
    """Run a single experiment (one dataset + one template with given self-consistency flag)
    and write out the CSV and TXT results.
    Returns a summary dictionary for the experiment.
    """
    try:
        correct, total, results = process_dataset_batch(pipe, dataset, template_name, current_args, batch_size)
        
        final_accuracy = correct / total if total > 0 else 0.0
        print(f"Final Accuracy of {template_name} on {dataset_key} with model {current_args.model_size} (self-consistency: {current_args.self_consistency}): {final_accuracy:.2%}")
        print(f"Total Correct Answers: {correct}/{total} Questions")
        
        # Save results and metrics
        os.makedirs('results', exist_ok=True)
        sc_info = f"_sc{SELF_CONSISTENCY_PATHS}" if current_args.self_consistency else ""
        csv_file_path = os.path.join('results', f'{dataset_key}_{template_name}_{current_args.model_size}{sc_info}_results.csv')
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["sample_index", "question", "prompt", "generated_text", "pred_answer", "gold_answer", "is_correct"])
            writer.writeheader()
            writer.writerows(results)

        txt_file_path = os.path.join('results', f'{dataset_key}_{template_name}_{current_args.model_size}{sc_info}_total_accuracy.txt')
        with open(txt_file_path, mode='w') as file:
            file.write(f"Final Accuracy of {template_name} on {dataset_key} (self-consistency: {current_args.self_consistency}): {final_accuracy:.2%}\n")
            file.write(f"Total Correct Answers: {correct}/{total} Questions\n")
            file.write(f"\nUnextracted Answers: {total - len(results)} samples\n")
            if total > len(results):
                file.write(f"Number of unextracted answers: {total - len(results)}\n")
            file.write("\nHyperparameters:\n")
            file.write(f"Model: {model_name}\n")
            file.write(f"Batch Size: {batch_size}\n")
            file.write(f"Number of GPUs: {num_gpus}\n")
            file.write(f"Max Memory per GPU: {max_memory}\n")
            file.write(f"Min New Tokens: {MIN_NEW_TOKENS}\n")
            file.write(f"Max New Tokens: {MAX_NEW_TOKENS}\n")
            file.write(f"Temperature: {TEMPERATURE}\n")
            file.write(f"Top P: {TOP_P}\n")
            file.write(f"Top K: {TOP_K}\n")
            file.write(f"Do Sample: {DO_SAMPLE}\n")
            file.write(f"Number of Return Sequences: {NUM_RETURN_SEQUENCES}\n")
            file.write(f"Self Consistency: {current_args.self_consistency}\n")
            if current_args.self_consistency:
                file.write(f"Self Consistency Paths: {SELF_CONSISTENCY_PATHS}\n")
            file.write(f"Device: {device}\n")
            file.write(f"Random Seed: {SEED}\n")

        print(f"Completed template: {template_name}")
        print("Moving to next template...\n")
        
        # Return summary for use in a final comparison table
        return {
            'dataset': dataset_key,
            'template': template_name,
            'model_size': current_args.model_size,
            'self_consistency': current_args.self_consistency,
            'accuracy': final_accuracy,
            'correct': correct,
            'total': total
        }
    except Exception as e:
        print(f"Error processing template {template_name}: {str(e)}")
        print("Moving to next template...")
        return None


def main():
    # Set all the required seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # For best performance with deterministic results, set benchmark to True
    # but deterministic to False - this still maintains reproducibility with fixed seed
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 for faster computation on Ampere GPUs (A100, etc.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gsm8k',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to evaluate on')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--model_size', type=str, default='1b',
                        choices=['1b', '3b'],
                        help='LLaMA model size (1b or 3b)')
    parser.add_argument('--self_consistency', action='store_true',
                        help='Enable self-consistency with multiple paths')
    parser.add_argument('--template', type=str, default=None,
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help='Specific prompt template to evaluate (default: all templates)')
    parser.add_argument('--optimize', action='store_true',
                        help='Enable additional optimizations for inference speed')
    args = parser.parse_args()

    # Set environment variable for model size to optimize batch size in dataset_utils
    os.environ['MODEL_SIZE'] = args.model_size

    # If no additional CLI arguments (besides the module name) are provided,
    # then perform a full sweep over datasets, models, templates, and SC settings.
    full_sweep = len(sys.argv) == 1
    summary_results = []  # to accumulate experiment summaries

    if full_sweep:
        print("Running full sweep (all datasets, all prompt templates, all self-consistency settings, both model sizes)")
        for model_size in ["1b", "3b"]:
            print(f"\n{'#'*50}\nRunning experiments for model size: {model_size}\n{'#'*50}\n")
            # Configure hardware and load model/tokenizer for this model_size
            device, batch_size, num_gpus, max_memory = configure_hardware()
            print(f"Detected device: {device}")
            print(f"Using hyperparameters: BATCH_SIZE={batch_size}, NUM_GPUS={num_gpus}, MAX_MEMORY={max_memory}")
            
            MODEL_NAME = f"meta-llama/Llama-3.2-{model_size.upper()}"
            
            # Advanced model loading optimizations
            print(f"Loading model {MODEL_NAME} with optimized settings...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance/accuracy balance
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,      # Reduce CPU memory usage during loading
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,  # Use flash attention if available
            )
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            # Optimize tokenizer settings
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = 'left'  # Left padding for more efficient batching
            
            # Set model to evaluation mode
            model.eval()
            
            # Use model parallelism for efficient multi-GPU usage
            if num_gpus > 1:
                # If model is already using device_map="auto", don't wrap with DataParallel
                if hasattr(model, 'hf_device_map'):
                    print(f"Using HF's native device mapping across {num_gpus} GPUs")
                    model_for_pipeline = model
                else:
                    print(f"Using PyTorch DataParallel across {num_gpus} GPUs")
                    model = torch.nn.DataParallel(model)
                    model_for_pipeline = model.module
            else:
                model_for_pipeline = model

            # Create optimized pipeline
            pipe = pipeline(
                "text-generation",
                model=model_for_pipeline,
                tokenizer=tokenizer,
                use_cache=True,
                batch_size=batch_size,
                num_workers=min(4, os.cpu_count() // 2),  # Optimize worker threads based on CPU cores
            )
            
            # Ensure tokenizer settings are applied to pipeline
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
            pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
            pipe.tokenizer.padding_side = 'left'

            # Loop over every dataset
            for dataset_key, dataset_config in DATASET_CONFIGS.items():
                print(f"\n{'='*50}\nDataset: {dataset_key}\n{'='*50}\n")
                dataset = load_custom_dataset(dataset_config)
                # Loop over self-consistency settings: first False then True
                for sc in [False, True]:
                    print(f"\n{'-'*50}\nSelf Consistency: {sc}\n{'-'*50}\n")
                    # Loop over every prompt template available
                    for template_name in PROMPT_TEMPLATES.keys():
                        print(f"\n{'='*50}")
                        print(f"Starting template: {template_name} on dataset: {dataset_key} with model: {model_size} and self-consistency: {sc}")
                        print(f"{'='*50}\n")
                        # Create a new args namespace for the current run and override values as needed.
                        current_args = argparse.Namespace(**vars(args))
                        current_args.dataset = dataset_key
                        current_args.model_size = model_size
                        current_args.self_consistency = sc
                        # If a specific template was passed via CLI, you might want to honor that; otherwise use all.
                        current_args.template = None

                        summary = run_experiment(pipe, dataset, dataset_key, template_name, current_args, batch_size, MODEL_NAME, device, num_gpus, max_memory)
                        if summary is not None:
                            summary_results.append(summary)
            print(f"\nCompleted experiments for model size {model_size}\n")
        print("\nCompleted all experiments!")
        
        # Print a summary table for all experiments.
        if summary_results:
            print("\nSummary Table of Results:\n")
            header = f"{'Dataset':<10} {'Template':<12} {'Model':<6} {'SC':<6} {'Acc %':<8} {'Correct':<8} {'Total':<8}"
            print(header)
            print("-" * len(header))
            for entry in summary_results:
                print(f"{entry['dataset']:<10} {entry['template']:<12} {entry['model_size']:<6} {str(entry['self_consistency']):<6} {entry['accuracy']*100:<8.2f} {entry['correct']:<8} {entry['total']:<8}")
    else:
        # Normal behavior based on provided CLI options.
        device, batch_size, num_gpus, max_memory = configure_hardware()
        print(f"Detected device: {device}")
        print(f"Using hyperparameters: BATCH_SIZE={batch_size}, NUM_GPUS={num_gpus}, MAX_MEMORY={max_memory}")

        MODEL_NAME = f"meta-llama/Llama-3.2-{args.model_size.upper()}"
        
        # Advanced model loading optimizations
        print(f"Loading model {MODEL_NAME} with optimized settings...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance/accuracy balance
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,      # Reduce CPU memory usage during loading
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,  # Use flash attention if available
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Optimize tokenizer settings
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'  # Left padding for more efficient batching
        
        # Set model to evaluation mode
        model.eval()
        
        # Use model parallelism for efficient multi-GPU usage
        if num_gpus > 1:
            # If model is already using device_map="auto", don't wrap with DataParallel
            if hasattr(model, 'hf_device_map'):
                print(f"Using HF's native device mapping across {num_gpus} GPUs")
                model_for_pipeline = model
            else:
                print(f"Using PyTorch DataParallel across {num_gpus} GPUs")
                model = torch.nn.DataParallel(model)
                model_for_pipeline = model.module
        else:
            model_for_pipeline = model

        # Create optimized pipeline
        pipe = pipeline(
            "text-generation",
            model=model_for_pipeline,
            tokenizer=tokenizer,
            use_cache=True,
            batch_size=batch_size,
            num_workers=min(4, os.cpu_count() // 2),  # Optimize worker threads based on CPU cores
        )
        
        # Ensure tokenizer settings are applied to pipeline
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        pipe.tokenizer.padding_side = "left"

        dataset_config = DATASET_CONFIGS[args.dataset]
        dataset = load_custom_dataset(dataset_config)

        templates_to_evaluate = [args.template] if args.template else PROMPT_TEMPLATES.keys()

        for template_name in templates_to_evaluate:
            print(f"\n{'='*50}")
            print(f"Starting template: {template_name}")
            print(f"{'='*50}\n")
            
            try:
                correct, total, results = process_dataset_batch(pipe, dataset, template_name, args, batch_size)
                final_accuracy = correct / total if total > 0 else 0.0
                print(f"Final Accuracy of {template_name} on {args.dataset}: {final_accuracy:.2%}")
                print(f"Total Correct Answers: {correct}/{total} Questions")
                
                os.makedirs('results', exist_ok=True)
                sc_info = f"_sc{SELF_CONSISTENCY_PATHS}" if args.self_consistency else ""
                csv_file_path = os.path.join('results', f'{args.dataset}_{template_name}_{args.model_size}{sc_info}_results.csv')
                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=["sample_index", "question", "prompt", "generated_text", "pred_answer", "gold_answer", "is_correct"])
                    writer.writeheader()
                    writer.writerows(results)

                txt_file_path = os.path.join('results', f'{args.dataset}_{template_name}_{args.model_size}{sc_info}_total_accuracy.txt')
                with open(txt_file_path, mode='w') as file:
                    file.write(f"Final Accuracy of {template_name} on {args.dataset}: {final_accuracy:.2%}\n")
                    file.write(f"Total Correct Answers: {correct}/{total} Questions\n")
                    file.write(f"\nUnextracted Answers: {total - len(results)} samples\n")
                    if total > len(results):
                        file.write(f"Number of unextracted answers: {total - len(results)}\n")
                    file.write("\nHyperparameters:\n")
                    file.write(f"Model: {MODEL_NAME}\n")
                    file.write(f"Batch Size: {batch_size}\n")
                    file.write(f"Number of GPUs: {num_gpus}\n")
                    file.write(f"Max Memory per GPU: {max_memory}\n")
                    file.write(f"Min New Tokens: {MIN_NEW_TOKENS}\n")
                    file.write(f"Max New Tokens: {MAX_NEW_TOKENS}\n")
                    file.write(f"Temperature: {TEMPERATURE}\n")
                    file.write(f"Top P: {TOP_P}\n")
                    file.write(f"Top K: {TOP_K}\n")
                    file.write(f"Do Sample: {DO_SAMPLE}\n")
                    file.write(f"Number of Return Sequences: {NUM_RETURN_SEQUENCES}\n")
                    file.write(f"Self Consistency: {args.self_consistency}\n")
                    if args.self_consistency:
                        file.write(f"Self Consistency Paths: {SELF_CONSISTENCY_PATHS}\n")
                    file.write(f"Device: {device}\n")
                    file.write(f"Random Seed: {SEED}\n")
                print(f"\nCompleted template: {template_name}")
                print("Moving to next template...")
            except Exception as e:
                print(f"Error processing template {template_name}: {str(e)}")
                print("Moving to next template...")
                continue

        print("\nCompleted all templates!")

if __name__ == "__main__":
    main() 