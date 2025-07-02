# main.py

import argparse
import os
import csv
import sys
import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import random
import numpy as np
import signal

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

# Define a custom exception for skipping datasets that inherits from BaseException.
# This ensures it's not caught by general 'except Exception' blocks.
class SkipDataset(BaseException):
    pass

# Signal handler that raises the custom exception.
def signal_handler(signum, frame):
    """Raise a SkipDataset exception when a SIGINT is received."""
    print("\nCtrl+C detected. Attempting to skip to the next dataset...")
    raise SkipDataset()

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
        os.makedirs('reasoning_methods/prompting/debug_csvs', exist_ok=True)
        os.makedirs('reasoning_methods/prompting/results', exist_ok=True)
        sc_info = f"_sc{SELF_CONSISTENCY_PATHS}" if current_args.self_consistency else ""
        csv_file_path = os.path.join('reasoning_methods/prompting/debug_csvs', f'{dataset_key}_{template_name}_{current_args.model_size}{sc_info}_results.csv')
        
        # Define fieldnames based on whether self-consistency is enabled
        fieldnames = ["sample_index", "question", "prompt", "generated_text", "pred_answer", "gold_answer", "is_correct"]
        if "passage" in results[0] and results[0]["passage"]:
            fieldnames.insert(2, "passage")  # Add passage field if it exists in results
        
        if current_args.self_consistency:
            fieldnames.extend(["confidence", "sc_paths"])
        
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

        txt_file_path = os.path.join('reasoning_methods/prompting/results', f'{dataset_key}_{template_name}_{current_args.model_size}{sc_info}_total_accuracy.txt')
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

    # Store the original SIGINT handler to restore it on exit
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='gsm8k',
                            choices=['gsm8k', 'race', 'arc', 'mmlu', 'drop', 'commonsense_qa'],
                            help='Dataset to evaluate on')
        parser.add_argument('--debug', action='store_true',
                            help='Enable debug mode')
        parser.add_argument('--model_size', type=str, default='1b',
                            choices=['1b', '3b', '1b-instruct', '1b-sft-full', '1b-sft-lora', '1b-sft-lora-all', '1b-sft-full-slimorca-100k', '1b-sft-mixed-best', 'star_model1'],
                            help='LLaMA model size or fine-tuned variant')
        parser.add_argument('--self_consistency', action='store_true',
                            help='Enable self-consistency with multiple paths')
        parser.add_argument('--template', type=str, default=None,
                            choices=list(PROMPT_TEMPLATES.keys()),
                            help='Specific prompt template to evaluate (default: all templates)')
        args = parser.parse_args()

        # Set environment variable for model size to optimize batch size in dataset_utils
        # For SFT models, use the base model size for batch sizing
        base_model_size = args.model_size.split('-')[0]
        os.environ['MODEL_SIZE'] = base_model_size

        # Determine if a sweep should be performed. A sweep is triggered if a specific
        # dataset or template is not provided via the command line.
        is_sweep_run = '--dataset' not in sys.argv and '--template' not in sys.argv

        # A full sweep over all models is triggered if no specific model is provided either.
        is_full_sweep = is_sweep_run and '--model_size' not in sys.argv
        
        summary_results = []  # to accumulate experiment summaries

        if is_sweep_run:
            models_to_run = []
            if is_full_sweep:
                print("Running full sweep over all models, datasets, templates, and SC settings.")
                models_to_run = ["1b", "3b", "1b-instruct", "1b-sft-full", "1b-sft-lora", "1b-sft-lora-all", "1b-sft-full-slimorca-100k", "1b-sft-mixed-best", "star_model1"]
            else:
                print(f"Running sweep for model '{args.model_size}' over all datasets, templates, and SC settings.")
                models_to_run = [args.model_size]

            for model_size in models_to_run:
                print(f"\n{'#'*50}\nRunning experiments for model size: {model_size}\n{'#'*50}\n")
                
                # We create a temporary args object to pass to configure_hardware,
                # as the dataset and SC settings change within the loops.
                # The batch size needs to be determined for the most demanding configuration.
                temp_args_for_batch_size = argparse.Namespace(**vars(args))
                temp_args_for_batch_size.self_consistency = True # Assume worst-case for SC
                temp_args_for_batch_size.dataset = 'drop' # Assume worst-case for dataset length

                # Configure hardware and load model/tokenizer for this model_size
                device, batch_size, num_gpus, max_memory, usable_gpus_indices = configure_hardware(temp_args_for_batch_size)
                print(f"Detected device: {device}")
                print(f"Using hyperparameters: BATCH_SIZE={batch_size}, NUM_GPUS={num_gpus}, MAX_MEMORY={max_memory}")
                
                # Determine model path based on model_size
                if model_size == "1b-sft-full":
                    MODEL_NAME = "reasoning_methods/fine-tuning/Llama-3.2-1B-SFT-Full"
                elif model_size == "1b-sft-lora":
                    MODEL_NAME = "reasoning_methods/fine-tuning/Llama-3.2-1B-SFT-LoRA"
                elif model_size == "1b-sft-lora-all":
                    MODEL_NAME = "reasoning_methods/fine-tuning/Llama-3.2-1B-SFT-LoRA-All"
                elif model_size == "1b-sft-full-slimorca-100k":
                    MODEL_NAME = "reasoning_methods/fine-tuning/Llama-3.2-1B-SFT-Full-SlimOrca-100k"
                elif model_size == "1b-sft-mixed-best":
                    MODEL_NAME = "reasoning_methods/fine-tuning/Llama-3.2-1B-SFT-Mixed-Reasoning/checkpoint-2000"
                elif model_size == "star_model1":
                    MODEL_NAME = "reasoning_methods/hybrid/star_models/iteration_1/final_model"
                elif model_size == "1b-instruct":
                    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
                else:
                    MODEL_NAME = f"meta-llama/Llama-3.2-{model_size.upper()}"
                
                # Advanced model loading optimizations
                print(f"Loading model {MODEL_NAME} with optimized settings...")
                
                # Try to load with flash_attention_2 first, fallback to sdpa if not available
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME,
                        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance/accuracy balance
                        device_map="auto",
                        max_memory=max_memory,
                        low_cpu_mem_usage=True,      # Reduce CPU memory usage during loading
                        attn_implementation="flash_attention_2",  # Try Flash Attention 2 first
                    )
                    print("Successfully loaded model with Flash Attention 2")
                except Exception as e:
                    print(f"Flash Attention 2 not available ({str(e)}), falling back to SDPA")
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME,
                        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance/accuracy balance
                        device_map="auto",
                        max_memory=max_memory,
                        low_cpu_mem_usage=True,      # Reduce CPU memory usage during loading
                        attn_implementation="sdpa",  # Fallback to SDPA (Scaled Dot Product Attention)
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
                        model = torch.nn.DataParallel(model, device_ids=usable_gpus_indices)
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
                pipe.tokenizer.padding_side = 'left'  # Left padding for more efficient batching

                # Determine SC settings for this model sweep
                sc_settings_for_sweep = []
                if args.self_consistency:
                    sc_settings_for_sweep = [True]
                else:
                    if model_size in ["1b-sft-full", "1b-sft-lora", "1b-sft-lora-all", "1b-sft-full-slimorca-100k", "1b-sft-mixed-best"]:
                        sc_settings_for_sweep = [False]  # Fine-tuned models only run without SC by default
                    elif model_size == "1b-instruct":
                        sc_settings_for_sweep = [False]  # Instruct model only runs without self-consistency
                    elif model_size == "3b":
                        sc_settings_for_sweep = [False]  # For 3b run only False
                    else:
                        sc_settings_for_sweep = [False, True]  # For 1b and star_model1 run both

                # Loop over every dataset
                for dataset_key, dataset_config in DATASET_CONFIGS.items():
                    try:
                        print(f"\n{'='*50}\nDataset: {dataset_key}\n{'='*50}\n")
                        dataset = load_custom_dataset(dataset_config)
                        
                        for sc in sc_settings_for_sweep:
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
                    except SkipDataset:
                        print(f"--- SKIPPED DATASET: {dataset_key} ---")
                        continue
                print(f"\nCompleted experiments for model size {model_size}\n")
            print("\nCompleted all experiments!")
            
            # Print a summary table for all experiments.
            if summary_results:
                print("\nSummary Table of Results:\n")
                header = f"{'Dataset':<10} {'Template':<12} {'Model':<12} {'SC':<6} {'Acc %':<8} {'Correct':<8} {'Total':<8}"
                print(header)
                print("-" * len(header))
                for entry in summary_results:
                    print(f"{entry['dataset']:<10} {entry['template']:<12} {entry['model_size']:<12} {str(entry['self_consistency']):<6} {entry['accuracy']*100:<8.2f} {entry['correct']:<8} {entry['total']:<8}")
        else:
            # Normal behavior based on provided CLI options.
            device, batch_size, num_gpus, max_memory, usable_gpus_indices = configure_hardware(args)
            print(f"Detected device: {device}")
            print(f"Using hyperparameters: BATCH_SIZE={batch_size}, NUM_GPUS={num_gpus}, MAX_MEMORY={max_memory}")

            # Determine model path based on model_size
            if args.model_size == "1b-sft-full":
                MODEL_NAME = "reasoning_methods/fine-tuning/Llama-3.2-1B-SFT-Full"
            elif args.model_size == "1b-sft-lora":
                MODEL_NAME = "reasoning_methods/fine-tuning/Llama-3.2-1B-SFT-LoRA"
            elif args.model_size == "1b-sft-lora-all":
                MODEL_NAME = "reasoning_methods/fine-tuning/Llama-3.2-1B-SFT-LoRA-All"
            elif args.model_size == "1b-sft-full-slimorca-100k":
                MODEL_NAME = "reasoning_methods/fine-tuning/Llama-3.2-1B-SFT-Full-SlimOrca-100k"
            elif args.model_size == "1b-sft-mixed-best":
                MODEL_NAME = "reasoning_methods/fine-tuning/Llama-3.2-1B-SFT-Mixed-Reasoning/checkpoint-2000"
            elif args.model_size == "star_model1":
                MODEL_NAME = "reasoning_methods/hybrid/star_models/iteration_1/final_model"
            elif args.model_size == "1b-instruct":
                MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
            else:
                MODEL_NAME = f"meta-llama/Llama-3.2-{args.model_size.upper()}"
            
            # Advanced model loading optimizations
            print(f"Loading model {MODEL_NAME} with optimized settings...")
            
            # Try to load with flash_attention_2 first, fallback to sdpa if not available
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance/accuracy balance
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,      # Reduce CPU memory usage during loading
                    attn_implementation="flash_attention_2",  # Try Flash Attention 2 first
                )
                print("Successfully loaded model with Flash Attention 2")
            except Exception as e:
                print(f"Flash Attention 2 not available ({str(e)}), falling back to SDPA")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance/accuracy balance
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,      # Reduce CPU memory usage during loading
                    attn_implementation="sdpa",  # Fallback to SDPA (Scaled Dot Product Attention)
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
                    model = torch.nn.DataParallel(model, device_ids=usable_gpus_indices)
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
                    
                    os.makedirs('reasoning_methods/prompting/debug_csvs', exist_ok=True)
                    os.makedirs('reasoning_methods/prompting/results', exist_ok=True)
                    sc_info = f"_sc{SELF_CONSISTENCY_PATHS}" if args.self_consistency else ""
                    csv_file_path = os.path.join('reasoning_methods/prompting/debug_csvs', f'{args.dataset}_{template_name}_{args.model_size}{sc_info}_results.csv')
                    with open(csv_file_path, mode='w', newline='') as file:
                        # Add sc_paths to fieldnames if self-consistency is enabled
                        fieldnames = ["sample_index", "question", "prompt", "generated_text", "pred_answer", "gold_answer", "is_correct"]
                        if args.self_consistency:
                            fieldnames.append("sc_paths")
                            fieldnames.append("confidence")  # Changed from sc_confidences to confidence
                        
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        # Write results, converting sc_paths to string if present
                        for result in results:
                            row = {k: v for k, v in result.items() if k in fieldnames}
                            if args.self_consistency and "sc_paths" in result:
                                # Extract confidence as the weighted average if CISC is enabled, otherwise as frequency
                                if "confidence" in result:
                                    row["confidence"] = result["confidence"]
                                
                                # Convert sc_paths to string representation
                                row["sc_paths"] = str(result["sc_paths"])
                            
                            writer.writerow(row)

                    txt_file_path = os.path.join('reasoning_methods/prompting/results', f'{args.dataset}_{template_name}_{args.model_size}{sc_info}_total_accuracy.txt')
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

    except SkipDataset:
        print("\nExecution interrupted by user.")
    finally:
        # Restore the original SIGINT handler to ensure normal Ctrl+C behavior afterwards
        print("\nExecution finished. Restoring original signal handler.")
        signal.signal(signal.SIGINT, original_sigint_handler)

if __name__ == "__main__":
    main() 