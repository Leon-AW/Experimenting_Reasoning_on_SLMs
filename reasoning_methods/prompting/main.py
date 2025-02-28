# main.py

import argparse
import os
import csv
import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer

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
from reasoning_methods.prompting.evaluator import process_dataset_batch


def main():
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
                       help='Enable self-consistency with 15 paths')
    parser.add_argument('--template', type=str, default=None,
                       choices=list(PROMPT_TEMPLATES.keys()),
                       help='Specific template to evaluate (default: evaluate all templates)')
    args = parser.parse_args()

    # Configure hardware
    device, batch_size, num_gpus, max_memory = configure_hardware()
    print(f"Detected device: {device}")
    print(f"Using hyperparameters: BATCH_SIZE={batch_size}, NUM_GPUS={num_gpus}, MAX_MEMORY={max_memory}")

    # Load model & tokenizer
    MODEL_NAME = f"meta-llama/Llama-3.2-{args.model_size.upper()}"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="balanced" if num_gpus > 1 else "auto",
        max_memory={i: max_memory for i in range(num_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        batch_size=batch_size * num_gpus,
    )
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'

    # Load dataset
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset = load_custom_dataset(dataset_config)

    # Evaluate each template or just the specified one
    templates_to_evaluate = [args.template] if args.template else PROMPT_TEMPLATES.keys()
    
    for template_name in templates_to_evaluate:
        print(f"\n{'='*50}")
        print(f"Starting template: {template_name}")
        print(f"{'='*50}\n")
        
        try:
            correct, total, results = process_dataset_batch(pipe, dataset, template_name, args, batch_size)
            
            # Save results and print metrics
            final_accuracy = correct / total if total > 0 else 0.0
            print(f"Final Accuracy of {template_name} on {args.dataset}: {final_accuracy:.2%}")
            print(f"Total Correct Answers: {correct}/{total} Questions")
            
            # Save results to files
            os.makedirs('results', exist_ok=True)
            
            # Save to CSV with sample index
            sc_info = f"_sc{SELF_CONSISTENCY_PATHS}" if args.self_consistency else ""
            csv_file_path = os.path.join('results', f'{args.dataset}_{template_name}_{args.model_size}{sc_info}_results.csv')
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["sample_index", "question", "prompt", "generated_text", "pred_answer", "gold_answer", "is_correct"])
                writer.writeheader()
                writer.writerows(results)

            # Save accuracy and unextracted answers info with hyperparameters
            txt_file_path = os.path.join('results', f'{args.dataset}_{template_name}_{args.model_size}{sc_info}_total_accuracy.txt')
            with open(txt_file_path, mode='w') as file:
                # Write accuracy results
                file.write(f"Final Accuracy of {template_name} on {args.dataset}: {final_accuracy:.2%}\n")
                file.write(f"Total Correct Answers: {correct}/{total} Questions\n")
                file.write(f"\nUnextracted Answers: {total - len(results)} samples\n")
                if total > len(results):
                    file.write(f"Number of unextracted answers: {total - len(results)}\n")
                
                # Write hyperparameters
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
            print(f"Moving to next template...")

        except Exception as e:
            print(f"Error processing template {template_name}: {str(e)}")
            print("Moving to next template...")
            continue  # Continue to next template on error

    print("\nCompleted all templates!")


if __name__ == "__main__":
    main() 