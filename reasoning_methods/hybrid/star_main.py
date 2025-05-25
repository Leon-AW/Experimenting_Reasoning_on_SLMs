# reasoning_methods/hybrid/star_main.py

import argparse
import os
import json
import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
import subprocess

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting import (
    generate_rationale, 
    format_question, 
    TEMPERATURE, 
    TOP_P, 
    TOP_K, 
    DO_SAMPLE, 
    MAX_NEW_TOKENS,
    SEED,
)
from prepare_datasets import load_commonsense_qa, load_gsm8k, generate_arithmetic_dataset
from collect_rationales import collect_rationales_for_iteration, load_few_shot_prompt, get_ground_truth, load_model_and_tokenizer
from finetune import finetune_model

# Configuration
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
DATA_CACHE_DIR = './data_cache'
NUM_STAR_ITERATIONS = 5

def evaluate_model(
    model_path: str,
    dataset_type: str,
    max_samples: int = None,
    debug: bool = False
):
    """
    Evaluate a model on the test/validation set.
    
    Args:
        model_path: Path to the model to evaluate
        dataset_type: Type of dataset ('cqa', 'gsm8k', 'arithmetic')
        max_samples: Maximum number of samples to evaluate (for testing)
        debug: Enable debug printing
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n=== Evaluating Model ===")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_type}")
    
    # Load few-shot prompt
    few_shot_prompt = load_few_shot_prompt(dataset_type)
    
    # Load evaluation dataset
    print("Loading evaluation dataset...")
    if dataset_type == 'cqa':
        dataset = load_commonsense_qa(cache_dir=DATA_CACHE_DIR)
        eval_data = dataset['validation']
    elif dataset_type == 'gsm8k':
        dataset = load_gsm8k(cache_dir=DATA_CACHE_DIR)
        eval_data = dataset['test']
    elif dataset_type == 'arithmetic':
        # For arithmetic, we could create a separate test set or use a portion of the generated data
        print("Warning: Arithmetic dataset doesn't have a standard test set. Skipping evaluation.")
        return {'accuracy': 0.0, 'correct': 0, 'total': 0, 'note': 'No standard test set for arithmetic'}
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Apply max_samples limit if specified
    if max_samples is not None:
        eval_data = eval_data.select(range(min(max_samples, len(eval_data))))
        print(f"Using {len(eval_data)} examples for evaluation (limited by --max_samples).")
    
    # Load model
    print(f"Loading model: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()
    
    # Evaluate
    correct_count = 0
    total_count = 0
    
    progress_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}]"
    
    print(f"Evaluating on {len(eval_data)} examples...")
    for example in tqdm(eval_data, desc="Evaluation", bar_format=progress_bar_format):
        ground_truth = get_ground_truth(example, dataset_type)
        if ground_truth is None:
            continue
        
        # Generate rationale and answer
        r_eval, y_hat = generate_rationale(
            model=model,
            tokenizer=tokenizer,
            few_shot_prompt=few_shot_prompt,
            question_data=example,
            dataset_type=dataset_type,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=DO_SAMPLE,
        )
        
        # Compare answers
        if y_hat is not None:
            if dataset_type == 'cqa':
                is_correct = str(y_hat).lower() == str(ground_truth).lower()
            else:
                # For numeric datasets, normalize
                try:
                    y_hat_norm = str(int(float(str(y_hat))))
                    gt_norm = str(int(float(str(ground_truth))))
                    is_correct = y_hat_norm == gt_norm
                except ValueError:
                    is_correct = str(y_hat) == str(ground_truth)
        else:
            is_correct = False
        
        if debug and total_count < 5:  # Only show first few examples in debug mode
            print(f"\n--- Eval Debug Example {total_count} ---")
            print(f"Ground Truth: {ground_truth}")
            print(f"Generated Answer: {y_hat}")
            print(f"Correct: {is_correct}")
            print("--------------------------------------")
        
        if is_correct:
            correct_count += 1
        total_count += 1
    
    # Clean up model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Calculate results
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total_count,
        'dataset_type': dataset_type,
        'model_path': model_path
    }
    
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    print("==========================")
    
    return results

def run_star_process(
    dataset_type: str,
    num_iterations: int = NUM_STAR_ITERATIONS,
    max_samples: int = None,
    debug: bool = False,
    rationales_output_dir: str = "collected_rationales",
    models_output_dir: str = "star_models",
    evaluate_each_iteration: bool = True
):
    """
    Run the complete STaR process.
    
    This implements the full STaR algorithm from the paper:
    1. For each iteration n = 1 to N:
       a. Use model M_{n-1} to collect rationales (Steps 3-6)
       b. Fine-tune base model M_0 on collected rationales (Step 7) -> M_n
    2. Optionally evaluate each iteration
    
    Args:
        dataset_type: Type of dataset ('cqa', 'gsm8k', 'arithmetic')
        num_iterations: Number of STaR iterations
        max_samples: Maximum number of samples to process per iteration
        debug: Enable debug printing
        rationales_output_dir: Directory to save collected rationales
        models_output_dir: Directory to save fine-tuned models
        evaluate_each_iteration: Whether to evaluate after each iteration
        
    Returns:
        Dictionary with results from all iterations
    """
    print(f"\n{'='*60}")
    print(f"Starting STaR Process")
    print(f"Dataset: {dataset_type}")
    print(f"Base Model: {BASE_MODEL_ID}")
    print(f"Iterations: {num_iterations}")
    if max_samples:
        print(f"Max Samples per Iteration: {max_samples}")
    print(f"{'='*60}")
    
    # Create output directories
    os.makedirs(rationales_output_dir, exist_ok=True)
    os.makedirs(models_output_dir, exist_ok=True)
    
    # Track results across iterations
    all_results = {
        'dataset_type': dataset_type,
        'base_model': BASE_MODEL_ID,
        'num_iterations': num_iterations,
        'iterations': {}
    }
    
    # Initialize: M_0 is the base model
    current_model_path = BASE_MODEL_ID
    
    # STaR iterations
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"STaR Iteration {iteration}/{num_iterations}")
        print(f"{'='*60}")
        
        iteration_results = {
            'iteration': iteration,
            'generation_model': current_model_path
        }
        
        # Phase 1: Collect rationales using M_{n-1}
        print(f"\nPhase 1: Collecting rationales using model: {current_model_path}")
        
        try:
            collection_stats = collect_rationales_for_iteration(
                model_path=current_model_path,
                dataset_type=dataset_type,
                iteration=iteration,
                output_dir=rationales_output_dir,
                max_samples=max_samples,
                debug=debug
            )
            iteration_results['collection_stats'] = collection_stats
            
            # Check if we have enough data for fine-tuning
            total_rationales = collection_stats['generated_rationales_count'] + collection_stats['rationalized_rationales_count']
            if total_rationales == 0:
                print(f"Warning: No rationales collected for iteration {iteration}. Stopping STaR process.")
                break
                
        except Exception as e:
            print(f"Error in rationale collection for iteration {iteration}: {e}")
            iteration_results['collection_error'] = str(e)
            all_results['iterations'][iteration] = iteration_results
            break
        
        # Phase 2: Fine-tune base model M_0 on collected rationales
        print(f"\nPhase 2: Fine-tuning base model on collected rationales")
        
        try:
            model_save_path = finetune_model(
                rationales_dir=rationales_output_dir,
                dataset_type=dataset_type,
                iteration=iteration,
                output_dir=models_output_dir,
                debug=debug
            )
            iteration_results['model_path'] = model_save_path
            
            # Update current model for next iteration
            current_model_path = model_save_path
            
        except Exception as e:
            print(f"Error in fine-tuning for iteration {iteration}: {e}")
            iteration_results['finetuning_error'] = str(e)
            all_results['iterations'][iteration] = iteration_results
            break
        
        # Phase 3: Evaluate (optional)
        if evaluate_each_iteration:
            print(f"\nPhase 3: Evaluating iteration {iteration} model")
            
            try:
                eval_results = evaluate_model(
                    model_path=model_save_path,
                    dataset_type=dataset_type,
                    max_samples=max_samples,
                    debug=debug
                )
                iteration_results['evaluation'] = eval_results
                
                print(f"Iteration {iteration} Accuracy: {eval_results['accuracy']:.2f}%")
                
            except Exception as e:
                print(f"Error in evaluation for iteration {iteration}: {e}")
                iteration_results['evaluation_error'] = str(e)
        
        # Store iteration results
        all_results['iterations'][iteration] = iteration_results
        
        # Save intermediate results
        results_file = os.path.join(models_output_dir, f"{dataset_type}_star_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nIteration {iteration} completed successfully!")
        print(f"Model saved to: {model_save_path}")
        if evaluate_each_iteration and 'evaluation' in iteration_results:
            print(f"Accuracy: {iteration_results['evaluation']['accuracy']:.2f}%")
    
    # Final evaluation on the last model
    if current_model_path != BASE_MODEL_ID and not evaluate_each_iteration:
        print(f"\n{'='*60}")
        print("Final Evaluation")
        print(f"{'='*60}")
        
        try:
            final_eval_results = evaluate_model(
                model_path=current_model_path,
                dataset_type=dataset_type,
                max_samples=max_samples,
                debug=debug
            )
            all_results['final_evaluation'] = final_eval_results
            
        except Exception as e:
            print(f"Error in final evaluation: {e}")
            all_results['final_evaluation_error'] = str(e)
    
    # Save final results
    results_file = os.path.join(models_output_dir, f"{dataset_type}_star_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("STaR Process Completed!")
    print(f"Results saved to: {results_file}")
    print(f"Final model: {current_model_path}")
    print(f"{'='*60}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete STaR reasoning process.")
    parser.add_argument("--dataset", type=str, choices=['cqa', 'gsm8k', 'arithmetic'], required=True,
                       help="Dataset to use for training")
    parser.add_argument("--num_iterations", type=int, default=NUM_STAR_ITERATIONS,
                       help="Number of STaR iterations")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process per iteration (for testing)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug printing")
    parser.add_argument("--rationales_dir", type=str, default="collected_rationales",
                       help="Directory to save collected rationales")
    parser.add_argument("--models_dir", type=str, default="star_models",
                       help="Directory to save fine-tuned models")
    parser.add_argument("--no_eval", action="store_true",
                       help="Skip evaluation after each iteration")
    
    # Options for running individual phases
    parser.add_argument("--collect_only", action="store_true",
                       help="Only run rationale collection (requires --iteration and --model_path)")
    parser.add_argument("--finetune_only", action="store_true",
                       help="Only run fine-tuning (requires --iteration)")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only run evaluation (requires --model_path)")
    parser.add_argument("--iteration", type=int,
                       help="Iteration number (for --collect_only or --finetune_only)")
    parser.add_argument("--model_path", type=str,
                       help="Model path (for --collect_only or --eval_only)")
    
    args = parser.parse_args()
    
    # Validate arguments for individual phases
    if args.collect_only:
        if not args.iteration or not args.model_path:
            parser.error("--collect_only requires --iteration and --model_path")
        
        print(f"Running rationale collection only for iteration {args.iteration}")
        stats = collect_rationales_for_iteration(
            model_path=args.model_path,
            dataset_type=args.dataset,
            iteration=args.iteration,
            output_dir=args.rationales_dir,
            max_samples=args.max_samples,
            debug=args.debug
        )
        print("Collection completed.")
        
    elif args.finetune_only:
        if not args.iteration:
            parser.error("--finetune_only requires --iteration")
        
        print(f"Running fine-tuning only for iteration {args.iteration}")
        model_path = finetune_model(
            rationales_dir=args.rationales_dir,
            dataset_type=args.dataset,
            iteration=args.iteration,
            output_dir=args.models_dir,
            debug=args.debug
        )
        print(f"Fine-tuning completed. Model saved to: {model_path}")
        
    elif args.eval_only:
        if not args.model_path:
            parser.error("--eval_only requires --model_path")
        
        print(f"Running evaluation only")
        results = evaluate_model(
            model_path=args.model_path,
            dataset_type=args.dataset,
            max_samples=args.max_samples,
            debug=args.debug
        )
        print("Evaluation completed.")
        
    else:
        # Run the complete STaR process
        results = run_star_process(
            dataset_type=args.dataset,
            num_iterations=args.num_iterations,
            max_samples=args.max_samples,
            debug=args.debug,
            rationales_output_dir=args.rationales_dir,
            models_output_dir=args.models_dir,
            evaluate_each_iteration=not args.no_eval
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("STaR Process Summary")
        print(f"{'='*60}")
        
        for iteration, iter_results in results['iterations'].items():
            print(f"Iteration {iteration}:")
            if 'collection_stats' in iter_results:
                stats = iter_results['collection_stats']
                total_rationales = stats['generated_rationales_count'] + stats['rationalized_rationales_count']
                print(f"  Rationales collected: {total_rationales}")
            if 'evaluation' in iter_results:
                acc = iter_results['evaluation']['accuracy']
                print(f"  Accuracy: {acc:.2f}%")
            print()
        
        if 'final_evaluation' in results:
            print(f"Final Accuracy: {results['final_evaluation']['accuracy']:.2f}%") 