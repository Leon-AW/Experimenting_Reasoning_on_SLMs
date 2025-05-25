# reasoning_methods/hybrid/collect_rationales.py

import argparse
import os
import json
import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting import (
    generate_rationale, 
    rationalize, 
    format_question, 
    format_for_finetuning,
    TEMPERATURE, 
    TOP_P, 
    TOP_K, 
    DO_SAMPLE, 
    MAX_NEW_TOKENS,
    SEED,
)
from prepare_datasets import load_commonsense_qa, load_gsm8k, generate_arithmetic_dataset

# Configuration
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
DATA_CACHE_DIR = './data_cache'
ARITHMETIC_DATA_PATH = os.path.join(DATA_CACHE_DIR, 'arithmetic', 'arithmetic_train.jsonl')
NUM_ARITHMETIC_SAMPLE_PER_ITER = 10000

def load_few_shot_prompt(dataset_type):
    """Loads the few-shot prompt from a file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(script_dir, "prompts")
    few_shot_prompts = {
        'cqa': os.path.join(prompt_dir, 'cqa_few_shot.txt'),
        'gsm8k': os.path.join(prompt_dir, 'gsm8k_few_shot.txt'),
        'arithmetic': os.path.join(prompt_dir, 'arithmetic_few_shot.txt'),
    }
    
    filepath = few_shot_prompts.get(dataset_type)
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Few-shot prompt file not found for {dataset_type} at {filepath}. Please create it.")
    with open(filepath, 'r') as f:
        return f.read()

def get_ground_truth(example, dataset_type):
    """Extracts the ground truth answer from a dataset example."""
    import re
    
    if dataset_type == 'cqa':
        return example.get('answerKey')
    elif dataset_type == 'gsm8k':
        answer_field = example.get('answer')
        if answer_field:
            match = re.search(r"####\s*([\d\.]+)", answer_field)
            return match.group(1).strip() if match else None
        return None
    elif dataset_type == 'arithmetic':
        return example.get('answer')
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def load_model_and_tokenizer(model_id_or_path):
    """Loads the model and tokenizer."""
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

def collect_rationales_for_iteration(
    model_path: str,
    dataset_type: str,
    iteration: int,
    output_dir: str,
    max_samples: int = None,
    debug: bool = False
):
    """
    Collect rationales for a single STaR iteration.
    
    This implements Steps 3-6 from the STaR paper:
    - Step 3: Generate rationales for all problems
    - Step 4: Rationalize failures (generate explanations with hints)
    - Step 5: Filter successful self-generated rationales
    - Step 6: Filter successful rationalized explanations
    
    Args:
        model_path: Path to the model to use for generation
        dataset_type: Type of dataset ('cqa', 'gsm8k', 'arithmetic')
        iteration: Current iteration number
        output_dir: Directory to save collected rationales
        max_samples: Maximum number of samples to process (for testing)
        debug: Enable debug printing
        
    Returns:
        Dictionary with statistics about the collection process
    """
    print(f"\n=== Collecting Rationales for Iteration {iteration} ===")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_type}")
    
    # Create output directory for this iteration
    iter_output_dir = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(iter_output_dir, exist_ok=True)
    
    # Load few-shot prompt
    few_shot_prompt = load_few_shot_prompt(dataset_type)
    
    # Load dataset
    print("Loading dataset...")
    if dataset_type == 'cqa':
        dataset = load_commonsense_qa(cache_dir=DATA_CACHE_DIR)
        train_data = dataset['train']
    elif dataset_type == 'gsm8k':
        dataset = load_gsm8k(cache_dir=DATA_CACHE_DIR)
        train_data = dataset['train']
    elif dataset_type == 'arithmetic':
        if not os.path.exists(ARITHMETIC_DATA_PATH):
            print("Generating arithmetic dataset...")
            generate_arithmetic_dataset(
                num_samples=50000,
                output_dir=os.path.dirname(ARITHMETIC_DATA_PATH)
            )
        train_data = load_dataset('json', data_files=ARITHMETIC_DATA_PATH, split='train')
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Select data for this iteration
    if max_samples is not None:
        iteration_data = train_data.shuffle(seed=SEED + iteration).select(range(min(max_samples, len(train_data))))
    else:
        if dataset_type == 'arithmetic':
            num_samples = min(NUM_ARITHMETIC_SAMPLE_PER_ITER, len(train_data))
            iteration_data = train_data.shuffle(seed=SEED + iteration).select(range(num_samples))
        else:
            iteration_data = train_data
    
    print(f"Processing {len(iteration_data)} examples...")
    
    # Load model
    print(f"Loading model: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()
    
    # Initialize data collections
    generated_rationales = []  # D_n^gen from paper (Step 5)
    rationalized_rationales = []  # D_n^rat from paper (Step 6)
    
    # Initialize statistics
    stats = {
        'total_processed': 0,
        'initial_correct': 0,
        'rationalized_correct': 0,
        'final_failures': 0,
        'skipped_missing_gt': 0,
        'generated_rationales_count': 0,
        'rationalized_rationales_count': 0
    }
    
    # Progress bar format
    progress_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}]"
    
    # Process each example
    for idx, example in enumerate(tqdm(iteration_data, desc=f"Iteration {iteration} Collection", bar_format=progress_bar_format)):
        ground_truth = get_ground_truth(example, dataset_type)
        if ground_truth is None:
            if debug:
                print(f"Warning: Skipping example {idx} due to missing ground truth.")
            stats['skipped_missing_gt'] += 1
            continue
        
        stats['total_processed'] += 1
        
        # Step 3: Generate initial rationale without the answer
        r_hat, y_hat = generate_rationale(
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
        
        if r_hat is None or y_hat is None:
            if debug:
                print(f"Warning: Skipping example {idx} because initial rationale generation failed.")
            stats['final_failures'] += 1
            continue
        
        # Normalize answers for comparison
        y_hat_norm = str(y_hat).strip().lower()
        y_norm = str(ground_truth).strip().lower()
        
        # For numeric datasets, handle potential floating point differences
        if dataset_type in ['gsm8k', 'arithmetic']:
            try:
                y_hat_norm = str(int(float(y_hat_norm)))
                y_norm = str(int(float(y_norm)))
            except ValueError:
                pass
        
        is_correct = (y_hat_norm == y_norm)
        
        if debug:
            print(f"\n--- Generation Debug Example {idx} ---")
            print(f"Ground Truth: {ground_truth}")
            print(f"Generated Rationale: {r_hat}")
            print(f"Generated Answer: {y_hat}")
            print(f"Correct: {is_correct}")
            print("----------------------------------------")
        
        if is_correct:
            # Step 5: Keep successful self-generated rationale
            stats['initial_correct'] += 1
            
            formatted_instance = format_for_finetuning(
                question_data=example,
                rationale=r_hat,
                answer=ground_truth,
                dataset_type=dataset_type
            )
            
            if formatted_instance:
                generated_rationales.append({
                    'example_id': idx,
                    'question': example,
                    'rationale': r_hat,
                    'answer': ground_truth,
                    'formatted_text': formatted_instance,
                    'source': 'generated'
                })
                stats['generated_rationales_count'] += 1
            
            if debug:
                print(f"Debug: Example {idx} - Correct. Added to generated rationales.")
        
        else:
            # Step 4 & 6: Rationalize failure and filter
            if debug:
                print(f"Debug: Example {idx} - Incorrect. Initial rationale NOT SAVED:")
                print(f"  Initial rationale: {r_hat}")
                print(f"  Attempting rationalization...")
            
            # Try multiple attempts to generate a valid rationalization
            max_rationalization_attempts = 5
            successful_rationalization = False
            
            for attempt in range(max_rationalization_attempts):
                if debug:
                    print(f"    Rationalization attempt {attempt + 1}/{max_rationalization_attempts}")
                
                # Step 4: Generate rationalization with correct answer as hint
                r_star = rationalize(
                    model=model,
                    tokenizer=tokenizer,
                    question_data=example,
                    correct_answer=ground_truth,
                    dataset_type=dataset_type,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    do_sample=DO_SAMPLE,
                    few_shot_prompt=few_shot_prompt,
                    debug=debug,
                    attempt_number=attempt
                )
                
                if not r_star:
                    if debug:
                        print(f"    Attempt {attempt + 1} failed: empty rationalization")
                    continue
                
                if debug:
                    print(f"    Generated rationalization attempt {attempt + 1}: {r_star}")
                
                # Verify that the rationalization leads to the correct answer
                verification_successful = verify_rationalization(
                    model=model,
                    tokenizer=tokenizer,
                    question_data=example,
                    rationale=r_star,
                    expected_answer=ground_truth,
                    dataset_type=dataset_type,
                    few_shot_prompt=few_shot_prompt,
                    debug=debug
                )
                
                # Also check if the rationale shows quality reasoning
                if verification_successful:
                    quality_check = is_quality_rationale(r_star, dataset_type, debug=debug)
                    if not quality_check:
                        verification_successful = False
                        if debug:
                            print(f"    Attempt {attempt + 1} failed quality check - rationale is repetitive or low quality")
                
                if verification_successful:
                    # Step 6: Keep successful rationalized explanation
                    stats['rationalized_correct'] += 1
                    
                    formatted_instance = format_for_finetuning(
                        question_data=example,
                        rationale=r_star,
                        answer=ground_truth,
                        dataset_type=dataset_type
                    )
                    
                    if formatted_instance:
                        rationalized_rationales.append({
                            'example_id': idx,
                            'question': example,
                            'rationale': r_star,
                            'answer': ground_truth,
                            'formatted_text': formatted_instance,
                            'source': 'rationalized',
                            'attempt_number': attempt + 1
                        })
                        stats['rationalized_rationales_count'] += 1
                    
                    successful_rationalization = True
                    if debug:
                        print(f"    Attempt {attempt + 1} successful!")
                    break
                else:
                    if debug:
                        print(f"    Attempt {attempt + 1} failed verification")
            
            if not successful_rationalization:
                stats['final_failures'] += 1
                if debug:
                    print(f"    All rationalization attempts failed for example {idx}")
    
    # Clean up model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save collected rationales - COMBINE all rationales into one file
    all_rationales_file = os.path.join(iter_output_dir, f"{dataset_type}_rationales.jsonl")
    stats_file = os.path.join(iter_output_dir, f"{dataset_type}_collection_stats.json")
    
    # Combine all rationales (generated + rationalized) into one file
    all_rationales = generated_rationales + rationalized_rationales
    
    with open(all_rationales_file, 'w') as f:
        for item in all_rationales:
            f.write(json.dumps(item) + '\n')
    
    # Save statistics
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print statistics
    print(f"\n--- Iteration {iteration} Collection Statistics ---")
    print(f"Total examples processed: {stats['total_processed']}")
    if stats['total_processed'] > 0:
        print(f"  Initially correct: {stats['initial_correct']} ({stats['initial_correct']/stats['total_processed']*100:.2f}%)")
        print(f"  Correct after rationalization: {stats['rationalized_correct']} ({stats['rationalized_correct']/stats['total_processed']*100:.2f}%)")
        print(f"  Final failures: {stats['final_failures']} ({stats['final_failures']/stats['total_processed']*100:.2f}%)")
    print(f"  Generated rationales collected: {stats['generated_rationales_count']}")
    print(f"  Rationalized rationales collected: {stats['rationalized_rationales_count']}")
    print(f"  Total rationales for fine-tuning: {stats['generated_rationales_count'] + stats['rationalized_rationales_count']}")
    if stats['skipped_missing_gt'] > 0:
        print(f"Skipped due to missing ground truth: {stats['skipped_missing_gt']}")
    print("---------------------------------------------------")
    
    print(f"Rationales saved to:")
    print(f"  All rationales: {all_rationales_file}")
    print(f"  Statistics: {stats_file}")
    
    return stats

def verify_rationalization(model, tokenizer, question_data, rationale, expected_answer, dataset_type, few_shot_prompt, debug=False):
    """
    Verify that a rationalization actually leads to the expected answer.
    This is crucial for Step 6 filtering in the STaR paper.
    """
    try:
        if dataset_type == 'cqa':
            from prompting import extract_cqa_explicit_answer, score_cqa_answers
            
            # Try to get explicit answer from rationale first
            explicit_answer = extract_cqa_explicit_answer(rationale, question_data['choices']['label'])
            
            if explicit_answer:
                rationalized_answer = explicit_answer
                if debug:
                    print(f"      Found explicit answer in rationale: {explicit_answer}")
            else:
                # Fallback to log-likelihood scoring
                rationalized_answer = score_cqa_answers(model, tokenizer, question_data, rationale)
                if debug:
                    print(f"      Used log-likelihood scoring, got: {rationalized_answer}")
        
        else:
            # For numeric datasets, first try to extract answer directly from the rationalization
            from prompting import extract_numeric_answer
            rationalized_answer = extract_numeric_answer(rationale)
            
            if debug:
                print(f"      Trying to extract answer directly from rationale...")
                print(f"      Rationale: {rationale}")
                print(f"      Direct extraction result: {rationalized_answer}")
            
            # If we couldn't extract from rationale, generate continuation
            if not rationalized_answer:
                if debug:
                    print(f"      Direct extraction failed, trying verification generation...")
                
                # Create a more structured verification prompt
                if dataset_type == 'gsm8k':
                    verification_prompt = f"{format_question(question_data, dataset_type)}\n{rationale}\n\nTherefore, the final answer is:"
                else:  # arithmetic
                    verification_prompt = f"{format_question(question_data, dataset_type)}\n{rationale}\n\nFinal answer:"
                
                verification_inputs = tokenizer(verification_prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    verification_outputs = model.generate(
                        **verification_inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                verification_text = tokenizer.decode(
                    verification_outputs[0][verification_inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                if debug:
                    print(f"      Verification text generated: '{verification_text}'")
                
                # Extract answer from verification text
                rationalized_answer = extract_numeric_answer(verification_text)
                
                if debug:
                    print(f"      Extracted from verification: {rationalized_answer}")
                
                # If still no answer, try parsing with dataset-specific functions
                if not rationalized_answer:
                    if dataset_type == 'gsm8k':
                        from prompting import parse_gsm8k_output
                        _, rationalized_answer = parse_gsm8k_output(f"{verification_prompt}\n{verification_text}")
                    elif dataset_type == 'arithmetic':
                        from prompting import parse_arithmetic_output
                        _, rationalized_answer = parse_arithmetic_output(f"{verification_prompt}\n{verification_text}")
                    
                    if debug:
                        print(f"      Parsed with dataset function: {rationalized_answer}")
        
        # Normalize and compare answers
        if rationalized_answer:
            rationalized_answer_norm = str(rationalized_answer).strip().lower()
            expected_answer_norm = str(expected_answer).strip().lower()
            
            # For numeric datasets, handle floating point differences
            if dataset_type in ['gsm8k', 'arithmetic']:
                try:
                    rationalized_answer_norm = str(int(float(rationalized_answer_norm)))
                    expected_answer_norm = str(int(float(expected_answer_norm)))
                except ValueError:
                    pass
            
            is_correct = (rationalized_answer_norm == expected_answer_norm)
            
            if debug:
                print(f"      Verification: Expected={expected_answer_norm}, Got={rationalized_answer_norm}, Correct={is_correct}")
            
            return is_correct
        else:
            if debug:
                print(f"      Verification failed: Could not extract answer from verification")
            return False
    
    except Exception as e:
        if debug:
            print(f"      Verification error: {e}")
        return False

def is_quality_rationale(rationale, dataset_type, debug=False):
    """
    Check if a rationale shows proper reasoning and isn't just repetitive.
    
    Args:
        rationale: The generated rationale text
        dataset_type: Type of dataset ('cqa', 'gsm8k', 'arithmetic')
        debug: Enable debug printing
        
    Returns:
        True if the rationale shows quality reasoning, False otherwise
    """
    if not rationale or len(rationale.strip()) < 20:
        if debug:
            print(f"      Quality check: Too short")
        return False
    
    # Check for excessive repetition
    lines = rationale.strip().split('\n')
    if len(lines) > 3:
        # Count how many lines are nearly identical
        line_counts = {}
        for line in lines:
            clean_line = line.strip().lower()
            if len(clean_line) > 10:  # Only check substantial lines
                line_counts[clean_line] = line_counts.get(clean_line, 0) + 1
        
        # If any line appears more than 3 times, it's likely repetitive
        max_repetitions = max(line_counts.values()) if line_counts else 0
        if max_repetitions > 3:
            if debug:
                print(f"      Quality check: Too repetitive (max repetitions: {max_repetitions})")
            return False
    
    # Check for proper reasoning structure
    if dataset_type == 'arithmetic':
        # Should have step-by-step breakdown or proper calculation
        has_steps = any(keyword in rationale.lower() for keyword in ['step 1', 'step 2', 'break down', 'add', 'subtract', 'multiply', 'divide'])
        has_calculation = '=' in rationale and ('+' in rationale or '-' in rationale or '×' in rationale or '÷' in rationale or '*' in rationale or '/' in rationale)
        
        if not (has_steps or has_calculation):
            if debug:
                print(f"      Quality check: No clear mathematical reasoning")
            return False
            
        # Check if it's just stating the answer without work
        # Count how many actual calculation steps are shown
        calculation_lines = [line for line in lines if '=' in line and any(op in line for op in ['+', '-', '×', '÷', '*', '/'])]
        if len(calculation_lines) < 2 and 'step' not in rationale.lower():
            if debug:
                print(f"      Quality check: Insufficient calculation steps")
            return False
    
    elif dataset_type == 'gsm8k':
        # Should have step-by-step reasoning
        has_steps = any(keyword in rationale.lower() for keyword in ['step 1', 'step 2', 'first', 'then', 'next', 'finally'])
        has_reasoning = len([line for line in lines if len(line.strip()) > 15]) >= 3  # At least 3 substantial lines
        
        if not (has_steps and has_reasoning):
            if debug:
                print(f"      Quality check: Insufficient reasoning structure")
            return False
    
    elif dataset_type == 'cqa':
        # Should have logical reasoning, not just answer assertion
        has_reasoning = len([line for line in lines if len(line.strip()) > 10]) >= 2  # At least 2 substantial lines
        
        if not has_reasoning:
            if debug:
                print(f"      Quality check: Insufficient reasoning")
            return False
    
    if debug:
        print(f"      Quality check: PASSED")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect rationales for STaR training.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to use for generation")
    parser.add_argument("--dataset", type=str, choices=['cqa', 'gsm8k', 'arithmetic'], required=True, help="Dataset type")
    parser.add_argument("--iteration", type=int, required=True, help="Current iteration number")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save collected rationales")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    
    args = parser.parse_args()
    
    stats = collect_rationales_for_iteration(
        model_path=args.model_path,
        dataset_type=args.dataset,
        iteration=args.iteration,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        debug=args.debug
    )
    
    print(f"\nCollection completed for iteration {args.iteration}") 