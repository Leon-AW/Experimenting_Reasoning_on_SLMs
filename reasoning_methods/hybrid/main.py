# reasoning_methods/hybrid/main.py

import argparse # Add this import at the top
import os
import re # Added import
import json
import random
import torch
import gc
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer, # Using standard Trainer might be needed if SFTT causes issues without PEFT
)
# SFTTrainer can still work for full fine-tuning if no peft_config is passed
from trl import SFTTrainer
from tqdm import tqdm
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define generation parameters (copied from prompting/config.py)
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.5
TOP_P = 0.9
TOP_K = 0
DO_SAMPLE = True
SEED = 42

# Assuming prompting.py and prepare_datasets.py are in the same directory
from prompting import generate_rationale, rationalize, format_question
from prepare_datasets import load_commonsense_qa, load_gsm8k, generate_arithmetic_dataset

# --- Configuration ---
# Model Configuration
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B" # Or your 1B Llama 3.2 path
# BASE_MODEL_ID = "path/to/your/llama3.2-1b" # Example for local model
ADAPTER_SAVE_DIR_BASE = "reasoning_methods/hybrid/star_adapters"

# Dataset Configuration
DATASET_TYPE = 'cqa' # Choose 'cqa', 'gsm8k', or 'arithmetic'
DATA_CACHE_DIR = './data_cache'
ARITHMETIC_DATA_PATH = os.path.join(DATA_CACHE_DIR, 'arithmetic', 'arithmetic_train.jsonl')
NUM_ARITHMETIC_SAMPLE_PER_ITER = 10000 # As per paper for arithmetic

# Few-Shot Prompt Configuration (Replace with actual paths)
# Determine the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(script_dir, "prompts") # Look for prompts within the script's directory
FEW_SHOT_PROMPTS = {
    'cqa': os.path.join(PROMPT_DIR, 'cqa_few_shot.txt'),
    'gsm8k': os.path.join(PROMPT_DIR, 'gsm8k_few_shot.txt'),
    'arithmetic': os.path.join(PROMPT_DIR, 'arithmetic_few_shot.txt'),
}

# STaR Loop Configuration
NUM_STAR_ITERATIONS = 5 # Number of STaR iterations
OUTPUT_DIR_BASE = "reasoning_methods/hybrid/star_output" # For trainer logs/checkpoints within an iter

# Fine-Tuning Configuration (using SFTTrainer)
INITIAL_TRAIN_STEPS = 40 # Start with 40 steps as per paper
STEP_INCREASE_FACTOR = 1.2 # Increase steps by 20% each iteration
PER_DEVICE_TRAIN_BATCH_SIZE = 2 # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS = 4 # Adjust based on GPU memory
LEARNING_RATE = 2e-5 # Common learning rate for fine-tuning
MAX_SEQ_LENGTH = 256 # Adjust based on model and data
LOGGING_STEPS = 10
NUM_EPOCHS_PER_ITER = 1 # Train for 1 epoch over generated data per STaR iter

# --- Helper Functions ---

def load_few_shot_prompt(dataset_type):
    """Loads the few-shot prompt from a file."""
    filepath = FEW_SHOT_PROMPTS.get(dataset_type)
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Few-shot prompt file not found for {dataset_type} at {filepath}. Please create it.")
    with open(filepath, 'r') as f:
        return f.read()

def get_ground_truth(example, dataset_type):
    """Extracts the ground truth answer from a dataset example."""
    if dataset_type == 'cqa':
        # Ensure 'answerKey' exists and handle potential variations
        return example.get('answerKey')
    elif dataset_type == 'gsm8k':
        # GSM8K answer extraction needs the 'answer' field
        answer_field = example.get('answer')
        if answer_field:
            match = re.search(r"####\s*([\d\.]+)", answer_field) # Allow decimals
            return match.group(1).strip() if match else None
        return None
    elif dataset_type == 'arithmetic':
        return example.get('answer') # Already extracted in prepare_datasets
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def format_for_finetuning(question_data, rationale, answer, dataset_type):
    """Formats the rationale into a single string for SFTTrainer."""
    # Ensure rationale and answer are strings
    rationale_str = str(rationale) if rationale is not None else ""
    answer_str = str(answer) if answer is not None else ""

    # Gets the Q: part, handling potential missing keys gracefully
    formatted_question = format_question(question_data, dataset_type)

    if dataset_type == 'cqa':
        # Find the answer text corresponding to the answer letter
        answer_text = ""
        for label, text in zip(question_data['choices']['label'], question_data['choices']['text']):
            if label == answer:
                answer_text = text
                break
        # Example Format: "Q: ...\nAnswer Choices:...\nA: [Rationale] Therefore, the answer is [Answer Text] ([Letter])."
        # Ensure the rationale doesn't already end with the answer phrase
        final_answer_phrase = f"Therefore, the answer is {answer_text} ({answer})."
        if rationale.strip().endswith(final_answer_phrase):
             # Avoid duplicating the final answer phrase if the model generated it
             return f"{formatted_question.strip()}\n{rationale.strip()}"
        else:
             return f"{formatted_question.strip()}\n{rationale.strip()} {final_answer_phrase}"

    elif dataset_type == 'arithmetic':
        # Example Format: "Input:\nNUM1 + NUM2\nTarget:\n<scratch>...</scratch>\nANSWER"
        # Rationale here is the scratchpad content
        return f"{formatted_question.strip()}\n<scratch>{rationale.strip()}</scratch>\n{answer}"

    elif dataset_type == 'gsm8k':
        # Example Format: "Q: ...\nA: [Rationale] The final answer is: [Answer]"
        final_answer_phrase = f"The final answer is: {answer}"
         # Avoid duplicating the final answer phrase if the model generated it
        if rationale.strip().endswith(final_answer_phrase):
             return f"{formatted_question.strip()}\n{rationale.strip()}"
        else:
             return f"{formatted_question.strip()}\n{rationale.strip()} {final_answer_phrase}"

    else:
        raise ValueError(f"Unknown dataset_type for fine-tuning format: {dataset_type}")


def load_model_and_tokenizer(model_id_or_path):
    """Loads the model and tokenizer for full fine-tuning."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if supported
        device_map="auto", # Automatically distribute across GPUs
        trust_remote_code=True # If necessary for your model
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)

    # Handle padding token for Llama models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

# --- Main STaR Loop ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the STaR reasoning process.")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing for generation and rationalization steps.")
    args = parser.parse_args()

    print(f"Starting STaR process for dataset: {DATASET_TYPE}")
    print(f"Base model: {BASE_MODEL_ID}")
    if args.debug:
        print("DEBUG mode enabled.")

    # --- Initial Setup ---
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    os.makedirs(PROMPT_DIR, exist_ok=True) # Ensure prompt dir exists
    os.makedirs(ADAPTER_SAVE_DIR_BASE, exist_ok=True)

    # Load few-shot prompt
    few_shot_prompt = load_few_shot_prompt(DATASET_TYPE)

    # Load dataset
    print("Loading dataset...")
    if DATASET_TYPE == 'cqa':
        dataset = load_commonsense_qa(cache_dir=DATA_CACHE_DIR)
        train_data = dataset['train']
        eval_data = dataset['validation'] # Use validation for evaluation during loop
    elif DATASET_TYPE == 'gsm8k':
        dataset = load_gsm8k(cache_dir=DATA_CACHE_DIR)
        train_data = dataset['train']
        eval_data = dataset['test']
    elif DATASET_TYPE == 'arithmetic':
        if not os.path.exists(ARITHMETIC_DATA_PATH):
            print("Generating arithmetic dataset...")
            generate_arithmetic_dataset(
                num_samples=50000, # Generate the full pool first
                output_dir=os.path.dirname(ARITHMETIC_DATA_PATH)
            )
        # Load the generated data - treat the whole file as the potential training pool
        train_data = load_dataset('json', data_files=ARITHMETIC_DATA_PATH, split='train')
        # Arithmetic doesn't have a standard eval set, maybe create one or skip eval during loop?
        eval_data = None # Or create a small hold-out set from generated data
    else:
        raise ValueError("Invalid DATASET_TYPE configured.")
    print(f"Loaded {len(train_data)} training examples.")

    # --- STaR Iterations ---
    current_model_path = BASE_MODEL_ID
    is_adapter = False # Flag to track if current_model_path points to an adapter
    num_train_steps = INITIAL_TRAIN_STEPS

    # Initialize path for generation model for iteration 1
    current_model_path_for_generation = BASE_MODEL_ID

    # Define a progress bar format that displays elapsed time and estimated remaining time
    progress_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}]"

    for i in range(NUM_STAR_ITERATIONS):
        iteration = i + 1
        print(f"\n--- STaR Iteration {iteration}/{NUM_STAR_ITERATIONS} ---")
        output_dir = os.path.join(OUTPUT_DIR_BASE, f"iteration_{iteration}")
        adapter_save_dir = os.path.join(ADAPTER_SAVE_DIR_BASE, f"iteration_{iteration}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(adapter_save_dir, exist_ok=True)

        # Determine the model path for generation in this iteration
        if iteration == 1:
            print(f"Loading BASE model for generation: {BASE_MODEL_ID}")
        else:
            print(f"Loading model from previous iteration for generation: {current_model_path_for_generation}")

        # --- Generation Phase ---
        gen_model, gen_tokenizer = load_model_and_tokenizer(current_model_path_for_generation)
        gen_model.eval()

        # successful_rationales_data = [] # Rename this list
        finetuning_data = [] # List to store data formatted for fine-tuning
        failed_indices = [] # Store indices of problems the model failed initially

        # Select data for this iteration (especially for large datasets like arithmetic)
        if DATASET_TYPE == 'arithmetic':
            iteration_data = train_data.shuffle(seed=42+i).select(range(min(NUM_ARITHMETIC_SAMPLE_PER_ITER, len(train_data))))
        else:
            iteration_data = train_data # Use full dataset for CQA/GSM8K

        print(f"Generating rationales for {len(iteration_data)} examples...")
        for idx, example in enumerate(tqdm(iteration_data, desc=f"Iteration {iteration} Generation", bar_format=progress_bar_format)):
            ground_truth = get_ground_truth(example, DATASET_TYPE)
            if ground_truth is None:
                print(f"Warning: Skipping example {idx} due to missing ground truth.")
                continue

            # Step 1: Generate initial rationale without the answer
            r_hat, y_hat = generate_rationale(
                model=gen_model,
                tokenizer=gen_tokenizer,
                few_shot_prompt=few_shot_prompt,
                question_data=example,
                dataset_type=DATASET_TYPE,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                do_sample=DO_SAMPLE
            )

            if r_hat is None or y_hat is None:
                print(f"Warning: Skipping example {idx} because initial rationale generation failed.")
                failed_indices.append(idx)
                continue

            # Normalize answers for comparison
            y_hat_norm = str(y_hat).strip().lower()
            y_norm = str(ground_truth).strip().lower()

            # For numeric datasets, handle potential floating point differences if needed
            # Basic check for now, can be made more robust (e.g., float comparison)
            if DATASET_TYPE in ['gsm8k', 'arithmetic']:
                 try:
                     # Simple normalization: treat "123.0" and "123" as same
                     y_hat_norm = str(int(float(y_hat_norm)))
                     y_norm = str(int(float(y_norm)))
                 except ValueError:
                     # If conversion fails, compare as strings
                     pass

            # Step 2 & 3: Compare predicted answer with ground truth
            is_correct = (y_hat_norm == y_norm)

            if args.debug and idx < 25: # Print debug info for first few examples if debug enabled
                 print(f"\n--- Gen Debug Example {idx} ---")
                 print(f"Ground Truth: {ground_truth}")
                 print(f"Generated Rationale: {r_hat}")
                 print(f"Generated Answer: {y_hat}")
                 print(f"Correct: {is_correct}")
                 print("-------------------------")


            if is_correct:
                # Step 4 (Correct): Use the generated rationale
                rationale_for_finetuning = r_hat
                if args.debug and idx < 25:
                     print(f"Debug: Example {idx} - Correct. Using generated rationale.")
            else:
                # Step 5 (Incorrect): Generate rationalization using the correct answer
                failed_indices.append(idx)
                if args.debug and idx < 25:
                     print(f"Debug: Example {idx} - Incorrect (Predicted: {y_hat}, GT: {ground_truth}). Generating rationalization...")

                # Call rationalize function
                r_star = rationalize(
                    model=gen_model,
                    tokenizer=gen_tokenizer,
                    question_data=example,
                    correct_answer=ground_truth, # Provide the ground truth answer!
                    dataset_type=DATASET_TYPE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    do_sample=DO_SAMPLE
                )

                if r_star:
                    rationale_for_finetuning = r_star
                    if args.debug and idx < 25:
                         print(f"    Rationalization Details:")
                         print(f"      Question: {example.get('question', 'N/A')[:80]}...")
                         print(f"      Predicted Answer (Incorrect): {y_hat}")
                         print(f"      Gold Answer: {ground_truth}")
                         print(f"    Debug: Example {idx} - Rationalization generated: {r_star}...")
                else:
                    print(f"Warning: Skipping example {idx} because rationalization failed.")
                    if args.debug and idx < 25:
                         print(f"    Debug: Example {idx} - Rationalization FAILED.")
                    continue # Skip this example if rationalization fails

            # Format the data for fine-tuning using the chosen rationale and ALWAYS the ground truth answer
            formatted_instance = format_for_finetuning(
                question_data=example,
                rationale=rationale_for_finetuning,
                answer=ground_truth, # Use ground_truth here
                dataset_type=DATASET_TYPE
            )

            if formatted_instance:
                 # successful_rationales_data.append({"text": formatted_instance}) # Old way
                 finetuning_data.append({"text": formatted_instance})
            else:
                 print(f"Warning: Skipping example {idx} due to formatting error.")


        # --- Clean up Generation Model ---
        del gen_model, gen_tokenizer
        gc.collect()
        torch.cuda.empty_cache()


        # --- Fine-tuning Phase ---
        if not finetuning_data:
            print("No successful rationales generated in this iteration. Skipping fine-tuning.")
            # Decide how to proceed: continue to next iter? Stop? Use previous model?
            # For now, we'll just continue, using the same model for the next gen phase
            print(f"Using model from previous step for next generation: {current_model_path_for_generation}")
            continue # Skip finetuning if no data

        print(f"Fine-tuning on {len(finetuning_data)} generated examples...")

        # Prepare dataset for SFTTrainer
        # finetuning_dataset = Dataset.from_dict({"text": [item['text'] for item in successful_rationales_data]}) # Old
        finetuning_dataset = Dataset.from_dict({"text": [item['text'] for item in finetuning_data]})


        # Load model for fine-tuning: Always start from the original BASE_MODEL_ID as per STaR paper.
        print(f"Loading model for fine-tuning: {BASE_MODEL_ID}")
        ft_model, ft_tokenizer = load_model_and_tokenizer(BASE_MODEL_ID)

        # Calculate training steps based on dataset size and epoch config
        # total_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * torch.cuda.device_count()
        # steps_per_epoch = len(finetuning_dataset) // total_train_batch_size
        # max_steps_for_iter = int(NUM_EPOCHS_PER_ITER * steps_per_epoch)
        # Use max_steps based on dataset size or fixed steps, whichever is smaller? Paper uses fixed steps.
        max_steps_for_iter = num_train_steps
        print(f"Calculated max_steps for this iteration: {max_steps_for_iter}")


        training_args = TrainingArguments(
            output_dir=output_dir,
            # num_train_epochs=NUM_EPOCHS_PER_ITER, # Train for a fixed number of epochs on the generated data
            max_steps=max_steps_for_iter, # Train for a fixed number of steps as per STaR paper
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            logging_dir=f"{output_dir}/logs",
            logging_steps=LOGGING_STEPS,
            save_strategy="steps", # Save checkpoints periodically if needed
            save_steps=max_steps_for_iter // 2 if max_steps_for_iter > 10 else max_steps_for_iter, # Save mid-epoch? Or just at end?
            save_total_limit=1, # Keep only the last checkpoint
            report_to="none", # Disable wandb/tensorboard unless configured
            fp16=False, # Use BF16 if available via torch_dtype in model loading
            bf16=torch.cuda.is_bf16_supported(),
        )

        trainer = SFTTrainer(
            model=ft_model,
            tokenizer=ft_tokenizer,
            train_dataset=finetuning_dataset,
            dataset_text_field="text", # Specify the column containing the formatted text
            max_seq_length=MAX_SEQ_LENGTH,
            args=training_args,
            # No PEFT config needed for full fine-tuning
        )

        print("Starting fine-tuning...")
        trainer.train()
        print("Fine-tuning finished.")

        # Save the fine-tuned model (full model, not adapter)
        # trainer.save_model(adapter_save_dir) # This saves the full model when no PEFT is used
        ft_model.save_pretrained(adapter_save_dir) # Use standard save_pretrained
        ft_tokenizer.save_pretrained(adapter_save_dir)
        print(f"Fine-tuned model saved to {adapter_save_dir}")

        # --- Update for next iteration ---
        # current_model_path = adapter_save_dir # This variable is no longer needed as FT always starts from BASE_MODEL_ID
        current_model_path_for_generation = adapter_save_dir # Use the newly trained model for the next generation phase
        num_train_steps = int(num_train_steps * STEP_INCREASE_FACTOR) # Increase steps for next iter

        # --- Clean up Fine-tuning Model ---
        del ft_model, ft_tokenizer, trainer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n--- STaR Process Completed ---")

    # --- Final Evaluation ---
    # Load the final model (current_model_path_for_generation)
    # Run evaluation on the test/validation set (eval_data)
    # Implement evaluation logic here...
    print("\n--- Final Evaluation ---")
    if eval_data:
        print(f"Loading final model for evaluation: {current_model_path_for_generation}")
        eval_model, eval_tokenizer = load_model_and_tokenizer(current_model_path_for_generation)
        eval_model.eval()
        
        correct_count = 0
        total_count = 0
        # Add evaluation loop here using generate_rationale (without filtering/rationalization)
        # Similar to the generation loop inside STaR, but on eval_data
        print(f"Evaluating on {len(eval_data)} examples...")
        for example in tqdm(eval_data, desc="Final Evaluation", bar_format=progress_bar_format):
            ground_truth = get_ground_truth(example, DATASET_TYPE)
            if ground_truth is None: continue

            # Use generate_rationale from prompting.py, NOT the full STaR loop
            r_eval, y_hat = generate_rationale(
                model=eval_model,
                tokenizer=eval_tokenizer,
                few_shot_prompt=few_shot_prompt,
                question_data=example,
                dataset_type=DATASET_TYPE,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                do_sample=DO_SAMPLE
            )
            
            # Compare case-insensitively for CQA
            if DATASET_TYPE == 'cqa':
                is_correct_eval = y_hat is not None and str(y_hat).lower() == str(ground_truth).lower()
            else:
                is_correct_eval = y_hat is not None and str(y_hat) == str(ground_truth)

            if args.debug:
                print(f"\n--- Eval Debug Example ---")
                # Optionally print question identifier if available
                # print(f"Question ID: {example.get('id')}") # Example if ID exists
                print(f"Ground Truth: {ground_truth}")
                print(f"Eval Rationale: {r_eval}")
                print(f"Eval Answer: {y_hat}")
                print(f"Correct: {is_correct_eval}")
                print("-------------------------")

            if is_correct_eval:
                correct_count += 1
            total_count += 1

        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        print(f"Final Accuracy on {DATASET_TYPE} ({'validation' if DATASET_TYPE=='cqa' else 'test'}): {accuracy:.2f}% ({correct_count}/{total_count})")
        
        del eval_model, eval_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("No evaluation data available for this dataset type.")