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
MAX_NEW_TOKENS = 256
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
ADAPTER_SAVE_DIR_BASE = "./star_adapters"

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
OUTPUT_DIR_BASE = "./star_output" # For trainer logs/checkpoints within an iter

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
        print("Setting pad token to eos token")
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

        successful_rationales_data = []
        failed_indices = [] # Store indices of problems the model failed

        # Select data for this iteration (especially for large datasets like arithmetic)
        if DATASET_TYPE == 'arithmetic':
            iteration_data = train_data.shuffle(seed=42+i).select(range(min(NUM_ARITHMETIC_SAMPLE_PER_ITER, len(train_data))))
        else:
            iteration_data = train_data # Use full dataset for CQA/GSM8K

        print(f"Generating rationales for {len(iteration_data)} examples...")
        for idx, example in enumerate(tqdm(iteration_data, desc=f"Iteration {iteration} Generation", disable=args.debug)):
            ground_truth = get_ground_truth(example, DATASET_TYPE)
            if ground_truth is None:
                print(f"Warning: Skipping example {idx} due to missing ground truth.")
                continue

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

            # Compare case-insensitively for CQA
            if DATASET_TYPE == 'cqa':
                is_correct = r_hat is not None and y_hat is not None and str(y_hat).lower() == str(ground_truth).lower()
            else:
                is_correct = r_hat is not None and y_hat is not None and str(y_hat) == str(ground_truth)

            if args.debug:
                print(f"\n--- Gen Debug Example {idx} ---")
                print(f"Ground Truth: {ground_truth}")
                print(f"Generated Rationale: {r_hat}")
                print(f"Generated Answer: {y_hat}")
                print(f"Correct: {is_correct}")
                print("-------------------------")

            if is_correct:
                try:
                    formatted_text = format_for_finetuning(example, r_hat, y_hat, DATASET_TYPE)
                    successful_rationales_data.append({"text": formatted_text})
                except Exception as e:
                    print(f"Error formatting successful rationale {idx}: {e}")
            else:
                failed_indices.append(idx) # Store original index if using subset, or direct index otherwise

        print(f"Generated {len(successful_rationales_data)} successful rationales.")

        # --- Rationalization Phase ---
        print(f"Attempting rationalization for {len(failed_indices)} failed examples...")
        successful_rationalizations_data = []

        # Need to get the actual failed examples based on indices
        failed_examples = [iteration_data[i] for i in failed_indices]

        for example_idx, example in enumerate(tqdm(failed_examples, desc=f"Iteration {iteration} Rationalization", disable=args.debug)):
            ground_truth = get_ground_truth(example, DATASET_TYPE)
            if ground_truth is None: continue # Should not happen if check passed before

            r_rat, y_rat = rationalize(
                model=gen_model,
                tokenizer=gen_tokenizer,
                few_shot_prompt=few_shot_prompt,
                question_data=example,
                correct_answer=ground_truth,
                dataset_type=DATASET_TYPE,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                do_sample=DO_SAMPLE
            )

            # Compare case-insensitively for CQA
            if DATASET_TYPE == 'cqa':
                rationalization_produces_correct_answer = r_rat is not None and y_rat is not None and str(y_rat).lower() == str(ground_truth).lower()
            else:
                rationalization_produces_correct_answer = r_rat is not None and y_rat is not None and str(y_rat) == str(ground_truth)

            if args.debug:
                 print(f"\n--- Rat Debug Example {example_idx} (Original Index: {failed_indices[example_idx]}) ---")
                 print(f"Ground Truth (Hint): {ground_truth}")
                 print(f"Rationalized Rationale: {r_rat}")
                 print(f"Rationalized Answer: {y_rat}")
                 print(f"Produced Correct Answer: {rationalization_produces_correct_answer}")
                 print("-------------------------")

            # Check if rationalization produced the correct answer (Algorithm line 6)
            if rationalization_produces_correct_answer:
                 try:
                    formatted_text = format_for_finetuning(example, r_rat, ground_truth, DATASET_TYPE) # Use r_rat!
                    successful_rationalizations_data.append({"text": formatted_text})
                 except Exception as e:
                    print(f"Error formatting successful rationalization {example_idx}: {e}")


        print(f"Generated {len(successful_rationalizations_data)} successful rationalizations.")

        # Clean up generation model from memory
        del gen_model, gen_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # --- Prepare Fine-tuning Data ---
        fine_tuning_data = successful_rationales_data + successful_rationalizations_data
        if not fine_tuning_data:
            print("No successful rationales or rationalizations generated in this iteration. Stopping STaR.")
            break

        print(f"Total examples for fine-tuning in iteration {iteration}: {len(fine_tuning_data)}")
        # Check a few examples
        print("Sample fine-tuning data point:")
        print(random.choice(fine_tuning_data)['text'])

        finetune_dataset = Dataset.from_list(fine_tuning_data)

        # --- Fine-tuning Phase ---
        print("Starting fine-tuning...")
        # Load BASE model for fine-tuning (STaR Algorithm Line 7)
        print(f"Loading BASE model for fine-tuning: {BASE_MODEL_ID}")
        train_model, train_tokenizer = load_model_and_tokenizer(BASE_MODEL_ID)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=LOGGING_STEPS,
            num_train_epochs=NUM_EPOCHS_PER_ITER,
            save_strategy="no", # Don't save checkpoints during training, save full model after
            report_to="none",
            bf16=True, # Use bf16 if available for performance
            gradient_checkpointing=True, # Enable gradient checkpointing to save memory
            gradient_checkpointing_kwargs={'use_reentrant': False}, # Recommended setting
        )

        # Use SFTTrainer for simplicity with text formatting
        trainer = SFTTrainer(
            model=train_model,
            train_dataset=finetune_dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            tokenizer=train_tokenizer,
            args=training_args,
        )

        print(f"Training model...")
        trainer.train()

        # Save the fully fine-tuned model
        print(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir) # Saves the full model
        train_tokenizer.save_pretrained(output_dir) # Save tokenizer with the model

        # Update path for the *next* iteration's generation model
        current_model_path_for_generation = output_dir # This now correctly points to the model trained in *this* iteration

        # Clean up training model from memory
        del train_model, train_tokenizer, trainer, finetune_dataset
        gc.collect()
        torch.cuda.empty_cache()
        # num_train_steps = int(INITIAL_TRAIN_STEPS * (STEP_INCREASE_FACTOR**iteration)) # Update steps for next iter based on fixed increase


    print("\n--- STaR Process Finished ---")
    print(f"Final model saved at: {current_model_path_for_generation}")

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
        for example in tqdm(eval_data, desc="Final Evaluation", disable=args.debug):
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