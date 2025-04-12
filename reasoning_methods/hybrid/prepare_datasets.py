import datasets
import random
import json
import os
from tqdm import tqdm

def load_commonsense_qa(cache_dir='./data_cache'):
    """Loads the CommonsenseQA dataset."""
    print("Loading CommonsenseQA...")
    try:
        ds = datasets.load_dataset("commonsense_qa", cache_dir=cache_dir)
        print("CommonsenseQA loaded successfully.")
        print(f"Train split: {len(ds['train'])} examples")
        print(f"Validation split: {len(ds['validation'])} examples")
        print(f"Test split: {len(ds['test'])} examples")
        return ds
    except Exception as e:
        print(f"Error loading CommonsenseQA: {e}")
        return None

def load_gsm8k(cache_dir='./data_cache'):
    """Loads the GSM8K dataset."""
    print("Loading GSM8K...")
    try:
        # GSM8K has different configurations, 'main' is common
        ds = datasets.load_dataset("gsm8k", "main", cache_dir=cache_dir)
        print("GSM8K loaded successfully.")
        print(f"Train split: {len(ds['train'])} examples")
        print(f"Test split: {len(ds['test'])} examples")
        return ds
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return None

def generate_arithmetic_scratchpad(num1_str, num2_str):
    """Generates the scratchpad rationale for adding two numbers."""
    n_digits = len(num1_str) # Assume equal length, padded if necessary
    scratchpad_lines = []
    carry = 0
    result_so_far = ""

    for i in range(n_digits - 1, -1, -1):
        d1 = int(num1_str[i])
        d2 = int(num2_str[i])
        current_sum = d1 + d2 + carry
        digit_sum = current_sum % 10
        new_carry = current_sum // 10

        # Format scratchpad line
        line = f"{d1} + {d2}"
        if carry > 0:
            line += f" + {carry}"
        line += f" = {current_sum}"

        result_so_far = str(digit_sum) + result_so_far

        scratchpad_lines.append(f"{line} , {result_so_far} C: {new_carry}")
        carry = new_carry

    # Handle final carry if any
    if carry > 0:
        result_so_far = str(carry) + result_so_far
        scratchpad_lines.append(f"carry = {carry}, {result_so_far} C: 0") # No more carry

    # Reverse lines for correct order and add final formatting
    final_scratchpad = "<scratch>"
    final_scratchpad += f"{num1_str} + {num2_str} , C: 0" # Initial state
    final_scratchpad += "".join(reversed(scratchpad_lines))
    final_scratchpad += f"</scratch>"
    return final_scratchpad, result_so_far

def generate_arithmetic_problem(num_digits):
    """Generates a single arithmetic problem with its rationale."""
    if num_digits <= 0:
        return None

    low = 10**(num_digits - 1) if num_digits > 1 else 0
    high = 10**num_digits - 1

    num1 = random.randint(low, high)
    num2 = random.randint(low, high)
    correct_sum = num1 + num2

    # Pad with leading zeros if needed
    num1_str = str(num1).zfill(num_digits)
    num2_str = str(num2).zfill(num_digits)

    question = f"{num1_str} + {num2_str}"
    scratchpad, answer = generate_arithmetic_scratchpad(num1_str, num2_str)

    # Ensure generated answer matches calculation
    if str(answer) != str(correct_sum):
         # This case can happen if padding logic differs slightly from sum logic, rare.
         # Regenerate if mismatch to ensure consistency.
         print(f"Warning: Mismatch for {num1_str} + {num2_str}. Calculated: {correct_sum}, Scratchpad: {answer}. Regenerating...")
         return generate_arithmetic_problem(num_digits) # Try again


    target = f"{scratchpad}\n{answer}"

    return {"question": question, "target": target, "answer": str(answer), "num_digits": num_digits}


def generate_arithmetic_dataset(num_samples, max_digits=5, output_dir='./data_cache/arithmetic', filename='arithmetic_train.jsonl'):
    """Generates and saves the arithmetic dataset."""
    print(f"Generating {num_samples} arithmetic problems (up to {max_digits} digits)...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)

    count = 0
    with open(output_path, 'w') as f:
        for _ in tqdm(range(num_samples), desc="Generating Arithmetic"):
            num_digits = random.randint(1, max_digits)
            problem = generate_arithmetic_problem(num_digits)
            if problem:
                f.write(json.dumps(problem) + '\n')
                count += 1

    print(f"Generated and saved {count} arithmetic problems to {output_path}")
    return output_path

if __name__ == "__main__":
    # Define cache directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(script_dir, 'data_cache')
    # CACHE_DIR = './data_cache' # Old definition
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # --- Load Hugging Face Datasets ---
    cqa_dataset = load_commonsense_qa(cache_dir=CACHE_DIR)
    gsm8k_dataset = load_gsm8k(cache_dir=CACHE_DIR)

    # --- Generate Arithmetic Dataset ---
    # Paper uses 50k samples for the full pool, sampling 10k per iteration.
    # Let's generate the 50k pool.
    NUM_ARITHMETIC_SAMPLES = 50000
    MAX_ARITHMETIC_DIGITS = 5 # As per few-shot examples used in paper
    arithmetic_file = generate_arithmetic_dataset(
        num_samples=NUM_ARITHMETIC_SAMPLES,
        max_digits=MAX_ARITHMETIC_DIGITS,
        output_dir=os.path.join(CACHE_DIR, 'arithmetic')
    )

    print("\n--- Dataset Preparation Summary ---")
    if cqa_dataset:
        print("CommonsenseQA: Available")
    else:
        print("CommonsenseQA: Failed to load")

    if gsm8k_dataset:
        print("GSM8K: Available")
    else:
        print("GSM8K: Failed to load")

    if os.path.exists(arithmetic_file):
        print(f"Arithmetic: Generated dataset at {arithmetic_file}")
    else:
        print("Arithmetic: Failed to generate dataset")

    print(f"All downloaded/generated data stored in or referenced from: {os.path.abspath(CACHE_DIR)}")
