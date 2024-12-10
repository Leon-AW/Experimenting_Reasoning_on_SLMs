import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, pipeline
from tqdm import tqdm
import csv

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 1.0
TOP_K = 0
DO_SAMPLE = True
NUM_RETURN_SEQUENCES = 1
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEBUG = False  # Set to True to enable debug prints

# Simple question prompt
PROMPT_TEMPLATE = """Question: {question}
"""

# Generic prompt template
PROMPT_TEMPLATE1 = """You are a helpful and accurate reasoning assistant. Solve the following math problem step-by-step, then conclude with 'The final answer is X' without any formatting or special characters.\n\nProblem: {question}\n\nStep-by-step reasoning:\n
"""

# More detailed general reasoning prompt
PROMPT_TEMPLATE2 = """Question: {question}

    Let's solve this step by step:
    1. First, let's understand what the question is asking
    2. Break down the problem into smaller parts
    3. Solve each part systematically
    4. Double check our calculations
    5. Combine the results to get our final answer
    6. The final answer is: [your answer] (without any formatting or special characters)

    Your Reasoning: """

# 2 More detailed general reasoning prompt
PROMPT_TEMPLATE3 = """Question: {question}

    Let's solve this problem step by step:
    1. First, understand the problem and clarify what the question is asking.
    2. Break the problem into smaller, manageable parts or sub-problems.
    3. Solve each sub-problem systematically, providing clear reasoning for each step.
    4. Review and verify the calculations and logic to ensure accuracy.
    5. Combine the results from each step to determine the final answer.
    6. The final answer is: [your answer] (without any formatting or special characters)

    Your Reasoning: """


def extract_numeric_answer(generated_text):
    """
    Extract numeric answer from generated text.
    """
    # First try to find "The final answer is" pattern
    pattern = r"(?:The final answer is|Answer:)\s*\$?([\d,]+(?:\.\d+)?)"
    match = re.search(pattern, generated_text, re.IGNORECASE)
    if match:
        clean_number = match.group(1).replace(',', '').replace('$', '')
        try:
            return str(int(float(clean_number)))
        except ValueError:
            return clean_number

    # If not found, look for any number after a dollar sign
    pattern = r"\$\s*([\d,]+(?:\.\d+)?)"
    numbers = re.findall(pattern, generated_text)
    if numbers:
        clean_number = numbers[-1].replace(',', '')
        try:
            return str(int(float(clean_number)))
        except ValueError:
            return clean_number

    # Last resort: find any numbers
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", generated_text)
    if numbers:
        clean_number = numbers[-1].replace(',', '')
        try:
            return str(int(float(clean_number)))
        except ValueError:
            return clean_number

    return None

def main():
    # Load dataset
    dataset = load_dataset("gsm8k", "main")
    test_dataset = dataset['test']

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto").to(DEVICE)
    model.eval()

    # Create a pipeline for reasoning tasks
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", device=0 if DEVICE == "cuda" else -1)

    # Get the eos_token_id from the tokenizer
    eos_token_id = pipe.tokenizer.eos_token_id

    # Evaluate with progress bar
    correct = 0
    total = 0
    results = []  # To store results for CSV

    for example in tqdm(test_dataset, desc="Processing examples"):
        question = example["question"]
        gold_answer = example["answer"].strip()

        # Extract the final numeric answer from the gold answer
        gold_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", gold_answer)
        gold_final = gold_numbers[-1].strip() if gold_numbers else None

        prompt = PROMPT_TEMPLATE.format(question=question)
        # Generate answer with explicit pad_token_id
        outputs = pipe(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=DO_SAMPLE,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            pad_token_id=eos_token_id  # Explicitly set pad_token_id
        )
        generated_text = outputs[0]["generated_text"]
        pred_answer = extract_numeric_answer(generated_text)
        
        # Print details of the prediction if debugging is enabled
        if DEBUG:
            print(f"Question: {question}")
            print(f"Generated Text: {generated_text}")
            print(f"Extracted Answer: {pred_answer}")
            print(f"Gold Answer: {gold_final}")
        
        # Compare prediction to gold_final
        is_correct = False
        if pred_answer is not None and gold_final is not None:
            try:
                pred_num = float(pred_answer.replace(",", ""))
                gold_num = float(gold_final.replace(",", ""))
                if abs(pred_num - gold_num) < 1e-7:
                    correct += 1
                    is_correct = True
            except ValueError:
                pass

        if DEBUG:
            print(f"Result: {'Correct' if is_correct else 'Incorrect'}\n")
        
        total += 1

        if total % 10 == 0:
            print(f"Processed {total} examples. Current Accuracy: {correct/total:.2%}")

        # Store the result for CSV
        results.append({
            "question": question,
            "prompt": prompt,
            "generated_text": generated_text,
            "pred_answer": pred_answer,
            "gold_answer": gold_final,
            "is_correct": is_correct
        })

    final_accuracy = correct / total if total > 0 else 0.0
    print(f"Final Accuracy on GSM8K test set: {final_accuracy:.2%}")

    # Print the total number of correct answers
    print(f"Total Correct Answers: {correct}")

    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Save results to CSV in the specified directory
    csv_file_path = os.path.join('results', 'gsm8k_prompting_results.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["question", "prompt", "generated_text", "pred_answer", "gold_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()
