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
PROMPT_TEMPLATE = """Solve the following Question.
Question: {question}
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

PROMPT_TEMPLATES = {
    "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
    "PROMPT_TEMPLATE1": PROMPT_TEMPLATE1,
    "PROMPT_TEMPLATE2": PROMPT_TEMPLATE2,
    "PROMPT_TEMPLATE3": PROMPT_TEMPLATE3
}

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

    for template_name, prompt_template in PROMPT_TEMPLATES.items():
        correct = 0
        total = 0
        results = []  # To store results for CSV

        for example in tqdm(test_dataset, desc=f"Processing examples with {template_name}"):
            question = example["question"]
            gold_answer = example["answer"].strip()

            # Extract the final numeric answer from the gold answer
            gold_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", gold_answer)
            gold_final = gold_numbers[-1].strip() if gold_numbers else None

            prompt = prompt_template.format(question=question)
            outputs = pipe(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                do_sample=DO_SAMPLE,
                num_return_sequences=NUM_RETURN_SEQUENCES,
                pad_token_id=eos_token_id
            )
            generated_text = outputs[0]["generated_text"]
            pred_answer = extract_numeric_answer(generated_text)

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

            total += 1

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
        print(f"Final Accuracy of {template_name} on GSM8K test set: {final_accuracy:.2%}")
        print(f"Total Correct Answers: {correct}/{total} Questions")

        # Ensure the results directory exists
        os.makedirs('results', exist_ok=True)

        # Save results to CSV
        csv_file_path = os.path.join('results', f'{template_name}_results.csv')
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["question", "prompt", "generated_text", "pred_answer", "gold_answer", "is_correct"])
            writer.writeheader()
            writer.writerows(results)

        # Save accuracy to a text file
        txt_file_path = os.path.join('results', f'{template_name}_total_accuracy.txt')
        with open(txt_file_path, mode='w') as file:
            file.write(f"Final Accuracy of {template_name} on GSM8K test set: {final_accuracy:.2%}\n")
            file.write(f"Total Correct Answers: {correct}/{total} Questions\n")

if __name__ == "__main__":
    main()
