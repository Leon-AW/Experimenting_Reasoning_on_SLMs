import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, pipeline
from tqdm import tqdm
import csv

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
SEED = 42
TOP_P = 1.0
TOP_K = 0
DO_SAMPLE = True
NUM_RETURN_SEQUENCES = 1
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEBUG = True  # Set to True to enable debug prints

# Simple question prompt
PROMPT_TEMPLATE = """Solve the following problem, then conclude it with 'The final answer is X' without any formatting or special characters.\n\nProblem: {question}\n\nAnswer: """

# Large Language Models are Zero-Shot Reasoners: https://arxiv.org/abs/2205.11916
PROMPT_TEMPLATE1 = """Solve the following problem step-by-step, then conclude it with 'The final answer is X' without any formatting or special characters.\n\nProblem: {question}\n\nLet's think step by step:\n
"""

# Role-Setting Prompt: https://aclanthology.org/2024.naacl-long.228/
PROMPT_TEMPLATE2 = """From now on, you are an excellent teacher and are teaching your students to get a new word by concatenating the last
letters of several words. I am one of your students and want to ask you a related question.\nFinally conclude your answer with 'The final answer is X' without any formatting or special characters. \n\Question: {question}\n"""

# Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models: https://arxiv.org/abs/2305.04091
PROMPT_TEMPLATE3 = """Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numeral calculation and commonsense), solve the problem step by step, and show the answer. 
\n\nFinally conclude your answer with 'The final answer is X' without any formatting or special characters.\n\n Question: {question}\n"""



def extract_numeric_answer(generated_text):
    """
    Extract numeric answer from generated text.
    """
    # First try to find "The final answer is" pattern
    final_answer_patterns = [
        r"The final answer is:?\s*\$?\\?\s*(?:boxed{)?(\d[\d,]*(?:\.\d+)?)}?",  # Matches LaTeX \boxed{} and similar
        r"The final answer is:?\s*\$?([\d,]+(?:\.\d+)?)",  # Regular number pattern
        r"The final answer is:?\s*\$?\s*(?:is\s+)?(\d[\d,]*(?:\.\d+)?)",  # More flexible pattern
    ]
    
    for pattern in final_answer_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            clean_number = match.group(1).replace(',', '').replace('$', '').replace('\\', '').strip()
            try:
                return clean_number
            except ValueError:
                continue

    # Look for "Answer:" or "Answer is:" followed by a number
    answer_patterns = [
        r"[Aa]nswer(?:\s+is)?:\s*=\s*(\d[\d,]*(?:\.\d+)?)[^a-zA-Z]*",  # Match number after equals sign
        r"[Aa]nswer(?:\s+is)?:\s*(\d[\d,]*(?:\.\d+)?)[^a-zA-Z]*"  # Match first number after Answer:
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, generated_text)  # Removed re.IGNORECASE since we handle case in pattern
        if match:
            clean_number = match.group(1).replace(',', '').replace('$', '')
            try:
                return clean_number
            except ValueError:
                continue

    # If no calculation found, try other patterns
    other_patterns = [
        r"(?:answer|solution)(?:\s+is)?:?\s*\$?\s*(\d[\d,]*(?:\.\d+)?)",
        r"=\s*(\d[\d,]*(?:\.\d+)?)\s*$",
    ]
    
    for pattern in other_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            clean_number = match.group(1).replace(',', '').replace('$', '')
            try:
                return clean_number
            except ValueError:
                continue

    # Last resort: find the last number in the text
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", generated_text)
    if numbers:
        clean_number = numbers[-1].replace(',', '').replace('$', '')
        try:
            return clean_number
        except ValueError:
            pass

    return None

def main():
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.backends.mps.is_available():
        torch.manual_seed(SEED)  # MPS doesn't have a separate seed function

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load dataset
    dataset = load_dataset("gsm8k", "main")
    test_dataset = dataset['test']

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto").to(DEVICE)
    model.eval()

    # Create a pipeline for reasoning tasks
    pipe = pipeline(
        "text-generation", 
        model=MODEL_NAME, 
        device=DEVICE,
        use_cache=True,
    )

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
            except ValueError as e:
                if DEBUG:
                    print(f"Error converting numbers: {e}")

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
