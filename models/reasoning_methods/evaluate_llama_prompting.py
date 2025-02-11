import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, pipeline
from tqdm import tqdm
import csv
import argparse
import time

# Configuration
BATCH_SIZE = 1
MIN_NEW_TOKENS = 1
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.5
SEED = 42
TOP_P = 0.95
TOP_K = 0
DO_SAMPLE = True
NUM_RETURN_SEQUENCES = 1
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEBUG = True  # Set to True to enable debug prints

# Define prompt templates as a dictionary
PROMPT_TEMPLATES = {
   # Simple question prompt
    "simple": {
        "numeric": """Problem: {question} \n\nSolve the problem, then conclude it with 'The final answer is: <insert your answer here>'. \n\nAnswer: """,
        "multiple_choice": """Question: {question} \n\nOptions:\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n\nChoose the correct answer and conclude with 'The answer is: <A/B/C/D>'. \n\nAnswer: """
    },
    # Large Language Models are Zero-Shot Reasoners: https://arxiv.org/abs/2205.11916
    "chain": {
        "numeric": """Problem: {question} \n\nSolve the problem step-by-step, then conclude it with 'The final answer is: <insert your answer here>'. \n\nLet's think step by step: """,
        "multiple_choice": """Question: {question} \n\nOptions:\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n\nYou must provide a complete answer and conclude with 'The answer is: <A/B/C/D>'.

Let's solve this step-by-step: """
    },
    # Role-Setting Prompt: https://aclanthology.org/2024.naacl-long.228/
    "role": {
        "numeric": """From now on, you are an excellent teacher. One of your students wants to ask you a question. \nYou explain it and conclude your answer with 'The final answer is: <insert your answer here>'.
\n\nQuestion: {question} \n\nAnswer: """,
        "multiple_choice": """From now on, you are an excellent teacher. One of your students wants to ask you a question. 

Question: {question}

Options:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Explain your answer to the student and conclude it with 'The answer is: <A/B/C/D>'.

Teacher: """
    },
    # Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models: https://arxiv.org/abs/2305.04091
    "plan": {
        "numeric": """Problem: {question} \n\nLet's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate intermediate variables, solve the problem step by step, and show the answer. 
\n\nFinally conclude your answer with 'The final answer is: <insert your answer here>'. \n\nAnswer: """,
        "multiple_choice": """Question: {question}

Options:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Let's approach this systematically:
1. First, let's understand the question
2. Then, analyze each option carefully
3. Finally, choose the most appropriate answer and conclude it with 'The answer is: <A/B/C/D>'.

"""
    }
}

# Add new dataset configurations
DATASET_CONFIGS = {
    "gsm8k": {
        "name": "gsm8k",
        "split": "main",
        "subset": "test",
        "question_key": "question",
        "answer_key": "answer"
    },
    "race": {
        "name": "race",
        "split": "high",
        "subset": "test",
        "question_key": "question",
        "answer_key": "answer"
    },
    "arc": {
        "name": "ai2_arc",
        "split": "ARC-Challenge",
        "subset": "test",
        "question_key": "question",
        "answer_key": "answerKey"
    },
    "mmlu": {
        "name": "cais/mmlu",
        "split": "test",
        "subset": None,
        "question_key": "question",
        "answer_key": "answer"
    },
    "drop": {
        "name": "drop",
        "split": None,
        "subset": "validation",
        "question_key": "question",
        "answer_key": "answer"
    },
    "agieval": {
        "name": "cais/agieval",
        "split": "test",
        "subset": None,
        "question_key": "question",
        "answer_key": "answer"
    },
}

def extract_gsm8k_answer(generated_text):
    """
    Extract numeric answer from GSM8K-style responses.
    Returns the last number found after "The final answer is" or similar patterns.
    """
    # First try to find "The final answer is" pattern
    final_answer_patterns = [
        r"The final answer is:?\s*\$?\\?\s*(?:boxed{)?(\d[\d,]*(?:\.\d+)?)}?",  # Matches LaTeX \boxed{} and similar
        r"The final answer is:?\s*\$?([\d,]+(?:\.\d+)?)",  # Regular number pattern
        r"The final answer is:?\s*\$?\s*(?:is\s+)?(\d[\d,]*(?:\.\d+)?)",  # More flexible pattern
    ]
    
    for pattern in final_answer_patterns:
        matches = list(re.finditer(pattern, generated_text, re.IGNORECASE))
        if matches:
            match = matches[-1]  # Take the last match
            clean_number = match.group(1).replace(',', '').replace('$', '').replace('\\', '').strip()
            try:
                return str(int(float(clean_number) + 0.5))
            except ValueError:
                continue
    
    # If no explicit "final answer" found, look for the last number in the text
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", generated_text)
    if numbers:
        clean_number = numbers[-1].replace(',', '').replace('$', '')
        try:
            return str(int(float(clean_number) + 0.5))
        except ValueError:
            pass
    
    return None

def extract_drop_answer(generated_text):
    """
    Extract numeric answers from DROP-style responses.
    Similar to GSM8K but handles more number formats and units.
    """
    # Try to find "The final answer is" pattern first
    final_patterns = [
        r"The final answer is:?\s*(\d+(?:\.\d+)?)",
        r"The answer is:?\s*(\d+(?:\.\d+)?)",
    ]
    
    for pattern in final_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            try:
                return str(int(float(match.group(1)) + 0.5))
            except ValueError:
                continue
    
    # Look for numbers with common units
    unit_patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:people|points|yards|feet|meters|kilometers|miles|years|dollars)",
        r"\$\s*(\d+(?:\.\d+)?)",
    ]
    
    for pattern in unit_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            try:
                return str(int(float(match.group(1)) + 0.5))
            except ValueError:
                continue
    
    return None

def extract_arc_answer(generated_text):
    """
    Extract multiple choice answers from ARC-style responses.
    Expects A, B, C, or D as answers.
    """
    # First check if the model actually generated any reasoning past the prompt
    reasoning_text = generated_text.split("Answer:")[-1]  # Get only the model's response
    if not reasoning_text.strip():
        return None

    # Look for explicit answer statements with better pattern coverage
    patterns = [
        r"Answer:\s*([A-D])",  # Matches "Answer: A" format
        r"The (?:correct )?answer is[:\s]\s*([A-D])",  # Matches "The answer is: A"
        r"(?:Option|Choice)\s*([A-D])",  # Matches "Option C" or "Choice B"
        r"([A-D])\s*is (?:the )?correct",  # Matches "C is correct"
        r"I (?:choose|select)\s*([A-D])",  # Matches "I choose D"
        r"\b([A-D])\b(?=[^a-z]*$)",  # Single letter at end of response
        r"answer\s*[=:]\s*([A-D])"  # Matches "answer = B"
    ]
    
    # First try to find explicit letter answers
    for pattern in patterns:
        matches = re.findall(pattern, generated_text, re.IGNORECASE)
        if matches:
            # Take last match as final answer
            return matches[-1].upper()
    
    # If no letter found, try to match numeric values to options
    numeric_match = re.search(r'(\d+(?:\.\d+)?(?:\s*(?:km/h|meters|kg|points|dollars|years))?)\s*$', generated_text)
    if numeric_match:
        # Look for this value in the original options
        value = numeric_match.group(1).strip()
        options_pattern = r'[A-D]\)\s*(' + re.escape(value) + r')'
        option_match = re.search(options_pattern, generated_text)
        if option_match:
            # Find which option letter corresponds to this value
            option_section = generated_text.split("Options:")[-1].split("Answer:")[0]
            for line in option_section.split('\n'):
                if value in line:
                    letter_match = re.match(r'([A-D])\)', line)
                    if letter_match:
                        return letter_match.group(1).upper()
    
    # Fallback: Look for first capital letter A-D in last 3 lines
    last_lines = generated_text.strip().split('\n')[-3:]
    for line in reversed(last_lines):
        match = re.search(r'\b([A-D])\b', line)
        if match:
            return match.group(0).upper()
    
    # Return None if no answer found
    return None

def extract_multiple_choice_answer(generated_text):
    """
    Extract multiple choice answers from any multiple choice dataset responses (RACE, ARC, MMLU, AGIEval).
    Handles both direct letter answers and full text answers.
    """
    # First check if the model actually generated any reasoning past the prompt
    parts = generated_text.split("Options:")
    if len(parts) < 2:
        return None
    
    options_and_answer = parts[1]
    
    # Try to split on either "Answer:" or "Let's solve this step-by-step:"
    if "Answer:" in options_and_answer:
        options_text, answer_text = options_and_answer.split("Answer:", 1)
    elif "Let's solve this step-by-step:" in options_and_answer:
        options_text, answer_text = options_and_answer.split("Let's solve this step-by-step:", 1)
    else:
        # If neither delimiter is found, use the entire text after options
        options_text = options_and_answer
        answer_text = options_and_answer

    # Parse options into a dictionary
    options = {}
    for line in options_text.strip().split('\n'):
        match = re.match(r'([A-D])\)(.*)', line.strip())
        if match:
            letter, text = match.groups()
            options[letter.upper()] = text.strip()
            # Also store normalized version (lowercase, no punctuation, no extra spaces)
            options[f"{letter.upper()}_norm"] = re.sub(r'[^\w\s]', '', text.lower()).strip()
    
    # First try to find explicit letter answers in the model's direct response
    answer_section = answer_text.strip()
    
    # Check for single letter answer at the start
    if answer_section and len(answer_section) >= 1 and answer_section[0] in "ABCD":
        return answer_section[0].upper()
    
    # Look for explicit letter answers with common patterns
    letter_patterns = [
        r"^([A-D])[.\s]*$",  # Just the letter with optional period/space
        r"The (?:correct )?answer is[:\s]\s*([A-D])",
        r"(?:Option|Choice)[:\s]\s*([A-D])",
        r"([A-D])\s*is (?:the )?correct",
        r"Answer:\s*([A-D])\b",
        r"I (?:choose|select)\s*([A-D])",
        r"\b([A-D])\b(?=[^a-z]*$)"  # Letter at the end of response
    ]
    
    for pattern in letter_patterns:
        match = re.search(pattern, answer_section, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # If no direct letter found, try to match the content
    if answer_section:
        # Normalize the model's answer
        normalized_answer = re.sub(r'[^\w\s]', '', answer_section.lower()).strip()
        
        # Try exact match first
        for letter, text in options.items():
            if letter.endswith('_norm'):
                continue
            if normalized_answer == text.lower():
                return letter
        
        # Try partial match
        for letter, text in options.items():
            if letter.endswith('_norm'):
                continue
            # Convert both strings to numbers if possible (for temperature comparisons)
            answer_num = re.search(r'(\d+)', normalized_answer)
            option_num = re.search(r'(\d+)', text.lower())
            if answer_num and option_num and answer_num.group(1) == option_num.group(1):
                return letter
        
        # Try normalized match as last resort
        for letter, text in options.items():
            if not letter.endswith('_norm'):
                continue
            if normalized_answer in text or text in normalized_answer:
                return letter[0]  # Remove _norm suffix
    
    # Final fallback: Look for any letter A-D in the last few lines
    last_lines = answer_section.strip().split('\n')[-3:]
    for line in reversed(last_lines):
        match = re.search(r'\b([A-D])\b', line)
        if match:
            return match.group(1).upper()
    
    # Return None if no answer found
    return None

def extract_mmlu_answer(generated_text):
    """
    Extract multiple choice answers from MMLU-style responses.
    Handles both letter choices and numeric choices.
    """
    # Look for explicit answer statements with letters
    letter_patterns = [
        r"The (?:correct )?answer is[:\s]\s*([A-D])",
        r"(?:Option|Choice)[:\s]\s*([A-D])",
        r"([A-D])\s*is correct",
    ]
    
    for pattern in letter_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Look for numeric choices (1-4)
    number_patterns = [
        r"The (?:correct )?answer is[:\s]\s*([1-4])",
        r"(?:Option|Choice)[:\s]\s*([1-4])",
        r"([1-4])\s*is correct",
    ]
    
    for pattern in number_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            # Convert numeric choice to letter (1->A, 2->B, etc.)
            return chr(ord('A') + int(match.group(1)) - 1)
    
    return None

def extract_agieval_answer(generated_text):
    """
    Extract answers from AGIEval-style responses.
    Handles multiple formats including multiple choice and numeric answers.
    """
    # First try multiple choice patterns
    mc_patterns = [
        r"The (?:correct )?answer is[:\s]\s*([A-D])",
        r"(?:Option|Choice)[:\s]\s*([A-D])",
        r"([A-D])\s*is correct",
    ]
    
    for pattern in mc_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Then try numeric patterns
    numeric_patterns = [
        r"The (?:final )?answer is:?\s*(\d+(?:\.\d+)?)",
        r"=\s*(\d+(?:\.\d+)?)\s*$",
    ]
    
    for pattern in numeric_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            try:
                return str(int(float(match.group(1)) + 0.5))
            except ValueError:
                continue
    
    return None

def get_answer_extractor(dataset_name):
    """
    Returns the appropriate answer extraction function based on dataset name.
    """
    if dataset_name in ["race", "arc", "mmlu", "agieval"]:
        return extract_multiple_choice_answer
    elif dataset_name == "gsm8k":
        return extract_gsm8k_answer
    elif dataset_name == "drop":
        return extract_drop_answer
    else:
        return extract_gsm8k_answer  # Default to GSM8K extractor

def get_prompt_template(template_name, dataset_name):
    """Returns the appropriate prompt template based on dataset type"""
    numeric_datasets = ["gsm8k", "drop"]
    template_type = "numeric" if dataset_name in numeric_datasets else "multiple_choice"
    return PROMPT_TEMPLATES[template_name][template_type]

def format_prompt(template_name, dataset_name, question, options=None, passage=None):
    """
    Format the prompt with the appropriate template and options if needed
    """
    template = get_prompt_template(template_name, dataset_name)
    if dataset_name in ["gsm8k", "drop"]:
        return template.format(question=question)
    elif dataset_name == "race":
        if not options or not passage:
            raise ValueError(f"Options and passage are required for RACE dataset")
        return f"Passage: {passage}\n\n" + template.format(question=question, options=options)
    else:
        if not options:
            raise ValueError(f"Options are required for multiple choice dataset {dataset_name}")
        return template.format(question=question, options=options)

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       choices=list(DATASET_CONFIGS.keys()),
                       help='Dataset to evaluate on')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--model_size', type=str, default='3b',
                       choices=['1b', '3b'],
                       help='LLaMA model size (1b or 3b)')
    args = parser.parse_args()

    # Set model name based on size parameter
    MODEL_NAME = f"meta-llama/Llama-3.2-{args.model_size.upper()}"
    
    # Validate dataset choice
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {args.dataset} not supported. Choose from: {list(DATASET_CONFIGS.keys())}")
    
    dataset_config = DATASET_CONFIGS[args.dataset]

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.backends.mps.is_available():
        torch.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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

    for template_name, prompt_template in PROMPT_TEMPLATES.items():
        try:
            correct = 0
            total = 0
            results = []
            
            # Load dataset with error handling and retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    dataset = load_dataset(dataset_config["name"], dataset_config["split"])
                    if dataset_config["subset"]:
                        dataset = dataset[dataset_config["subset"]]
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to load dataset after {max_retries} attempts: {str(e)}")
                        raise
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(5)  # Wait 5 seconds before retrying

            # Limit to the first 1000 samples
            dataset = dataset.select(range(min(1000, len(dataset))))

            for example in tqdm(dataset, desc=f"Processing {template_name}"):
                try:
                    question = example[dataset_config["question_key"]]
                    gold_answer = str(example[dataset_config["answer_key"]]).strip()
                    
                    # Extract options with error handling
                    options = None
                    passage = None
                    if args.dataset == "race":
                        options = example.get("options", [])
                        passage = example.get("article", "")
                    elif args.dataset == "arc":
                        choices = example.get("choices", {})
                        if isinstance(choices, dict) and "text" in choices:
                            options = choices["text"][:4]  # Ensure we only get 4 options
                        else:
                            print(f"Skipping example due to invalid choices format: {choices}")
                            continue
                    elif args.dataset == "mmlu":
                        options = [example.get(f"choice_{i}", "") for i in range(4)]
                    elif args.dataset == "agieval":
                        options = example.get("options", [])
                    
                    # Validate options
                    if options is None or (isinstance(options, list) and len(options) < 4):
                        print(f"Skipping example due to insufficient options: {options}")
                        continue
                    
                    # Format prompt with appropriate template and options
                    try:
                        prompt = format_prompt(template_name, args.dataset, question, options, passage)
                    except Exception as e:
                        print(f"Error formatting prompt: {str(e)}")
                        continue

                    outputs = pipe(
                        prompt,
                        min_new_tokens=MIN_NEW_TOKENS, 
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        do_sample=DO_SAMPLE,
                        num_return_sequences=NUM_RETURN_SEQUENCES,
                        pad_token_id=eos_token_id
                    )
                    
                    generated_text = outputs[0]["generated_text"]
                    pred_answer = get_answer_extractor(args.dataset)(generated_text)
                    
                    # Print details of the prediction if debugging is enabled
                    if args.debug:
                        if args.dataset == "race":
                            print(f"Passage: {passage}")
                        print(f"Generated Text: {generated_text}")
                        print(f"Extracted Answer: {pred_answer}")
                        print(f"Gold Answer: {gold_answer}")
                    
                    # Compare prediction to gold_answer
                    is_correct = False
                    if pred_answer is not None and gold_answer is not None:
                        if args.dataset in ["gsm8k", "drop"]:
                            # For numeric answers
                            try:
                                pred_num = float(pred_answer.replace(',', ''))
                                gold_num = float(gold_answer.replace(',', ''))
                                is_correct = abs(pred_num - gold_num) < 1e-7
                            except ValueError as e:
                                if args.debug:
                                    print(f"Error converting numbers: {e}")
                        else:
                            # For multiple choice answers (A, B, C, D)
                            is_correct = pred_answer.upper() == gold_answer.upper()
                    
                    if is_correct:
                        correct += 1
                    
                    if args.debug:
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
                        "gold_answer": gold_answer,
                        "is_correct": is_correct
                    })

                except Exception as e:
                    print(f"Error processing example: {str(e)}")
                    continue

            final_accuracy = correct / total if total > 0 else 0.0
            print(f"Final Accuracy of {template_name} on {args.dataset}: {final_accuracy:.2%}")
            print(f"Total Correct Answers: {correct}/{total} Questions")

            # Save results
            os.makedirs('results', exist_ok=True)
            
            # Save to CSV
            csv_file_path = os.path.join('results', f'{args.dataset}_{template_name}_{args.model_size}_results.csv')
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["question", "prompt", "generated_text", "pred_answer", "gold_answer", "is_correct"])
                writer.writeheader()
                writer.writerows(results)

            # Save accuracy
            txt_file_path = os.path.join('results', f'{args.dataset}_{template_name}_{args.model_size}_total_accuracy.txt')
            with open(txt_file_path, mode='w') as file:
                file.write(f"Final Accuracy of {template_name} on {args.dataset}: {final_accuracy:.2%}\n")
                file.write(f"Total Correct Answers: {correct}/{total} Questions\n")

        except Exception as e:
            print(f"Error processing dataset {args.dataset}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
