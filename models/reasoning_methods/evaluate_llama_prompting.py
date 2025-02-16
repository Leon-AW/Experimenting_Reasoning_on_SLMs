import os
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from tqdm import tqdm
import csv
import argparse
import time
from collections import Counter

# Configuration
BATCH_SIZE = 64
NUM_GPUS = 4
MAX_MEMORY = "40GB"
MIN_NEW_TOKENS = 1
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
SEED = 42
TOP_P = 0.90
TOP_K = 40
DO_SAMPLE = True
NUM_RETURN_SEQUENCES = 1
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEBUG = True  # Set to True to enable debug prints
SELF_CONSISTENCY_PATHS = 15  # Reduced from paper's 40 to 15 paths

# Define prompt templates as a dictionary
PROMPT_TEMPLATES = {
   # Simple question prompt
    "simple": {
        "numeric": """Problem: {question} \n\nSolve the problem, then conclude it with 'The final answer is: <insert your answer here>'. \n\nAnswer: """,
        "multiple_choice": """Question: {question} \n\nOptions:\n{options}\n\nChoose the correct answer and conclude with 'The answer is: <A/B/C/D>'. \n\nAnswer: """
    },
    # Large Language Models are Zero-Shot Reasoners: https://arxiv.org/abs/2205.11916
    "chain": {
        "numeric": """Problem: {question} \n\nSolve the problem step-by-step, then conclude it with 'The final answer is: <insert your answer here>'. \n\nLet's think step by step: """,
        "multiple_choice": """Question: {question} \n\nOptions:\n{options}\n\nYou must provide a complete answer and conclude with 'The answer is: <A/B/C/D>'.

Let's solve this step-by-step: """
    },
    # Role-Setting Prompt: https://aclanthology.org/2024.naacl-long.228/
    "role": {
        "numeric": """From now on, you are an excellent teacher. One of your students wants to ask you a question. \nYou explain it and conclude your answer with 'The final answer is: <insert your answer here>'.
\n\nQuestion: {question} \n\nAnswer: """,
        "multiple_choice": """From now on, you are an excellent teacher. One of your students wants to ask you a question. 

Question: {question}

Options:
{options}

Explain what the correct answer is and conclude it with 'The answer is: <A/B/C/D>'.

"""
    },
    # Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models: https://arxiv.org/abs/2305.04091
    "plan": {
        "numeric": """Problem: {question} \n\nLet's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate intermediate variables, solve the problem step by step, and show the answer. 
\n\nFinally conclude your answer with 'The final answer is: <insert your answer here>'. \n\nAnswer: """,
        "multiple_choice": """Question: {question}

Options:
{options}

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
    Extract multiple choice answers from any multiple choice dataset responses.
    This updated version includes additional fallback steps to capture cases where the answer
    appears on a standalone line (or after an empty "Answer:" delimiter).
    """
    # First check if the model actually generated any reasoning past the prompt
    parts = generated_text.split("Options:")
    if len(parts) < 2:
        return None
    
    options_and_answer = parts[1]
    
    # Try to split on various possible delimiters
    delimiters = [
        "Answer:", 
        "Let's solve this step-by-step:",
        "Solution:",
        "Therefore,",
        "Thus,",
        "So,",
        "In conclusion,"
    ]
    
    for delimiter in delimiters:
        if delimiter in options_and_answer:
            options_text, answer_text = options_and_answer.split(delimiter, 1)
            break
    else:
        # If no delimiter found, use the entire text after options
        options_text = options_and_answer
        answer_text = options_and_answer

    # Parse options into a dictionary
    options = {}
    for line in options_text.strip().split('\n'):
        match = re.match(r'([A-D])\)(.*)', line.strip())
        if match:
            letter, text = match.groups()
            letter = letter.upper()
            text = text.strip()
            options[letter] = text
            # Store normalized text version (for general text matching)
            options[f"{letter}_norm"] = re.sub(r'[^\w\s]', '', text.lower()).strip()
            # Store chemical equation version (removing spaces and normalizing)
            chemical_text = text.replace(" ", "")
            chemical_text = re.sub(r'[{}_]', '', chemical_text)
            options[f"{letter}_chem"] = chemical_text
            # Try to extract numeric value if present
            num_match = re.search(r'(?:^|[^\d,])(\d+(?:,\d+)?)', text.replace(" ", ""))
            if num_match:
                # Convert string to number, handling commas
                num_str = num_match.group(1).replace(",", "")
                options[f"{letter}_num"] = int(num_str)
    
    # Get the answer section
    answer_section = answer_text.strip()
    
    # Enhanced letter patterns with better prioritization
    letter_patterns = [
        # Explicit "The correct answer is" patterns (highest priority)
        r"The correct answer is ['\"]*([A-D])['\"]*",  # New pattern for quoted answers
        r"The (?:correct )?answer is:?\s*([A-D])\b",
        r"The (?:correct )?answer is:?\s*(?:option|choice)?\s*([A-D])\b",
        
        # Clear statement patterns
        r"(?:Therefore|Thus|Hence|So),?\s+(?:the\s+)?(?:answer|choice|option)\s+is\s+([A-D])\b",
        r"(?:I\s+)?conclude\s+(?:that\s+)?(?:the\s+)?(?:answer|choice|option)\s+is\s+([A-D])\b",
        
        # Other explicit patterns
        r"(?:Option|Choice)[:\s]\s*([A-D])\b",
        r"([A-D])\s*is\s+(?:the\s+)?correct\b",
        r"(?:select|choose|pick)\s+(?:option|choice)?\s*([A-D])\b",
        
        # Last line patterns (only if it's a single letter)
        r"^([A-D])$",  # Exact single letter on its own line
        r"^(?:Option|Choice)?\s*([A-D])$",  # Single letter with possible prefix
        
        # Fallback patterns (lowest priority)
        r"\b([A-D])\b(?=[^a-z]*$)",  # Letter at the end
        r"(?:answer|option|choice)\s*=\s*([A-D])\b"
        # immediate answer followed by newline and explanation
        r"^\s*([A-D])\s*(?:\n|$)",  # Matches single letter at start followed by newline
    ]
    
    # First try explicit letter patterns
    for pattern in letter_patterns:
        matches = list(re.finditer(pattern, answer_section, re.IGNORECASE))
        if matches:
            # Take the last match if multiple exist
            return matches[-1].group(1).upper()
    
    # Get base option keys (A-D only)
    base_keys = [k for k in options if len(k) == 1 and k in "ABCD"]
    
    # If no direct letter found, try content matching
    if answer_section:
        # Special handling for chemical equations
        if '->' in answer_section or '→' in answer_section:
            chemical_answer = answer_section.replace(" ", "")
            chemical_answer = re.sub(r'[{}_]', '', chemical_answer)
            for letter in base_keys:
                if chemical_answer == options[f"{letter}_chem"]:
                    return letter
        
        # Try to extract numeric value from answer
        num_match = re.search(r'(?:^|[^\d,])(\d+(?:,\d+)?)', answer_section.replace(" ", ""))
        if num_match:
            # Convert answer string to number, handling commas
            answer_num = int(num_match.group(1).replace(",", ""))
            
            # Look for exact numeric matches
            for letter in base_keys:
                if f"{letter}_num" in options and options[f"{letter}_num"] == answer_num:
                    return letter
        
        # Clean answer for text comparison
        normalized_answer = re.sub(r'[^\w\s]', '', answer_section.lower()).strip()
        
        # Try exact normalized match first
        for letter in base_keys:
            if normalized_answer == options[f"{letter}_norm"]:
                return letter
        
        # Only try partial matching if no exact matches found
        best_match = None
        best_match_length = 0
        min_match_length = 5  # Increased minimum length for partial matches
        
        for letter in base_keys:
            norm_text = options[f"{letter}_norm"]
            
            # Check both directions of containment
            if len(norm_text) >= min_match_length and norm_text in normalized_answer:
                if len(norm_text) > best_match_length:
                    best_match = letter
                    best_match_length = len(norm_text)
            elif len(normalized_answer) >= min_match_length and normalized_answer in norm_text:
                if len(normalized_answer) > best_match_length:
                    best_match = letter
                    best_match_length = len(normalized_answer)
        
        if best_match:
            return best_match
    
    # If no match found, return None instead of defaulting
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
    The returned function expects a full_text parameter that contains both 
    the prompt and model response concatenated.
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

def format_prompt(template_name, dataset_type, question, options=None, passage=None):
    """Format prompt according to template and dataset type."""
    
    # Get the correct template
    if dataset_type in ["race", "arc", "mmlu", "agieval"]:
        template = PROMPT_TEMPLATES[template_name]["multiple_choice"]
    else:
        template = PROMPT_TEMPLATES[template_name]["numeric"]
    
    # Format options if present
    formatted_options = ""
    if options:
        if isinstance(options, list):
            for i, opt in enumerate(options):
                formatted_options += f"{chr(65+i)}) {opt}\n"
        elif isinstance(options, dict):
            for letter, text in sorted(options.items()):
                formatted_options += f"{letter}) {text}\n"
    
    # Create the main body of the prompt using the template
    prompt_body = template.format(question=question, options=formatted_options)
    
    # If the dataset is RACE and a passage exists, prepend the passage to the prompt
    if dataset_type == "race" and passage:
        return f"Passage: {passage}\n\n" + prompt_body
    else:
        return prompt_body

def get_prompt_ending(template_name, dataset_type):
    """Extract the ending text from the prompt template."""
    template_type = "numeric" if dataset_type in ["gsm8k", "drop"] else "multiple_choice"
    template = PROMPT_TEMPLATES[template_name][template_type]
    
    # Find the last segment after the last \n\n
    segments = template.split('\n\n')
    # Remove trailing whitespace from the ending
    return segments[-1].rstrip()

def process_dataset_batch(pipe, dataset, template_name, args):
    """Process dataset using batched inference with HF Dataset API"""
    if args.debug:
        print("Debug mode is ON in process_dataset_batch")

    correct = 0
    total = 0
    results = []
    max_samples = min(1000, len(dataset))
    batch_size = BATCH_SIZE  # use the global BATCH_SIZE for batching

    if args.self_consistency:
        # Process in batches for self-consistency (multiple paths per example)
        for start_idx in tqdm(range(0, max_samples, batch_size),
                              desc=f"Processing {template_name} in batches (self consistency)"):
            batch_meta = []      # For each sample in the batch, store metadata and the original prompt.
            replicated_prompts = []  # Will store (prompt repeated SELF_CONSISTENCY_PATHS times) for each sample.

            # Prepare the batch: for each sample replicate the prompt
            for idx in range(start_idx, min(start_idx + batch_size, max_samples)):
                try:
                    example = dataset[idx]
                    question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                    gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

                    # Extract options and passage if available
                    options = None
                    passage = None
                    if args.dataset == "race":
                        options = example.get("options", [])
                        passage = example.get("article", "") or example.get("passage", "")
                    elif args.dataset == "arc":
                        choices = example.get("choices", {})
                        if isinstance(choices, dict) and "text" in choices:
                            options = choices["text"]
                        else:
                            continue
                    elif args.dataset == "mmlu":
                        options = [example.get(f"choice_{i}", "") for i in range(4) if example.get(f"choice_{i}")]
                    elif args.dataset == "agieval":
                        options = example.get("options", [])

                    formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
                    # Append the same prompt SELF_CONSISTENCY_PATHS times
                    replicated_prompts.extend([formatted_prompt] * SELF_CONSISTENCY_PATHS)
                    batch_meta.append({
                        "sample_index": idx,
                        "question": question,
                        "gold_answer": gold_answer,
                        "prompt": formatted_prompt
                    })
                except Exception as e:
                    if args.debug:
                        print(f"Error preparing self-consistency batch for sample index {idx}: {str(e)}")
                    continue

            if not replicated_prompts:
                continue

            try:
                # Note: Since we already replicated each prompt, set num_return_sequences=1.
                outputs = pipe(
                    replicated_prompts,
                    min_new_tokens=MIN_NEW_TOKENS,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=pipe.tokenizer.eos_token_id
                )

                if args.debug:
                    expected_count = len(batch_meta) * SELF_CONSISTENCY_PATHS
                    print(f"\nGenerated batch self-consistency outputs for sample indices {start_idx} to {min(start_idx + batch_size, max_samples)-1}")
                    print(f"Total outputs generated: {len(outputs)} (expected {expected_count})")
                    if outputs:
                        print(f"First output in batch: {outputs[0]}")
            except Exception as e:
                if args.debug:
                    print(f"Error in batch self-consistency generation for samples {start_idx} to "
                          f"{min(start_idx + batch_size, max_samples)-1}: {str(e)}")
                continue

            # Group outputs for each sample (each sample should have SELF_CONSISTENCY_PATHS outputs)
            for i, meta in enumerate(batch_meta):
                start_pos = i * SELF_CONSISTENCY_PATHS
                end_pos = start_pos + SELF_CONSISTENCY_PATHS
                sample_outputs = outputs[start_pos:end_pos]

                answers = []
                model_responses = []

                for output in sample_outputs:
                    try:
                        # Fix: Extract generated_text from the pipeline output structure
                        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                            generated_text = output[0].get('generated_text', '')
                        else:
                            generated_text = output.get("generated_text", "") if isinstance(output, dict) else str(output)

                        if generated_text:
                            # Remove the prompt from the generated output
                            model_response = generated_text[len(meta["prompt"]):].strip()
                            model_responses.append(model_response)
                            answer = get_answer_extractor(args.dataset)(generated_text)
                            if answer is not None:
                                answers.append(str(answer).upper())

                            if args.debug:
                                print(f"\nSample index {meta['sample_index']} self-consistency path generated text:")
                                print(generated_text)
                                print(f"Extracted answer: {answer}")
                    except Exception as e:
                        if args.debug:
                            print(f"Error processing self-consistency output for sample index {meta['sample_index']}: {str(e)}")
                        continue

                # Majority vote on the answers
                pred_answer = None
                if answers:
                    counts = Counter(answers)
                    max_count = max(counts.values())  # not used further, but shows max frequency
                    candidates = [k for k, v in counts.items() if v == max_count]
                    pred_answer = candidates[0] if candidates else None

                    if args.debug:
                        print(f"\nAll answers for sample index {meta['sample_index']}: {answers}")
                        print(f"Selected answer: {pred_answer}")
                        print(f"Gold answer: {meta['gold_answer']}")

                # Compare prediction with gold answer
                is_correct = False
                if pred_answer is not None and meta["gold_answer"] is not None:
                    if args.dataset in ["gsm8k", "drop"]:
                        try:
                            pred_num = float(pred_answer.replace(',', ''))
                            gold_num = float(meta["gold_answer"].replace(',', ''))
                            is_correct = abs(pred_num - gold_num) < 1e-7
                        except ValueError as e:
                            if args.debug:
                                print(f"Error converting numbers for sample index {meta['sample_index']}: {e}")
                    else:
                        is_correct = pred_answer.upper() == meta["gold_answer"].upper()

                if is_correct:
                    correct += 1
                total += 1

                model_response_text = "\n".join(model_responses)

                if args.debug:
                    print(f"Result for sample index {meta['sample_index']}: {'Correct' if is_correct else 'Incorrect'}\n")

                results.append({
                    "sample_index": meta["sample_index"],
                    "prompt": meta["prompt"],
                    "generated_text": model_response_text,
                    "pred_answer": pred_answer,
                    "gold_answer": meta["gold_answer"],
                    "is_correct": is_correct
                })

    else:
        # Single path processing in batches
        for start_idx in tqdm(range(0, max_samples, batch_size), desc=f"Processing {template_name} in batches"):
            batch_prompts = []
            batch_examples = []
            for idx in range(start_idx, min(start_idx + batch_size, max_samples)):
                try:
                    example = dataset[idx]
                    question = example[DATASET_CONFIGS[args.dataset]["question_key"]]
                    gold_answer = str(example[DATASET_CONFIGS[args.dataset]["answer_key"]]).strip()

                    # Extract options and passage if available
                    options = None
                    passage = None
                    if args.dataset == "race":
                        options = example.get("options", [])
                        passage = example.get("article", "")
                    elif args.dataset == "arc":
                        choices = example.get("choices", {})
                        if isinstance(choices, dict) and "text" in choices:
                            options = choices["text"]
                        else:
                            continue
                    elif args.dataset == "mmlu":
                        options = [example.get(f"choice_{i}", "") for i in range(4) if example.get(f"choice_{i}")]
                    elif args.dataset == "agieval":
                        options = example.get("options", [])

                    formatted_prompt = format_prompt(template_name, args.dataset, question, options, passage)
                    batch_prompts.append(formatted_prompt)
                    batch_examples.append({
                        "sample_index": idx,
                        "question": question,
                        "gold_answer": gold_answer,
                        "prompt": formatted_prompt
                    })
                except Exception as e:
                    if args.debug:
                        print(f"Error preparing batch for sample index {idx}: {str(e)}")
                    continue

            try:
                outputs = pipe(
                    batch_prompts,
                    min_new_tokens=MIN_NEW_TOKENS,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    do_sample=DO_SAMPLE,
                    num_return_sequences=NUM_RETURN_SEQUENCES,
                    pad_token_id=pipe.tokenizer.eos_token_id
                )

                if args.debug:
                    print(f"\nGenerated batch output for sample indices {start_idx} to {min(start_idx + batch_size, max_samples)-1}")
                    print(f"Output structure: {type(outputs)}")
                    if outputs:
                        print(f"First output in batch: {outputs[0]}")
            except Exception as e:
                if args.debug:
                    print(f"Error in batch single-path processing for samples {start_idx} to {min(start_idx + batch_size, max_samples)-1}: {str(e)}")
                continue

            for i, output in enumerate(outputs):
                idx = batch_examples[i]["sample_index"]
                question = batch_examples[i]["question"]
                gold_answer = batch_examples[i]["gold_answer"]
                formatted_prompt = batch_examples[i]["prompt"]

                # Fix: Extract generated_text from the pipeline output structure
                if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                    generated_text = output[0].get('generated_text', '')
                else:
                    generated_text = output.get("generated_text", "") if isinstance(output, dict) else str(output)

                if generated_text:
                    # Remove the prompt from the generated output by finding where it starts
                    model_response = generated_text[len(formatted_prompt):].strip()
                    
                    # Create full text by concatenating prompt and model response
                    full_text = formatted_prompt + model_response
                    
                    # Use the full text for answer extraction
                    pred_answer = get_answer_extractor(args.dataset)(full_text)

                    if args.debug:
                        print(f"\nBatch sample index: {idx}")
                        print(f"Prompt: {formatted_prompt}")
                        print(f"Model response: {model_response}")
                        print(f"Extracted answer: {pred_answer}")
                        print(f"Gold answer: {gold_answer}")

                    is_correct = False
                    if pred_answer is not None and gold_answer is not None:
                        if args.dataset in ["gsm8k", "drop"]:
                            try:
                                pred_num = float(pred_answer.replace(',', ''))
                                gold_num = float(gold_answer.replace(',', ''))
                                is_correct = abs(pred_num - gold_num) < 1e-7
                            except ValueError as e:
                                if args.debug:
                                    print(f"Error converting numbers for batch sample index {idx}: {e}")
                        else:
                            is_correct = pred_answer.upper() == gold_answer.upper()

                    results.append({
                        "sample_index": idx,
                        "prompt": formatted_prompt,
                        "generated_text": model_response,
                        "pred_answer": pred_answer,
                        "gold_answer": gold_answer,
                        "is_correct": is_correct
                    })

                    if is_correct:
                        correct += 1
                    total += 1
                else:
                    if args.debug:
                        print(f"No generated text found in batch output for sample index {idx}.")
    return correct, total, results

def main():
    # Add argument parsing
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
    
    # Configure max memory for each GPU
    max_memory = {i: MAX_MEMORY for i in range(NUM_GPUS)}
    
    # Load model with balanced memory allocation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="balanced" if NUM_GPUS > 1 else "auto",
        max_memory=max_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    # Create pipeline without device specification
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        batch_size=BATCH_SIZE * NUM_GPUS,
    )
    
    # Configure tokenizer
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'

    # Get the eos_token_id from the tokenizer
    eos_token_id = pipe.tokenizer.eos_token_id

    print(f"\nStarting evaluation with templates: {list(PROMPT_TEMPLATES.keys())}")
    
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
            time.sleep(5)

    # Process each prompt template using batched inference
    for template_name in PROMPT_TEMPLATES.keys():
        print(f"\n{'='*50}")
        print(f"Starting template: {template_name}")
        print(f"{'='*50}\n")
        
        try:
            correct, total, results = process_dataset_batch(pipe, dataset, template_name, args)
            
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

            # Save accuracy and unextracted answers info
            txt_file_path = os.path.join('results', f'{args.dataset}_{template_name}_{args.model_size}{sc_info}_total_accuracy.txt')
            with open(txt_file_path, mode='w') as file:
                file.write(f"Final Accuracy of {template_name} on {args.dataset}: {final_accuracy:.2%}\n")
                file.write(f"Total Correct Answers: {correct}/{total} Questions\n")
                file.write(f"\nUnextracted Answers: {total - len(results)} samples\n")
                if total > len(results):
                    file.write(f"Number of unextracted answers: {total - len(results)}\n")

            print(f"\nCompleted template: {template_name}")
            print(f"Moving to next template...")

        except Exception as e:
            print(f"Error processing template {template_name}: {str(e)}")
            print("Moving to next template...")
            continue  # Continue to next template on error

    print("\nCompleted all templates!")

if __name__ == "__main__":
    main()
