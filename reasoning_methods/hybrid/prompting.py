import torch
import re
import sys
import os
from transformers import PreTrainedModel, PreTrainedTokenizer

# Define config values directly in this file
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.5
TOP_P = 0.9
TOP_K = 0
DO_SAMPLE = True
SEED = 42

def extract_numeric_answer(generated_text):
    """
    Extract numeric answer from generated text and round up at x.5 or higher, otherwise round down.
    
    Updated:
    - If there's a calculation in the final answer line, extract the last number (after the last '=')
    - Otherwise extract the first number after phrases like "The final answer is:"
    """
    import re
    
    # First try: Look for "The final answer is:" followed by a calculation with '='
    # Extract the LAST number after the LAST '=' in that line
    final_answer_calc_pattern = re.compile(
        r"(?:The final answer is|Answer(?:\s+is)?):?.*?=\s*\$?\\?\s*(\d[\d,]*(?:\.\d+)?)(?=[^\d=]*(?:\n|\.|$))",
        re.IGNORECASE
    )
    
    # Find all matches and take the last one (in case there are multiple calculations)
    matches = list(final_answer_calc_pattern.finditer(generated_text))
    if matches:
        match = matches[-1]
        try:
            # Clean the extracted number string and return rounded value
            return str(int(float(match.group(1).replace(',', '').replace('$', '').replace('\\', '')) + 0.5))
        except ValueError:
            pass  # Fall through to next pattern if conversion fails

    # Second try: no '=' after the phrase, so extract the first number immediately following the phrase
    no_eq_pattern = re.compile(
        r"(?:The final answer is|Answer(?:\s+is)?):?\s*\$?\\?\s*(\d[\d,]*(?:\.\d+)?)",
        re.IGNORECASE
    )
    m = no_eq_pattern.search(generated_text)
    if m:
        try:
            return str(int(float(m.group(1).replace(',', '').replace('$', '').replace('\\', '')) + 0.5))
        except ValueError:
            pass

    # Third try: Look for the last occurrence of "=" and extract the number after it
    last_eq_pattern = re.compile(r"=\s*\$?\\?\s*(\d[\d,]*(?:\.\d+)?)(?=[^\d=]*(?:\n|\.|$))")
    matches = list(last_eq_pattern.finditer(generated_text))
    if matches:
        match = matches[-1]  # Take the last match
        try:
            return str(int(float(match.group(1).replace(',', '').replace('$', '').replace('\\', '')) + 0.5))
        except ValueError:
            pass

    # Fallback patterns: try looking for "Answer:" or similar patterns
    answer_patterns = [
        r"[Aa]nswer(?:\s+is)?:\s*=\s*(\d[\d,]*(?:\.\d+)?)[^a-zA-Z]*",
        r"[Aa]nswer(?:\s+is)?:\s*(\d[\d,]*(?:\.\d+)?)[^a-zA-Z]*",
        r"[Aa]nswer(?:\s*[^=\n]*)?=\s*[^=\n]*?(\d[\d,]*(?:\.\d+)?)\s*$"
    ]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, generated_text))
        if matches:
            match = matches[-1]  # Take the last match
            number_str = match.group(1).replace(',', '').replace('$', '')
            try:
                return str(int(float(number_str) + 0.5))
            except ValueError:
                continue

    # Last resort: find the very last number in the text
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", generated_text)
    if numbers:
        number_str = numbers[-1].replace(',', '').replace('$', '')
        try:
            return str(int(float(number_str) + 0.5))
        except ValueError:
            pass

    return None

def score_cqa_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question_data: dict,
    rationale: str
):
    """
    Use log-likelihood scoring to determine the most likely answer choice.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question_data: Dictionary with question data including choices
        rationale: The generated rationale
        
    Returns:
        The most likely answer letter (a, b, c, d, e) based on log-likelihood
    """
    answer_choices = question_data['choices']['label']
    answer_texts = question_data['choices']['text']
    
    # Create completion prompt template
    question = question_data['question']
    
    # Step 1: Build a prompt with the question, rationale, and answer template
    prompt_template = f"Q: {question}\nA: {rationale}\nTherefore, the answer is"
    
    # Tokenize the prompt template
    template_tokens = tokenizer(prompt_template, return_tensors="pt").to(model.device)
    
    # Step 2: For each choice, compute the logits/logprobs
    choice_scores = []
    
    for i, (label, text) in enumerate(zip(answer_choices, answer_texts)):
        # Format like: "Therefore, the answer is pencil (a)."
        completion = f" {text} ({label})."
        
        # Tokenize and get token IDs for the completion
        completion_tokens = tokenizer(completion, add_special_tokens=False, return_tensors="pt").to(model.device)
        completion_ids = completion_tokens.input_ids[0]
        
        # Get logits for the prompt
        with torch.no_grad():
            outputs = model(**template_tokens)
            
        # Get the last token's logits as our starting point
        next_token_logits = outputs.logits[0, -1, :]
        
        # Score each token in the completion
        score = 0.0
        for token_id in completion_ids:
            # Get probability (apply softmax to logits)
            probs = torch.nn.functional.softmax(next_token_logits, dim=0)
            # Get log probability of this token
            token_score = torch.log(probs[token_id]).item()
            score += token_score
            
            # Update for next position (simulate auto-regressive generation)
            inputs = torch.cat([template_tokens.input_ids, 
                               torch.tensor([[token_id]]).to(model.device)], dim=1)
            with torch.no_grad():
                outputs = model(inputs)
            next_token_logits = outputs.logits[0, -1, :]
        
        # Store score for this choice
        choice_scores.append((label.lower(), score))
    
    # Return the choice with highest score
    if choice_scores:
        return max(choice_scores, key=lambda x: x[1])[0]
    return None

def parse_gsm8k_output(text: str):
    """
    Parses the model output for GSM8K using the extract_numeric_answer function.
    Returns (rationale, answer) or (None, None).
    """
    # Try to extract the numeric answer
    answer = extract_numeric_answer(text)
    if answer is None:
        print(f"Warning: Could not parse GSM8K output, no numeric answer found: {text[:100]}...")
        return None, None
    
    # Find the start of the generated part (usually after Q: ... A:)
    a_marker = text.find('\nA:')
    if a_marker != -1:
        rationale_start_index = a_marker + len('\nA:')
        rationale = text[rationale_start_index:].strip()
    else:
        rationale = text.strip()

    return rationale, answer


def parse_arithmetic_output(text: str):
    """
    Parses the model output for Arithmetic using extract_numeric_answer.
    Expected format is still <scratch>...</scratch>\nFINAL_ANSWER but we'll use more robust extraction.
    Returns (scratchpad_content, final_answer) or (None, None).
    """
    # Extract the scratchpad content
    scratch_match = re.search(r"<scratch>(.*?)</scratch>", text, re.DOTALL)
    scratchpad = None
    if scratch_match:
        scratchpad = scratch_match.group(1).strip()
    
    # Extract the numeric answer using the robust function
    answer = extract_numeric_answer(text)
    
    if answer is None:
        print(f"Warning: Could not extract numeric answer from Arithmetic output: {text[:100]}...")
        return scratchpad, None  # Return scratchpad even if answer parsing failed
    
    return scratchpad, answer


def format_question(question_data, dataset_type):
    """ Formats the question based on dataset type for the prompt """
    if dataset_type == 'cqa':
        q = question_data['question']
        choices = "\n".join([f"({label}) {text}" for label, text in zip(question_data['choices']['label'], question_data['choices']['text'])])
        return f"Q: {q}\nAnswer Choices:\n{choices}\nA:"
    elif dataset_type == 'arithmetic':
        # For generation, we only give the input part.
        return f"Input:\n{question_data['question']}\nTarget:" # Model should generate scratchpad + answer
    elif dataset_type == 'gsm8k':
        return f"Q: {question_data['question']}\nA:"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def generate_rationale(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    few_shot_prompt: str,
    question_data: dict,
    dataset_type: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,  # Default to greedy if not specified
    top_p: float = 0.9,        # Add these parameters
    top_k: int = 0,
    do_sample: bool = True
):
    """
    Generates rationale and answer for a given question using few-shot prompting.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        few_shot_prompt: The few-shot prompt string (examples).
        question_data: Dictionary containing the current question's data.
        dataset_type: 'cqa', 'arithmetic', or 'gsm8k'.
        max_new_tokens: Max tokens to generate.
        temperature: Generation temperature.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        do_sample: Whether to use sampling vs greedy decoding.

    Returns:
        Tuple (rationale, answer) or (None, None) if generation/parsing fails.
    """
    formatted_question = format_question(question_data, dataset_type)
    full_prompt = few_shot_prompt + "\n\n" + formatted_question

    # Inputs will be placed on the correct device by model.generate
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - max_new_tokens)
    # Explicitly move inputs to the model's device to avoid warning
    inputs = inputs.to(model.device)

    # Configure generation parameters
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    # Overrides for greedy decoding
    if temperature == 0.0:
        generate_kwargs["do_sample"] = False

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

        # Decode only the generated part
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # For debugging, print first few examples
        if question_data.get('id', '') in ['00621703', '00621704', '00621705']:  # First few examples
            print(f"\nGenerated text for question {question_data.get('id', 'unknown')}:")
            print(generated_text[:300] + "..." if len(generated_text) > 300 else generated_text)

        # Construct the parse input - we need to handle the whole output format
        parse_input = formatted_question + generated_text

    except Exception as e:
        print(f"Error during generation: {e}")
        return None, None

    # Parse the output based on dataset type
    if dataset_type == 'cqa':
        # For CQA, use log-likelihood scoring instead of text parsing
        answer_letter = score_cqa_answers(model, tokenizer, question_data, generated_text)
        return generated_text, answer_letter  # Return the rationale and the answer letter
    elif dataset_type == 'arithmetic':
        # Arithmetic needs special handling as scratchpad IS the rationale
        # Check for <scratch> tag first
        scratch_match = re.search(r"<scratch>", generated_text, re.DOTALL)
        if scratch_match:
            # Parse using the updated function with extract_numeric_answer
            parse_input_arith = generated_text[scratch_match.start():]
            return parse_arithmetic_output(parse_input_arith)
        else:
            # If no <scratch> tag found, try to extract just a numeric answer
            answer = extract_numeric_answer(generated_text)
            if answer:
                # No scratchpad but we have an answer
                print(f"Warning: Arithmetic output missing <scratch> tag but found answer {answer}")
                return generated_text, answer
            else:
                print(f"Warning: Arithmetic output missing <scratch> tag and no answer found: {generated_text}")
                return None, None
    elif dataset_type == 'gsm8k':
        return parse_gsm8k_output(parse_input)
    else:
        return None, None


def rationalize(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    few_shot_prompt: str,
    question_data: dict,
    correct_answer: str,
    dataset_type: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,  # Default to greedy if not specified
    top_p: float = 0.9,        # Add these parameters
    top_k: int = 0,
    do_sample: bool = True
):
    """
    Generates a rationale given the question and the correct answer (hint).

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        few_shot_prompt: Few-shot prompt.
        question_data: Dictionary containing the current question's data.
        correct_answer: The ground truth answer to provide as a hint.
        dataset_type: 'cqa', 'arithmetic', or 'gsm8k'.
        max_new_tokens: Max tokens to generate.
        temperature: Generation temperature.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        do_sample: Whether to use sampling vs greedy decoding.

    Returns:
        Tuple (rationale, answer) or (None, None) if generation/parsing fails.
        The returned answer should match the correct_answer hint.
    """

    # Format the question part first
    formatted_question_base = format_question(question_data, dataset_type)

    # Inject the hint based on dataset type (inspired by Fig 2 for CQA)
    if dataset_type == 'cqa':
        # Find the correct choice text based on the answer label (e.g., 'b')
        correct_choice_text = ""
        for label, text in zip(question_data['choices']['label'], question_data['choices']['text']):
            if label.lower() == correct_answer.lower():
                correct_choice_text = text
                break
        # Inject hint like in Fig 2. Modify the end of formatted_question
        if correct_choice_text and formatted_question_base.endswith("\nA:"):
            hint = f" (Hint: The answer is {correct_choice_text} ({correct_answer}))"
            # Insert hint before the final 'A:'
            prompt_with_hint = formatted_question_base[:-len("\nA:")] + hint + "\nA:"
        else:
            print("Warning: Could not format CQA hint properly.")
            prompt_with_hint = formatted_question_base # Fallback to no hint
    elif dataset_type == 'arithmetic':
        # For arithmetic, the hint is providing the answer after "Target:"
        if formatted_question_base.endswith("\nTarget:"):
            # Provide answer directly + prompt to start scratchpad
            prompt_with_hint = formatted_question_base + f" {correct_answer}\n<scratch>" 
            # The prompt becomes "Input: ... Target: ANSWER\n<scratch>" -> model generates the rest of "<scratch>...</scratch>\nANSWER"
        else:
            prompt_with_hint = formatted_question_base # Fallback
    elif dataset_type == 'gsm8k':
        # Similar to CQA, add hint before the 'A:'
        if formatted_question_base.endswith("\nA:"):
            hint = f" (Hint: The final answer is {correct_answer})"
            prompt_with_hint = formatted_question_base[:-len("\nA:")] + hint + "\nA:"
        else:
            prompt_with_hint = formatted_question_base
    else:
        prompt_with_hint = formatted_question_base # Fallback

    full_prompt = few_shot_prompt + "\n\n" + prompt_with_hint

    # Inputs will be placed on the correct device by model.generate
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - max_new_tokens)
    # Explicitly move inputs to the model's device to avoid warning
    inputs = inputs.to(model.device)

    # Configure generation parameters
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    # Overrides for greedy decoding
    if temperature == 0.0:
        generate_kwargs["do_sample"] = False

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # For debugging, print first few examples of rationalization
        if question_data.get('id', '') in ['00621703', '00621704', '00621705']:
            print(f"\nRationalized text for question {question_data.get('id', 'unknown')}:")
            print(generated_text[:300] + "..." if len(generated_text) > 300 else generated_text)

    except Exception as e:
        print(f"Error during rationalization generation: {e}")
        return None, None

    # For CQA, just return the generated text and the correct answer since we already know it
    if dataset_type == 'cqa':
        return generated_text, correct_answer.lower()
    # For arithmetic, extract the scratch tag and use extract_numeric_answer
    elif dataset_type == 'arithmetic':
        scratch_match = re.search(r"<scratch>", generated_text, re.DOTALL)
        if scratch_match:
            # Parse using the updated function with extract_numeric_answer
            parse_input_arith = generated_text[scratch_match.start():]
            return parse_arithmetic_output(parse_input_arith)
        else:
            # If no scratch tag, try to extract just the answer
            answer = extract_numeric_answer(generated_text)
            if answer:
                print(f"Warning: Rationalized Arithmetic output missing <scratch> tag but found answer {answer}")
                return generated_text, answer
            else:
                print(f"Warning: Rationalized Arithmetic output missing <scratch> tag and no answer found")
                return None, None
    # For GSM8K, use the extract_numeric_answer function
    elif dataset_type == 'gsm8k':
        answer = extract_numeric_answer(generated_text)
        if answer is not None:
            return generated_text, answer
        else:
            print(f"Warning: Could not extract numeric answer from GSM8K rationalization")
            return None, None
    else:
        return None, None 