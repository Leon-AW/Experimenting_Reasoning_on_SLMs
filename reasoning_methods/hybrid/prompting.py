import torch
import re
import sys
import os
from transformers import PreTrainedModel, PreTrainedTokenizer

# Define config values directly in this file
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.5
TOP_P = 0.9
TOP_K = 0
DO_SAMPLE = True
SEED = 42

# Helper function to find the start of the next question in the generated text
def find_next_question_marker(text):
    markers = ["\nQ:", "\nInput:"]
    positions = [text.find(marker) for marker in markers if text.find(marker) != -1]
    return min(positions) if positions else -1

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

def parse_rationale_only(text: str, dataset_type: str) -> str | None:
    """
    Parses the model output to extract only the rationale part,
    stripping away any leading prompt remnants or trailing answer extractions.
    Used specifically for cleaning the output of the 'rationalize' function.
    """
    rationale = text.strip() # Start with the full generated text

    # Attempt to remove the prompt part if the model included it
    # This depends on the prompt structure used in 'rationalize'
    prompt_markers = {
        'cqa': "\nRationale:",
        'gsm8k': "\nRationale:",
        'arithmetic': "\nRationale:",
    }
    marker = prompt_markers.get(dataset_type)
    if marker:
        marker_pos = rationale.find(marker)
        if marker_pos != -1:
            # Take text after the marker
            rationale = rationale[marker_pos + len(marker):].strip()
        else:
            # If marker not found, assume generation started directly with rationale
            # Check if it starts with typical rationale phrases vs prompt text
            if rationale.startswith("Q:") or rationale.startswith("Input:"):
                 # Looks like prompt wasn't stripped, try removing first line? Risky.
                 pass # Keep as is for now, maybe log a warning if problematic

    # Remove common answer extraction phrases if they appear at the end
    # Be less aggressive than before, target specific phrases from format_for_finetuning
    rationale = re.sub(r'\s*Therefore, the answer is.*?\(.*?\)(\.\s*)?$', '', rationale, flags=re.IGNORECASE | re.DOTALL).strip() # CQA format
    rationale = re.sub(r'\s*The final answer is:.*?$', '', rationale, flags=re.IGNORECASE | re.DOTALL).strip() # GSM8K format
    # For arithmetic, remove potential scratchpad tags or just the final number
    rationale = re.sub(r'\n<scratch>.*?</scratch>\n\d+(\.\d+)?$', '', rationale, flags=re.DOTALL).strip()
    rationale = re.sub(r'\n\d+(\.\d+)?$', '', rationale).strip() # If only number at end
    rationale = re.sub(r'^<scratch>\s*</scratch>$', '', rationale, flags=re.DOTALL).strip() # Empty scratchpad

    # Return None if rationale is empty after stripping
    return rationale if rationale else None

def generate_rationale(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    few_shot_prompt: str,
    question_data: dict,
    dataset_type: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    do_sample: bool = DO_SAMPLE
):
    """
    Generates rationale and attempts to parse the answer for a given question.
    (Signature kept similar to original for compatibility in main.py)

    Returns:
        Tuple (rationale, answer) or (None, None) if generation/parsing fails.
    """
    full_prompt = few_shot_prompt + "\n\n" + format_question(question_data, dataset_type)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Adjust sampling parameters based on temperature
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id, # Use EOS for padding in open-ended generation
        "eos_token_id": tokenizer.eos_token_id,
    }
    current_do_sample = do_sample
    if temperature > 0.0 and current_do_sample:
        gen_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": True,
        })
    else:
        # Use greedy decoding if temp is 0 or do_sample is False
        gen_kwargs["do_sample"] = False
        # Optionally set num_beams for greedy/beam search, but simple greedy is default
        # gen_kwargs["num_beams"] = 1

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated part
    # Handle potential warning about exceeding max length by truncating the input_ids passed to decode
    output_token_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()


    # --- Answer Parsing ---
    answer = None
    rationale = generated_text # Use the full generated text as potential rationale initially

    if dataset_type == 'cqa':
        # For CQA, use log-likelihood scoring based on the generated rationale text
        # Pass the *full* generated text to score_cqa_answers, as it includes rationale + answer phrase context
        # Let score_cqa_answers handle building the appropriate prompt for scoring
        answer = score_cqa_answers(model, tokenizer, question_data, generated_text)

        # The 'rationale' returned should ideally be just the reasoning steps.
        # Attempt to parse out the reasoning part before the final answer phrase.
        parsed_rationale = parse_rationale_only(generated_text, dataset_type)
        if parsed_rationale:
             rationale = parsed_rationale
        # If parsing fails, keep the full generated_text as rationale (better than nothing)


    elif dataset_type == 'gsm8k':
        rationale, answer = parse_gsm8k_output(generated_text) # This function extracts both
        if rationale is None: rationale = generated_text # Fallback if parsing fails badly

    elif dataset_type == 'arithmetic':
        rationale, answer = parse_arithmetic_output(generated_text) # Assumes scratchpad is rationale
        if rationale is None: rationale = generated_text # Fallback

    else:
        print(f"Warning: Unknown dataset_type '{dataset_type}' for parsing.")
        # Fallback: return raw generation as rationale, no answer parsed
        return generated_text, None

    # Ensure rationale is not None before returning
    if rationale is None:
        rationale = "" # Return empty string instead of None if parsing removed everything

    return rationale, answer


def rationalize(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question_data: dict,
    correct_answer: str,
    dataset_type: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    do_sample: bool = DO_SAMPLE
) -> str | None:
    """
    Generates a rationale (rationalization) given the question AND the correct answer.
    This function focuses on generating the reasoning steps only.

    Returns:
        The generated rationale string, or None if generation fails or yields empty rationale.
    """
    # Format the question part of the prompt
    formatted_question_part = format_question(question_data, dataset_type)
    # Remove the trailing 'A:' or 'Target:' added by format_question, as we'll add our own prompt ending
    if formatted_question_part.endswith("A:"):
        formatted_question_part = formatted_question_part[:-2].strip()
    elif formatted_question_part.endswith("Target:"):
         formatted_question_part = formatted_question_part[:-7].strip()


    # Construct the specific rationalization prompt asking for reasoning
    rationalization_prompt_suffix = ""
    if dataset_type == 'cqa':
        # Find the text corresponding to the correct answer label
        correct_answer_text = ""
        for label, text in zip(question_data['choices']['label'], question_data['choices']['text']):
            if str(label).lower() == str(correct_answer).lower():
                correct_answer_text = text
                break
        if not correct_answer_text:
             print(f"Warning: Could not find text for correct answer {correct_answer} in CQA.")
             return None

        rationalization_prompt_suffix = (
            f"{formatted_question_part.strip()}\n"
            f"Given that the correct answer is {correct_answer_text} ({correct_answer}), provide a concise, coherent, and step-by-step explanation that logically leads to this answer. "
            f"List your key reasoning steps in clear, numbered points, and avoid any repetition or irrelevant details.\n"
            f"Rationale:"
        )

    elif dataset_type in ['gsm8k', 'arithmetic']:
        rationalization_prompt_suffix = (
            f"{formatted_question_part.strip()}\n"
            f"Given that the correct answer is {correct_answer}, provide a concise and clear step-by-step explanation in 3-5 steps that leads to this answer. "
            f"Ensure the explanation is well-structured and does not contain any repetitive content.\n"
            f"Rationale:"
        )
    else:
        raise ValueError(f"Rationalization not implemented for dataset_type: {dataset_type}")

    # Combine few-shot examples, the formatted question, and the rationalization instruction
    full_prompt = formatted_question_part + rationalization_prompt_suffix
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Adjust sampling parameters (same logic as generate_rationale)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id, # Stop generation naturally or when max tokens hit
    }
    current_do_sample = do_sample
    if temperature > 0.0 and current_do_sample:
        gen_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": True,
        })
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated part (the rationalization)
    output_token_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()


    # Parse/clean the generated text to get just the rationale using the new helper
    parsed_rationale = parse_rationale_only(generated_text, dataset_type)

    # We don't need to parse an 'answer' here, just return the reasoning
    # The return type is str | None (parsed_rationale can be None if empty after cleaning)
    return parsed_rationale

# Ensure parse_gsm8k_output and parse_arithmetic_output are defined above or imported
# Ensure score_cqa_answers is defined above or imported 