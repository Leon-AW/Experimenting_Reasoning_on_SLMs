import torch
import re
import sys
import os
from transformers import PreTrainedModel, PreTrainedTokenizer
import random # Add random for selecting starter phrases

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
    # Set seed for reproducibility in scoring
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    answer_choices = question_data['choices']['label']
    answer_texts = question_data['choices']['text']

    answer_choices_str = "\n".join([f"({label}) {text}" for label, text in zip(answer_choices, answer_texts)])
    
    # Create completion prompt template
    question = question_data['question']
    
    # Step 1: Build a prompt with the question, rationale, and answer template
    prompt_template = f"Q: {question}\n Answer Choices: {answer_choices_str}\n A: {rationale}\nTherefore, the correct answer is"
    
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

    # Remove any additional questions that might be from few-shot examples
    # Find and remove patterns like "Q: ... A:" that appear after the first section
    q_markers = ["\nQ:", "\nQuestion:", "\nInput:"]
    for q_marker in q_markers:
        q_pos = rationale.find(q_marker)
        if q_pos > 0:  # Only remove if not at the beginning (so we keep first Q: if that's how our rationale starts)
            rationale = rationale[:q_pos].strip()
    
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

def extract_cqa_explicit_answer(generated_text, choices):
    """
    Extract an explicit answer label (a, b, c, etc.) from generated text for CQA questions.
    If successful, returns the answer letter; otherwise returns None.
    
    Args:
        generated_text: The generated rationale text
        choices: The answer choices labels from the question data
    """
    # Convert choices to lowercase for case-insensitive matching
    valid_choices = [c.lower() for c in choices]
    
    # Look for common answer patterns
    patterns = [
        # Final answer: (X)
        r"[Ff]inal\s+answer:?\s*\(?([a-zA-Z])\)?\.?$",
        # Therefore, the answer is (X).
        r"[Tt]herefore,?\s+(?:the\s+)?answer\s+is\s+\(?([a-zA-Z])\)?\.?$",
        # Therefore, (X).
        r"[Tt]herefore,?\s+\(?([a-zA-Z])\)?\.?$",
        # The answer is (X).
        r"[Tt]he\s+answer\s+is\s+\(?([a-zA-Z])\)?\.?$",
        # Option (X) is correct.
        r"[Oo]ption\s+\(?([a-zA-Z])\)?\s+is\s+correct\.?$",
        # Answer: (X)
        r"[Aa]nswer:?\s+\(?([a-zA-Z])\)?\.?$",
        # So the answer is (X).
        r"[Ss]o\s+(?:the\s+)?answer\s+is\s+\(?([a-zA-Z])\)?\.?$",
        # Ending with a letter like "... community existed. Therefore, B."
        r"(?:Therefore|Thus|So|Hence),?\s*([a-zA-Z])\.?$",
        # Common verb + letter patterns like "...should choose B."
        r"\s+(?:choose|select|pick|is)\s+([a-zA-Z])\.?$",
        # Just a single letter at the end of text
        r"[\.\s]+([a-zA-Z])\.?$"
    ]
    
    # Search for each pattern
    for i, pattern in enumerate(patterns):
        matches = re.finditer(pattern, generated_text)
        # Get the last match (in case there are multiple, take the final one)
        last_match = None
        for match in matches:
            last_match = match
        
        if last_match:
            answer_letter = last_match.group(1).lower()
            # Verify it's a valid choice
            if answer_letter in valid_choices:
                return answer_letter
    
    # If no pattern matched or the extracted letter wasn't in valid choices
    return None

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
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    full_prompt = few_shot_prompt + "\n\n" + format_question(question_data, dataset_type)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Adjust sampling parameters based on temperature
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id, # Use EOS for padding in open-ended generation
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Only set sampling parameters if we're actually sampling
    if temperature > 0.0 and do_sample:
        gen_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k > 0 else None,
            "do_sample": True,
        })
    else:
        # Use greedy decoding if temp is 0 or do_sample is False
        gen_kwargs["do_sample"] = False
        # Don't set temperature, top_p, top_k when not sampling to avoid warnings

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated part
    # Handle potential warning about exceeding max length by truncating the input_ids passed to decode
    output_token_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()

    # Clean up the generated text to remove potential few-shot examples that were repeated
    # Use parse_rationale_only to help clean up the text
    cleaned_generated_text = parse_rationale_only(generated_text, dataset_type)
    if cleaned_generated_text:
        generated_text = cleaned_generated_text

    # --- Answer Parsing ---
    answer = None
    rationale = generated_text # Use the full generated text as potential rationale initially

    if dataset_type == 'cqa':
        # First try to extract explicit answer from the rationale
        explicit_answer = extract_cqa_explicit_answer(generated_text, question_data['choices']['label'])
        
        if explicit_answer:
            # Use explicitly stated answer
            answer = explicit_answer
        else:
            # Fall back to log-likelihood scoring if no explicit answer found
            # Create an empty populated_starter since we don't have one in generation phase
            empty_populated_starter = ""
            answer = score_cqa_answers(model, tokenizer, question_data, generated_text)

        # Parse rationale as before (unchanged)
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


# Path to the starter phrases file
# RATIONALIZATION_STARTER_PHRASES_PATH = os.path.join(os.path.dirname(__file__), 'rationalization_starters.txt') # Removed

# def load_rationalization_starters(): # Removed
#     """Loads rationalization starter phrases from the text file."""
#     if not os.path.exists(RATIONALIZATION_STARTER_PHRASES_PATH):
#         print(f"Warning: Rationalization starter phrases file not found at {RATIONALIZATION_STARTER_PHRASES_PATH}")
#         return ["The correct answer is {answer_placeholder} because"] # Fallback
#     with open(RATIONALIZATION_STARTER_PHRASES_PATH, 'r') as f:
#         starters = [line.strip() for line in f if line.strip()]
#     if not starters:
#         print(f"Warning: No starter phrases found in {RATIONALIZATION_STARTER_PHRASES_PATH}")
#         return ["The correct answer is {answer_placeholder} because"] # Fallback
#     return starters

# RATIONALIZATION_STARTERS = load_rationalization_starters() # Removed


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
    do_sample: bool = DO_SAMPLE,
    few_shot_prompt: str = None,  # Add parameter for few-shot examples
    debug: bool = False,
    attempt_number: int = 0  # Add attempt_number parameter
) -> str | None:
    """
    Generates a rationale (rationalization) given the question AND the correct answer.
    This function focuses on generating the reasoning steps only.

    Returns:
        The generated rationale string, or None if generation fails or yields empty rationale.
    """
    # Set seed for reproducibility, but vary by attempt number to get different samples
    attempt_seed = SEED + attempt_number * 1234  # Use much larger increments
    torch.manual_seed(attempt_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(attempt_seed)
    
    # Format the question part of the prompt
    formatted_question_part = format_question(question_data, dataset_type)
    # Remove the trailing 'A:' or 'Target:' added by format_question, as we'll add our own prompt ending
    if formatted_question_part.endswith("A:"):
        formatted_question_part = formatted_question_part[:-2].strip()
    elif formatted_question_part.endswith("Target:"):
         formatted_question_part = formatted_question_part[:-7].strip()

    # Construct the specific rationalization prompt asking for reasoning
    # rationalization_prompt_suffix = "" # Old approach
    
    # Select a random starter phrase
    # starter_phrase_template = random.choice(RATIONALIZATION_STARTERS) # Removed


    if dataset_type == 'cqa':
        # Find the text corresponding to the correct answer label
        correct_answer_text = ""
        placeholder_value = correct_answer # Default to the letter
        
        # Create a modified answer choices string that clearly marks the correct answer
        answer_choices = question_data['choices']['label']
        answer_texts = question_data['choices']['text']
        
        # Create a modified answer choices presentation that clearly marks the correct answer
        modified_choices = []
        for label, text in zip(answer_choices, answer_texts):
            if str(label).lower() == str(correct_answer).lower():
                # Mark this as the correct answer
                modified_choices.append(f"({label}) {text} (Correct Answer)")
                correct_answer_text = text
                placeholder_value = f"{text} ({correct_answer})" # e.g., "apple (a)"
            else:
                modified_choices.append(f"({label}) {text}")
        
        modified_choices_str = "\n".join(modified_choices)
        
        if not correct_answer_text:
             print(f"Warning: Could not find text for correct answer {correct_answer} in CQA.")
             # Fallback to just using the letter if text not found
             placeholder_value = correct_answer
        
        # Populate the chosen starter phrase
        # populated_starter = starter_phrase_template.format(answer_placeholder=placeholder_value) # Removed
        rationalization_prompt_suffix = (
            f"Q: {question_data['question']}\n"
            f"Answer Choices:\n{modified_choices_str}\n"
            f"A: " # Model starts generating rationale here
        )
    elif dataset_type in ['gsm8k', 'arithmetic']:
        # placeholder_value = correct_answer # Not needed anymore for starter
        # populated_starter = starter_phrase_template.format(answer_placeholder=placeholder_value) # Removed
        rationalization_prompt_suffix = (
            f"{formatted_question_part.strip()}\n"
            f"I must provide a step-by-step explanation that DEFINITELY leads to answer {correct_answer}.\n"
            f"A:" # Model starts generating rationale here
            # f"The correct answer is {correct_answer}.\n" # Old
            # f"I must provide a step-by-step explanation that DEFINITELY leads to answer {correct_answer}.\n" # Old
            # f"Rationale that leads to {correct_answer}:" # Old
        )
    else:
        raise ValueError(f"Rationalization not implemented for dataset_type: {dataset_type}")

    # For debugging, print the full prompt used for rationalization
    if debug:
        print(f"\n==== RATIONALIZATION PROMPT ====")
        print(rationalization_prompt_suffix)
        print("================================")

    # Combine few-shot examples, the formatted question, and the rationalization instruction
    if few_shot_prompt:
        # Add few-shot examples before the current question
        full_prompt = few_shot_prompt + "\n\n" + rationalization_prompt_suffix
    else:
        full_prompt = rationalization_prompt_suffix

    # # For debugging, print the COMPLETE input that goes to the model
    # if debug:
    #     print(f"\n==== COMPLETE MODEL INPUT FOR RATIONALIZATION ====")
    #     print(full_prompt)
    #     print("===================================================")

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Adjust sampling parameters (same logic as generate_rationale)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id, # Stop generation naturally or when max tokens hit
        "min_new_tokens": 10,  # Ensure model generates at least 10 tokens
    }
    
    # Only set sampling parameters if we're actually sampling
    if temperature > 0.0 and do_sample:
        gen_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k > 0 else None,
            "do_sample": True,
        })
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated part (the rationalization)
    output_token_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()
    
    # Clean up generated text - remove any "Q:" or similar markers that might appear
    if dataset_type == 'cqa':
        # Find and remove everything after "Q:" if it appears
        q_markers = ["Q:", "\nQ:", "Question:", "\nQuestion:", "Q :", " Q:"]
        for marker in q_markers:
            marker_pos = generated_text.find(marker)
            if marker_pos != -1:
                generated_text = generated_text[:marker_pos].strip()
                break  # Stop after finding the first marker


    # Parse/clean the generated text to get just the rationale using the new helper
    parsed_rationale = parse_rationale_only(generated_text, dataset_type)

    # Remove leading "A: " if present (model might generate it mimicking few-shot format)
    if parsed_rationale and parsed_rationale.startswith("A: "):
        parsed_rationale = parsed_rationale[3:].strip()

    # We don't need to parse an 'answer' here, just return the reasoning
    # The return type is str | None (parsed_rationale can be None if empty after cleaning)
    return parsed_rationale # Return only the parsed rationale

# Ensure parse_gsm8k_output and parse_arithmetic_output are defined above or imported
# Ensure score_cqa_answers is defined above or imported 