import re
from collections import Counter


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
    Note: MMLU gold answers are provided as numbers (e.g. "0" means option A).
    """
    # Use only the part after the last "Answer:" and take just the first line to avoid extra explanation.
    if "Answer:" in generated_text:
        response = generated_text.split("Answer:")[-1].strip().splitlines()[0].strip()
    else:
        response = generated_text.strip().splitlines()[0].strip()
    
    # First try to find explicit numeric answers directly.
    numeric_patterns = [
        r"(?:[Tt]he (?:correct )?answer is[:\s]*)([0-9]+)\b",
        r"(?:Option|Choice)[:\s]*([0-9]+)\b",
        r"\b([0-9]+)\b\s*(?:is correct)?",
    ]
    for pattern in numeric_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Next try letter patterns and convert them to numbers (A->0, B->1, etc.)
    letter_patterns = [
        r"(?:[Tt]he (?:correct )?answer is[:\s]*)([A-D])\b",
        r"(?:Option|Choice)[:\s]*([A-D])\b",
        r"\b([A-D])\b\s*(?:is correct)?",
    ]
    for pattern in letter_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            letter_ans = match.group(1).upper()
            mapping = {"A": "0", "B": "1", "C": "2", "D": "3"}
            return mapping.get(letter_ans, None)
    
    # Fallback: if the response is exactly a single character or a digit.
    if response in ['A', 'B', 'C', 'D']:
        mapping = {"A": "0", "B": "1", "C": "2", "D": "3"}
        return mapping[response]
    if response.isdigit():
        return response
    
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
    if dataset_name in ["race", "arc", "agieval"]:
        return extract_multiple_choice_answer
    elif dataset_name == "gsm8k":
        return extract_gsm8k_answer
    elif dataset_name == "drop":
        return extract_drop_answer
    elif dataset_name == "mmlu":
        return extract_mmlu_answer
    else:
        return extract_gsm8k_answer  # Default to GSM8K extractor


def extract_numeric_answer(answer_text):
    """Extract the final numeric value from a GSM8K-style answer string."""
    # Look for the number after ####
    if '####' in answer_text:
        final_part = answer_text.split('####')[-1].strip()
        try:
            return float(final_part)
        except ValueError:
            pass
    
    # If no #### found, try to find the last number in the text
    numbers = re.findall(r'-?\d*\.?\d+', answer_text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
            
    return None
