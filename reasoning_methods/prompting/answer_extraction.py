import re
from collections import Counter


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


def extract_numeric_answer(generated_text):
    """
    Extract numeric answer from generated text and round up at x.5 or higher, otherwise round down.
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

    # Look for "Answer:" or "Answer is:" followed by a number
    answer_patterns = [
        r"[Aa]nswer(?:\s+is)?:\s*=\s*(\d[\d,]*(?:\.\d+)?)[^a-zA-Z]*",  # Match number after equals sign
        r"[Aa]nswer(?:\s+is)?:\s*(\d[\d,]*(?:\.\d+)?)[^a-zA-Z]*",  # Match first number after Answer:
        r"[Aa]nswer(?:\s*[^=\n]*)?=\s*[^=\n]*?(\d[\d,]*(?:\.\d+)?)\s*$"  # Match final number in equation after Answer:
    ]
    
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, generated_text))  # Removed re.IGNORECASE since we handle case in pattern
        if matches:
            match = matches[-1]  # Take the last match
            clean_number = match.group(1).replace(',', '').replace('$', '')
            try:
                return str(int(float(clean_number) + 0.5))
            except ValueError:
                continue

    # If no calculation found, try other patterns
    other_patterns = [
        r"(?:answer|solution)(?:\s+is)?:?\s*\$?\s*(\d[\d,]*(?:\.\d+)?)",
        r"=\s*(\d[\d,]*(?:\.\d+)?)\s*$",
    ]
    
    for pattern in other_patterns:
        matches = list(re.finditer(pattern, generated_text, re.IGNORECASE))
        if matches:
            match = matches[-1]  # Take the last match
            clean_number = match.group(1).replace(',', '').replace('$', '')
            try:
                return str(int(float(clean_number) + 0.5))
            except ValueError:
                continue

    # Last resort: find the last number in the text
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", generated_text)
    if numbers:
        clean_number = numbers[-1].replace(',', '').replace('$', '')
        try:
            return str(int(float(clean_number) + 0.5))
        except ValueError:
            pass

    return None


def extract_gold_gsm8k_answer(answer_text):
    """
    Extracts the gold numeric answer from the GSM8K dataset.
    Expects the answer to be prefixed with the token '####'.
    """
    import re
    match = re.search(r'####\s*(-?\d+)', answer_text)
    if match:
        return str(int(match.group(1)))
    else:
        raise ValueError("No valid answer found.")
