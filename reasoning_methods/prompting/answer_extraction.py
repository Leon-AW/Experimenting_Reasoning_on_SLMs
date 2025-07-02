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


def extract_drop_answer(generated_text):
    """
    Extract answers from generated text for the DROP dataset.
    This function handles both numeric answers and string-based answers.
    
    It first tries to extract a numeric answer using the existing extract_numeric_answer function.
    If that fails, it attempts to extract a string answer using various patterns.
    """
    
    # Pattern to find answers in \boxed{} or \fbox{}
    box_pattern = re.compile(r"\\boxed{([^}]+)}|\\fbox{([^}]+)}")
    box_match = box_pattern.search(generated_text)
    if box_match:
        # Extract from the first non-empty group
        answer = next((g for g in box_match.groups() if g is not None), None)
        if answer:
            return answer.strip().rstrip('.,:;!?').strip()

    # Pattern 1: Look for "The final answer is:" followed by text
    # Extract the entire line after the pattern
    final_answer_pattern = re.compile(
        r"(?:The final answer is|Answer(?:\s+is)?|[Ss]olution:):?\s*(.*?)(?:\n|$)",        
        re.IGNORECASE
    )
    m = final_answer_pattern.search(generated_text)
    if m:
        answer = m.group(1).strip()
        # Clean up the answer (remove quotes, extra spaces, and trailing punctuation)
        answer = answer.strip('"\'').rstrip('.,:;!?').strip()
        if answer:
            return answer
    
    # If no specific pattern is found, try taking the first line, if it's not numeric.
    first_line = generated_text.split('\\n')[0].strip()
    if first_line and not first_line.replace('.','',1).isdigit():
        return first_line.rstrip('.,:;!?').strip()

    # Try to extract a numeric answer as a fallback
    numeric_answer = extract_numeric_answer(generated_text)
    if numeric_answer:
        return numeric_answer
    
    # If all else fails, return None
    return None
