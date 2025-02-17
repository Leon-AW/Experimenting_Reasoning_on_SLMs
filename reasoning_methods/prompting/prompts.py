from .config import PROMPT_TEMPLATES

# Function to get the appropriate prompt template based on dataset type
def get_prompt_template(template_name, dataset_name):
    """Returns the appropriate prompt template based on dataset type"""
    numeric_datasets = ["gsm8k", "drop"]
    template_type = "numeric" if dataset_name in numeric_datasets else "multiple_choice"
    return PROMPT_TEMPLATES[template_name][template_type]

# Function to format the prompt according to template and dataset type
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

# Function to extract the ending text from the prompt template
def get_prompt_ending(template_name, dataset_type):
    """Extract the ending text from the prompt template."""
    template_type = "numeric" if dataset_type in ["gsm8k", "drop"] else "multiple_choice"
    template = PROMPT_TEMPLATES[template_name][template_type]
    
    # Find the last segment after the last \n\n
    segments = template.split('\n\n')
    # Remove trailing whitespace from the ending
    return segments[-1].rstrip()
