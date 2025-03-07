# config.py

# Constants and hyperparameters
MIN_NEW_TOKENS = 1
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
SEED = 42
TOP_P = 0.9
TOP_K = 0
DO_SAMPLE = True
NUM_RETURN_SEQUENCES = 1
SELF_CONSISTENCY_PATHS = 20

# Dataset configurations
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
        "split": "high_school_mathematics",
        "subset": None,
        "question_key": "question",
        "answer_key": "answer",
        "is_mmlu": True
    },
    # "drop": {
    #     "name": "drop",
    #     "split": None,
    #     "subset": "validation",
    #     "question_key": "question",
    #     "answer_key": "answers_spans"
    # },
    "agieval": {
        "name": "cais/agieval",
        "split": "test",
        "subset": None,
        "question_key": "question",
        "answer_key": "answer"
    },
}

# Prompt templates
PROMPT_TEMPLATES = {
    # Simple question prompt
    "simple": {
        "numeric": """Question: {question} \n\nAnswer the question directly. Do not return any preamble, explanation, or reasoning.\n\nAnswer: """,
        "multiple_choice": """Question: {question} \n\nOptions:\n{options}\n\nAnswer the question directly. Do not return any preamble, explanation, or reasoning.\n\nAnswer: """
    },
    # Large Language Models are Zero-Shot Reasoners: https://arxiv.org/abs/2205.11916
    "cot": {
        "numeric": """Think step by step to answer the following question. Return the answer at the end of the response after a separator ####.\n\nQuestion: {question}\n\n""",
        "multiple_choice": """Think step by step to answer the following question. Return the answer at the end of the response after a separator ####.\n\nQuestion: {question}\n\nOptions:\n{options}\n\n"""
    },
    # Chain of Draft: Thinking Faster by Writing Less: https://arxiv.org/abs/2502.18600
    "draftCot": {
        "numeric": """Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator ####.\n\nQuestion: {question}\n\n""",
        "multiple_choice": """Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator ####.\n\nQuestion: {question} \n\nOptions:\n{options}\n\n"""
    },
    # Role-Setting Prompt: https://aclanthology.org/2024.naacl-long.228/
    "role": {
        "numeric": """From now on, you are an excellent math professor. One of your students asks you the following question.\nYou explain the solution step by step and then return the answer at the end of the response after a separator ####.\n\nQuestion: {question}\n\n""",
        "multiple_choice": """From now on, you are an excellent math professor. One of your students asks you the following question.\nYou explain the solution step by step and then return the answer at the end of the response after a separator ####.\n\nQuestion: {question} \n\nOptions:\n{options}\n\n"""
    },
    # Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models: https://arxiv.org/abs/2305.04091
    "plan": {
        "numeric": """Question: {question} \n\nLet's first understand the question, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate intermediate variables, solve the problem step by step and return the answer at the end of the response after a separator ####.\n\n""",
        "multiple_choice": """Question: {question}

Options:
{options}

Let's approach this systematically:
1. First, let's understand the question
2. Then, analyze each option carefully
3. Finally, choose the highest probability answer.

"""
    }
}
