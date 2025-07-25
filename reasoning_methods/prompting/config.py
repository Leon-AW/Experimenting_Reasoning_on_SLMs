# config.py

# Constants and hyperparameters
MIN_NEW_TOKENS = 1
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.5
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
    "drop": {
        "name": "drop",
        "split": None,
        "subset": "validation",
        "question_key": "question",
        "answer_key": "answers_spans"
    },
    "commonsense_qa": {
        "name": "commonsense_qa",
        "split": "default",
        "subset": "validation",
        "question_key": "question",
        "answer_key": "answerKey"
    }
}

# Prompt templates
PROMPT_TEMPLATES = {
    # Simple question prompt
    # "direct": {
    #     "numeric": """Problem: {question}\nAnswer the question directly. Do not return any preamble, explanation, or reasoning. Answer: """,
    #     "multiple_choice": """Question: {question}\n\nOptions:\n{options}\n\n"""
    # },
    "simple": {
        "numeric": """Problem: {question}\nSolution: """,
        "multiple_choice": """Question: {question}\n\nOptions:\n{options}\n\nAnswer:"""
    },
    # Large Language Models are Zero-Shot Reasoners: https://arxiv.org/abs/2205.11916
    "cot": {
        "numeric": """Problem: {question}\n\nSolve and conclude your solution with 'The final answer is: <insert your answer here>'.\n\nLet's think step by step: """,
        "multiple_choice": """Question: {question}\n\nOptions:\n{options}\n\nLet's think step by step: """
    },
    # Role-Setting Prompt: https://aclanthology.org/2024.naacl-long.228/
    "role": {
        "numeric": """User: From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students.
Assistant: That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Finally I will conclude it with 'The final answer is: <insert your answer here>'. Feel free to ask any math problems or questions you have, and I'll be glad to assist you.
User: {question}
Assistant: """,
        "multiple_choice": """User: From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students.
Assistant: That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Feel free to ask any math problems or questions you have, and I'll be glad to assist you.
User: {question}
Options:
{options}
Assistant: """
    },
    # Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models: https://arxiv.org/abs/2305.04091
    "plan": {
        "numeric": """Problem: {question}\n\nLet's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate intermediate results (pay attention to calculation and common sense), solve the problem step by step, and then conclude it with 'The final answer is: <insert your answer here>'.
        Let's begin: """,
        "multiple_choice": """Question: {question}\n\nOptions:\n{options}\n\nLet's approach this systematically:\n1. First, let's understand the question\n2. Then, analyze each option carefully\n3. Finally, choose the highest probability answer. 
        Let's begin: """
    }
}
