# config.py

# Constants and hyperparameters
MIN_NEW_TOKENS = 1
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
SEED = 42
TOP_P = 0.9
TOP_K = 0
DO_SAMPLE = True
NUM_RETURN_SEQUENCES = 1
SELF_CONSISTENCY_PATHS = 20

# Dataset configurations
DATASET_CONFIGS = {
    # "gsm8k": {
    #     "name": "gsm8k",
    #     "split": "main",
    #     "subset": "test",
    #     "question_key": "question",
    #     "answer_key": "answer"
    # },
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
        "numeric": """Problem: {question} \n\nSolve the problem, then conclude it with 'The final answer is: <insert your answer here>'. \n\nAnswer: """,
        "multiple_choice": """Question: {question} \n\nOptions:\n{options}\n\n"""
    },
    # Large Language Models are Zero-Shot Reasoners: https://arxiv.org/abs/2205.11916
    "chain": {
        "numeric": """Problem: {question} \n\nSolve the problem step-by-step, then conclude it with 'The final answer is: <insert your answer here>'. \n\nLet's think step by step: """,
        "multiple_choice": """Question: {question} \n\nOptions:\n{options}\n\n

        Let's solve this step-by-step: """
            },
            # Role-Setting Prompt: https://aclanthology.org/2024.naacl-long.228/
            "role": {
                "numeric": """From now on, you are an excellent math teacher. One of your students wants to ask you a question. \nYou explain it and conclude your answer with 'The final answer is: <insert your answer here>'.
        \n\nQuestion: {question} \n\nAnswer: """,
                "multiple_choice": """From now on, you are an excellen math teacher. One of your students wants to ask you a question. 

        Question: {question}

        Options:
        {options}

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
3. Finally, choose the highest probability answer.

"""
    }
}
