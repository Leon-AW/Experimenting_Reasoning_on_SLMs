# create_mixed_dataset.py
import random
from datasets import load_dataset, concatenate_datasets, DatasetDict


OUTPUT_DIR = "reasoning_methods/fine-tuning/mixed_finetuning_dataset"
TARGET_TOTAL_SAMPLES = 300000
VALIDATION_SPLIT_PERCENTAGE = 0.05

DATASET_CONFIG = [
    {
        "name": "Open-Orca/SlimOrca-Dedup",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.425), # 42.5%
        "formatting_func": "format_slimorca",
    },
    {
        "name": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.125), # 12.5%
        "formatting_func": "format_arc",
    },
    {
        "name": "commonsense_qa",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.125), # 12.5%
        "formatting_func": "format_commonsense_qa",
    },
    {
        "name": "gsm8k",
        "subset": "main",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.10), # 10%
        "formatting_func": "format_gsm8k",
    },
    {
        "name": "meta-math/MetaMathQA",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.125), # 12.5%
        "formatting_func": "format_metamathqa",
    },
     {
        "name": "squad",
        "split": "train",
        "target_samples": int(TARGET_TOTAL_SAMPLES * 0.10), # 10%
        "formatting_func": "format_squad",
    },
]

def format_slimorca(example):
    if 'conversations' in example and isinstance(example['conversations'], list):
        messages = []
        for msg in example['conversations']:
            role = msg.get('from')
            value = msg.get('value')
            if role == 'human':
                messages.append({'role': 'user', 'content': value})
            elif role == 'gpt':
                messages.append({'role': 'assistant', 'content': value})

        if messages and messages[0].get('role') == 'user' and messages[-1].get('role') == 'assistant':
            return {"messages": messages}
    return None

def format_arc(example):
    question = example['question']
    choices_text = "\\n".join([f"{label}. {text}" for label, text in zip(example['choices']['label'], example['choices']['text'])])
    prompt = f"{question}\\n\\nChoose the correct answer from the following options:\\n{choices_text}"
    answer = example['answerKey']
    
    try:
        answer_text = example['choices']['text'][example['choices']['label'].index(answer)]
        full_answer = f"{answer}. {answer_text}"
    except (ValueError, IndexError):
        full_answer = answer

    return {"messages": [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': full_answer}]}

def format_commonsense_qa(example):
    question = example['question']
    choices_text = "\\n".join([f"{label}. {text}" for label, text in zip(example['choices']['label'], example['choices']['text'])])
    prompt = f"{question}\\n\\nChoose the best answer from the following options:\\n{choices_text}"
    answer = example.get('answerKey', '')
    try:
        answer_text = example['choices']['text'][example['choices']['label'].index(answer)]
        full_answer = f"{answer}. {answer_text}"
    except (ValueError, IndexError):
        full_answer = answer
    if not answer: return None
    return {"messages": [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': full_answer}]}


def format_gsm8k(example):
    question = example['question']
    answer = example['answer']
    return {"messages": [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]}

def format_metamathqa(example):
    question = example['query']
    answer = example['response']
    return {"messages": [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]}

def format_squad(example):
    context = example['context']
    question = example['question']
    answer = example['answers']['text'][0] if example['answers']['text'] else "No answer found."
    prompt = f"Context:\\n{context}\\n\\nQuestion:\\n{question}\\n\\nAnswer based on the context."
    return {"messages": [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]}

FORMATTERS = {
    "format_slimorca": format_slimorca,
    "format_arc": format_arc,
    "format_commonsense_qa": format_commonsense_qa,
    "format_gsm8k": format_gsm8k,
    "format_metamathqa": format_metamathqa,
    "format_squad": format_squad,
}

all_formatted_samples = []

print("Starting download and formatting of datasets...")

for config in DATASET_CONFIG:
    print(f"Processing {config['name']} ({config.get('subset', 'default')})...")
    try:
        ds = load_dataset(config['name'], name=config.get('subset'), split=config['split'], streaming=False)

        num_available = len(ds)
        target_samples_for_ds = min(config['target_samples'], num_available)

        if num_available > target_samples_for_ds:
             indices = random.sample(range(num_available), target_samples_for_ds)
             ds = ds.select(indices)
        elif num_available < config['target_samples']:
             print(f"Warning: Could only load {num_available} of the desired {config['target_samples']} samples for {config['name']}.")


        formatting_func = FORMATTERS[config['formatting_func']]
        formatted_ds = ds.map(formatting_func, remove_columns=ds.column_names, num_proc=4)

        formatted_ds = formatted_ds.filter(lambda example: example is not None and example.get('messages') is not None)
        formatted_ds = formatted_ds.filter(lambda example: len(example['messages']) > 0 and example['messages'][-1]['role'] == 'assistant' and example['messages'][-1]['content'])


        print(f"-> Added {len(formatted_ds)} formatted samples.")
        all_formatted_samples.append(formatted_ds)

    except Exception as e:
        print(f"Error processing {config['name']}: {e}")

print("\nCombining all datasets...")
if not all_formatted_samples:
     raise ValueError("No datasets could be successfully loaded and formatted.")

mixed_dataset = concatenate_datasets(all_formatted_samples)

print(f"Shuffling the dataset ({len(mixed_dataset)} samples)...")
mixed_dataset = mixed_dataset.shuffle(seed=42)

print(f"Creating Train/Validation Split ({100-VALIDATION_SPLIT_PERCENTAGE*100}% / {VALIDATION_SPLIT_PERCENTAGE*100}%)...")
split_dataset = mixed_dataset.train_test_split(test_size=VALIDATION_SPLIT_PERCENTAGE, seed=42)

split_dataset["validation"] = split_dataset.pop("test")

print(f"Final training split: {len(split_dataset['train'])} samples")
print(f"Final validation split: {len(split_dataset['validation'])} samples")

print(f"Saving dataset to '{OUTPUT_DIR}'...")
split_dataset.save_to_disk(OUTPUT_DIR)

print("\nDone! The mixed dataset has been created and saved.")
print(f"You can now use it in the SFT script with `--dataset_name {OUTPUT_DIR}`.")
