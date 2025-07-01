# Prompting Methods for Reasoning in SLMs

This directory contains a study project for Humboldt University of Berlin, focused on experimenting with and evaluating various zero-shot prompting strategies to enhance reasoning in small language models (SLMs).

## Overview

The experiment evaluates several Llama 3.2 models, including base and fine-tuned variants, on a suite of mathematical and reasoning datasets. It systematically tests different prompting strategies to measure their impact on model performance.

The project is designed to be highly modular and configurable, allowing for easy extension and experimentation with new models, datasets, and prompting techniques.

## Project Structure

- `main.py`: The main entry point for running experiments.
- `config.py`: Contains all configurations for datasets, prompt templates, and hyperparameters.
- `answer_extraction.py`: Logic for parsing and extracting final answers from model outputs.
- `dataset_utils.py`: Utilities for loading datasets and configuring hardware.
- `numeric_processor.py` & `multiple_choice_processor.py`: Core logic for handling numeric and multiple-choice tasks.
- `generate_results_table.py`: A utility script to aggregate results from multiple experiments into a summary table.
- `results/`: Directory for storing final accuracy summaries.
- `debug_csvs/`: Directory for storing detailed, row-by-row CSV outputs for debugging.

## Supported Datasets

- GSM8K
- RACE
- ARC (AI2 Reasoning Challenge)
- MMLU (High School Mathematics)
- DROP
- CommonsenseQA

## Requirements

 ```bash
   #!/bin/bash

   # Create and activate environment
   conda create --name study_project_env python=3.10 -y
   conda activate study_project_env

   # Install requirements
   pip install -r requirements.txt
   ```

Required packages include:
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
tqdm>=4.65.0
sentencepiece
protobuf
accelerate>=0.26.0
```

## How to Run

The main script `main.py` is the primary entry point. It can be run with various flags to customize the experiment.

### Sweep Mode (Recommended)
The script includes a powerful "sweep" mode. **Running the script without any arguments will trigger a full evaluation sweep** across all configured models, datasets, prompt templates, and self-consistency settings. This is the easiest way to generate comprehensive results.

```bash
# Run a full evaluation sweep
python -m reasoning_methods.prompting.main
```

### Basic Usage
To run a single experiment, you need to specify a dataset and a model.
```bash
# Evaluate GSM8K with the 1B parameter model using all prompt templates
python -m reasoning_methods.prompting.main --dataset gsm8k --model_size 1b
```

### Command-Line Arguments

- `--dataset`: The dataset to evaluate.
  - Options: `['gsm8k', 'race', 'arc', 'mmlu', 'drop', 'commonsense_qa']`
- `--model_size`: The Llama 3.2 model to use.
  - Options: `['1b', '3b', '1b-instruct', '1b-sft-full', '1b-sft-lora', 'star_model1', ...]` (see `main.py` for the full list).
- `--template`: A specific prompt template to run (optional). If omitted, all templates are used.
  - Options: `['simple', 'cot', 'role', 'plan']`
- `--self_consistency`: Enable self-consistency with multiple generation paths (optional).
- `--debug`: Enable debug mode for detailed console output (optional).

### Example Commands

```bash
# Run a full evaluation sweep (recommended for comprehensive results)
python -m reasoning_methods.prompting.main

# Evaluate CommonsenseQA with the 3B parameter model
python -m reasoning_methods.prompting.main --dataset commonsense_qa --model_size 3b

# Evaluate GSM8K with a fine-tuned model using only the 'cot' template
python -m reasoning_methods.prompting.main --dataset gsm8k --model_size 1b-sft-full --template cot

# Evaluate ARC with the 1B model using self-consistency
python -m reasoning_methods.prompting.main --dataset arc --model_size 1b --self_consistency
```

## Prompting Strategies

The following prompt strategies are defined in `config.py`. The key used in the `--template` argument is shown in parentheses.

1. **Simple Prompting**: Direct question-answer format.
   - **Numeric Example**: 
     ```
     Problem: {question}
     Solution: 
     ```
   - **Multiple Choice Example**:
     ```
     Question: {question}

     Options:
     {options}
     ```

2. **Chain-of-Thought Prompting**: Step-by-step reasoning approach.
   - **Numeric Example**:
     ```
     Problem: {question}

     Solve and conclude your solution with 'The final answer is: <insert your answer here>'.

     Let's think step by step: 
     ```
   - **Multiple Choice Example**:
     ```
     Question: {question}

     Options:
     {options}

     Let's think step by step: 
     ```

3. **Role-Based Prompting**: Teacher-student interaction format.
   - **Numeric Example**:
     ```
     User: From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students.
     Assistant: That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Finally I will conclude it with 'The final answer is: <insert your answer here>'. Feel free to ask any math problems or questions you have, and I'll be glad to assist you.
     User: {question}
     Assistant: 
     ```
   - **Multiple Choice Example**:
     ```
     User: From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students.
     Assistant: That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Feel free to ask any math problems or questions you have, and I'll be glad to assist you.
     User: {question}
     Options:
     {options}
     Assistant: 
     ```

4. **Plan-and-Solve Prompting**: Structured planning and solving approach.
   - **Numeric Example**:
     ```
     Problem: {question}

     Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate intermediate results (pay attention to calculation and common sense), solve the problem step by step, and then conclude it with 'The final answer is: <insert your answer here>'.
     Let's begin: 
     ```
   - **Multiple Choice Example**:
     ```
     Question: {question}

     Options:
     {options}

     Let's approach this systematically:
     1. First, let's understand the question
     2. Then, analyze each option carefully
     3. Finally, choose the highest probability answer. 
     Let's begin: 
     ```

## Output

The script generates two types of output files:

1.  **Debug CSVs** (in `debug_csvs/`): `{dataset}_{template}_{model_size}{_sc_}_results.csv`
    - Contains detailed results for every sample, including the full prompt, generated text, and correctness.
    - `_sc` is appended if self-consistency is enabled.

2.  **Accuracy Summaries** (in `results/`): `{dataset}_{template}_{model_size}{_sc_}_total_accuracy.txt`
    - Contains the final accuracy score, total counts, and the hyperparameters used for the run.

   Example filenames:
   - `gsm8k_chain_1b_results.csv`
   - `race_simple_3b_total_accuracy.txt`
   - `gsm8k_chain_1b_sc20_results.csv` (self-consistency with 20 paths)
   - `gsm8k_role_1b_sc20_total_accuracy.txt` (self-consistency with 20 paths for role template)

## Debug Mode

When running with `--debug`, the script provides detailed output for each example in the console:
- Input question
- Generated model response (for numeric datasets) or predicted answer (for multiple-choice datasets)
- Extracted answer (for numeric datasets)
- Gold answer
- Accuracy status
- Progress updates for each batch

Debug mode also prints batch and overall accuracy summaries to the console during evaluation.

## Self-Consistency

The project supports self-consistency sampling, which generates multiple reasoning paths and selects the most common answer. This can improve performance on reasoning tasks, especially for Chain-of-Thought prompting.

When enabled with the `--self_consistency` flag, the system will:
- Generate multiple reasoning paths (default: 20 paths)
- Extract answers from each path
- Select the most frequent answer as the final prediction

## File Structure

```
.
├── reasoning_methods/
│   ├── __init__.py
│   └── prompting/
│       ├── __init__.py
│       ├── answer_extraction.py
│       ├── config.py
│       ├── dataset_utils.py
│       ├── main.py
│       ├── multiple_choice_processor.py
│       ├── numeric_processor.py
│       ├── process_dataset_batch.py
│       ├── prompt_helper.py
│       ├── generate_results_table.py
│       └── results/
│           ├── {dataset}_{template}_{model_size}{_sc[paths]}_total_accuracy.txt
│       └── debug_csvs/
│           ├── {dataset}_{template}_{model_size}{_sc[paths]}_results.csv
├── requirements.txt
└── README.md
```

## Notes

- Ensure sufficient GPU memory for model loading, especially for larger models and batch sizes. (See `dataset_utils.py` for hardware configuration and batch size details)
- Results are saved automatically in the `reasoning_methods/prompting/results/` directory.
- Debug mode might slow down execution but provides detailed insights into the evaluation process.
- Different datasets might require different evaluation metrics and prompt adjustments for optimal performance.
- Self-consistency can significantly increase computation time due to multiple inference paths.

