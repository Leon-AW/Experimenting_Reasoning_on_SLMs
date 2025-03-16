# Experimenting Reasoning on SLMs

This is a study project for Humboldt University of Berlin focused on experimenting with reasoning capabilities in small language models (SLMs).

## Overview

The experiment evaluates Llama 3.2 models (1B and 3B) on various mathematical and reasoning datasets using different zero-shot prompting strategies:
- Simple prompting
- Chain-of-thought prompting
- Role-based prompting
- Plan-and-solve prompting

## Project Structure

- `reasoning_methods/prompting/`: Directory containing the main script and supporting files.
- `requirements.txt`: List of dependencies required to run the project.

## Supported Datasets

- GSM8K: Grade School Math problems Test-Set
- GSM8K_2: Grade School Math problems Train-Set
- RACE: Reading Comprehension
- ARC: AI2 Reasoning Challenge
- MMLU: Massive Multitask Language Understanding
- DROP: Discrete Reasoning Over Paragraphs

## Requirements

Ensure you have Python installed. You can install the required packages using:

```bash
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

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Experimenting_Reasoning_on_SLMs.git
   cd Experimenting_Reasoning_on_SLMs
   ```

2. **Install Dependencies:**

   Install the necessary Python packages using `conda` and `pip`:

   ```bash
   #!/bin/bash

   # Create and activate environment
   conda create --name study_project_env python=3.10 -y
   conda activate study_project_env

   # Install requirements
   pip install -r requirements.txt
   ```

3. **Run the Experiment:**

   The main script `main.py` located in `reasoning_methods/prompting/` can be run with various configurations. Use `python -m reasoning_methods.prompting.main` to execute the script.

   ### Basic Usage
   ```bash
   python -m reasoning_methods.prompting.main --dataset gsm8k --model_size 3b
   ```

   ### Command Line Arguments
   - `--dataset`: Choose the dataset to evaluate (default: 'gsm8k')
     - Options: ['gsm8k', 'race', 'arc', 'mmlu', 'drop', 'gsm8k_2'] (See `config.py` startLine: 15 endLine: 59 for configuration details)
   - `--model_size`: Specify Llama 3.2 model size (default: '1b')
     - Options: ['1b', '3b']
   - `--debug`: Enable debug mode for detailed output (optional)
   - `--self_consistency`: Enable self-consistency with multiple paths (optional). Uses 20 paths by default.

   ### Example Commands
   ```bash
   # Evaluate GSM8K with Llama 3.2B
   python -m reasoning_methods.prompting.main --dataset gsm8k --model_size 3b

   # Evaluate RACE with LLaMA-1B in debug mode
   python -m reasoning_methods.prompting.main --dataset race --model_size 1b --debug

   # Evaluate GSM8K with LLaMA-1B using self-consistency
   python -m reasoning_methods.prompting.main --dataset gsm8k --model_size 1b --self_consistency
   ```

## Prompting Strategies

The code implements four different prompting strategies, configurable in `config.py`:

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

The script generates two types of output files in the `reasoning_methods/prompting/results/` directory:

1. CSV Results File: `{dataset}_{template}_{model_size}{_sc[paths]}_results.csv`
   - Contains detailed results for each question
   - Includes: sample index, question, prompt, generated text, predicted answer, gold answer, and correctness
   - `_sc[paths]` is appended when self-consistency is enabled, indicating the number of paths used (e.g., `_sc20`).

2. Accuracy Summary: `{dataset}_{template}_{model_size}{_sc[paths]}_total_accuracy.txt`
   - Contains overall accuracy statistics
   - Shows total correct answers and accuracy percentage
   - Includes hyperparameters used for the experiment
   - `_sc[paths]` is appended when self-consistency is enabled, indicating the number of paths used (e.g., `_sc20`).

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

## License

This project is licensed under the MIT License. (See `LICENSE`)

