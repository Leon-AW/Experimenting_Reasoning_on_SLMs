# Experimenting Reasoning on SLMs

This is a study project for Humboldt University of Berlin focused on experimenting with reasoning capabilities in small language models (SLMs).

## Overview

The experiment evaluates LLaMA-3 models (1B and 3B) on various mathematical and reasoning datasets using different zero-shot prompting strategies:
- Simple prompting
- Chain-of-thought prompting
- Role-based prompting
- Plan-and-solve prompting

## Project Structure

- `models/reasoning_methods/prompting/`: Directory containing the main script and supporting files.
- `requirements.txt`: List of dependencies required to run the project.

## Supported Datasets

- GSM8K: Grade School Math problems
- RACE: Reading Comprehension
- ARC: AI2 Reasoning Challenge
- MMLU: Massive Multitask Language Understanding
- DROP: Discrete Reasoning Over Paragraphs
- AGIEVAL: A General Evaluation Benchmark for Large Language Models

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
     - Options: ['gsm8k', 'race', 'arc', 'mmlu', 'drop', 'agieval'] (See `config.py` startLine: 15 endLine: 59 for configuration details)
   - `--model_size`: Specify LLaMA-3 model size (default: '1b')
     - Options: ['1b', '3b'] (See `main.py` startLine: 34 endLine: 36)
   - `--debug`: Enable debug mode for detailed output (optional) (See `main.py` startLine: 32 endLine: 33 and `evaluator.py` startLine: 142 endLine: 148, startLine: 150 endLine: 151, startLine: 153 endLine: 155, startLine: 189 endLine: 190, startLine: 206 endLine: 207, startLine: 232 endLine: 234, startLine: 250 endLine: 251, startLine: 265 endLine: 272, startLine: 301 endLine: 303, startLine: 318 endLine: 320, startLine: 333 endLine: 339, startLine: 347 endLine: 348, startLine: 362 endLine: 371)
   - `--self_consistency`: Enable self-consistency with multiple paths (optional). Uses 20 paths by default. (See `main.py` startLine: 37 endLine: 39 and `config.py` startLine: 12 and `evaluator.py` startLine: 160 endLine: 273)

   ### Example Commands
   ```bash
   # Evaluate GSM8K with LLaMA-3B
   python -m reasoning_methods.prompting.main --dataset gsm8k --model_size 3b

   # Evaluate RACE with LLaMA-1B in debug mode
   python -m reasoning_methods.prompting.main --dataset race --model_size 1b --debug

   # Evaluate GSM8K with LLaMA-1B using self-consistency
   python -m reasoning_methods.prompting.main --dataset gsm8k --model_size 1b --self_consistency
   ```

## Output

The script generates two types of output files in the `results` directory:

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

## Prompting Strategies

The code implements four different prompting strategies, configurable in `config.py` (startLine: 62 endLine: 104):

1. **Simple**: Direct question-answer format. (See `config.py` startLine: 63 endLine: 67)
2. **Chain**: Step-by-step reasoning approach, based on "Large Language Models are Zero-Shot Reasoners". (See `config.py` startLine: 68 endLine: 74)
3. **Role**: Teacher-student interaction format, based on NAACL 2024 paper. (See `config.py` startLine: 75 endLine: 87)
4. **Plan**: Structured planning and solving approach, based on "Plan-and-Solve Prompting". (See `config.py` startLine: 88 endLine: 103)

## Self-Consistency

The project supports self-consistency sampling, which generates multiple reasoning paths and selects the most common answer. This can improve performance on reasoning tasks, especially for Chain-of-Thought prompting. (See `README.md` startLine: 129 endLine: 137 and `evaluator.py` startLine: 160 endLine: 273)

When enabled with the `--self_consistency` flag, the system will:
- Generate multiple reasoning paths (default: 20 paths, configurable in `config.py` startLine: 12)
- Extract answers from each path
- Select the most frequent answer as the final prediction

## File Structure

```
.
├── models/
│   └── reasoning_methods/
│       └── prompting/
│           ├── __init__.py
│           ├── answer_extraction.py
│           ├── config.py
│           ├── dataset_utils.py
│           ├── evaluator.py
│           ├── main.py
│           └── prompts.py
├── results/
│   ├── {dataset}_{template}_{model_size}{_sc[paths]}_results.csv
│   └── {dataset}_{template}_{model_size}{_sc[paths]}_total_accuracy.txt
├── requirements.txt
└── README.md
```

## Notes

- Ensure sufficient GPU memory for model loading, especially for larger models and batch sizes. (See `dataset_utils.py` startLine: 64 endLine: 92 for hardware configuration and batch size details)
- Results are saved automatically in the `results` directory.
- Debug mode might slow down execution but provides detailed insights into the evaluation process.
- Different datasets might require different evaluation metrics and prompt adjustments for optimal performance.
- Self-consistency can significantly increase computation time due to multiple inference paths.

## License

This project is licensed under the MIT License. (See `LICENSE`)

