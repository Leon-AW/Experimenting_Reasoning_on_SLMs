# Experimenting Reasoning on SLMs

This is a study project for Humboldt University of Berlin focused on experimenting with reasoning capabilities in small language models (SLMs).

## Overview

The experiment evaluates LLaMA models (1B and 3B) on various mathematical and reasoning datasets using different prompting strategies:
- Simple prompting
- Chain-of-thought prompting
- Role-based prompting
- Plan-and-solve prompting

## Project Structure

- `models/reasoning_methods/experiment_gsm8k_prompting.py`: Main script for running experiments.
- `requirements.txt`: List of dependencies required to run the project.

## Supported Datasets

- GSM8K: Grade School Math problems
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
torch
transformers
datasets
tqdm
```

## How to Run

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Experimenting_Reasoning_on_SLMs.git
   cd Experimenting_Reasoning_on_SLMs
   ```

2. **Install Dependencies:**

   Install the necessary Python packages:

   ```bash
   #!/bin/bash

   # Create and activate environment
   conda create --name study_project_env python=3.10 -y
   source activate study_project_env

   # Install requirements
   pip install -r requirements.txt
   ```

3. **Run the Experiment:**

   The main script `experiment_gsm8k_prompting.py` can be run with various configurations:

   ### Basic Usage
   ```bash
   python models/reasoning_methods/experiment_gsm8k_prompting.py --dataset gsm8k --model_size 3b
   ```

   ### Command Line Arguments
   - `--dataset`: Choose the dataset to evaluate (default: 'gsm8k')
     - Options: ['gsm8k', 'race', 'arc', 'mmlu', 'drop']
   - `--model_size`: Specify LLaMA model size (default: '3b')
     - Options: ['1b', '3b']
   - `--debug`: Enable debug mode for detailed output (optional)

   ### Example Commands
   ```bash
   # Evaluate GSM8K with LLaMA-3B
   python experiment_gsm8k_prompting.py --dataset gsm8k --model_size 3b

   # Evaluate RACE with LLaMA-1B in debug mode
   python experiment_gsm8k_prompting.py --dataset race --model_size 1b --debug

   # Evaluate ARC with LLaMA-3B
   python experiment_gsm8k_prompting.py --dataset arc --model_size 3b
   ```

## Output

The script generates two types of output files in the `results` directory:

1. CSV Results File: `{dataset}_{template}_{model_size}_results.csv`
   - Contains detailed results for each question
   - Includes: question, prompt, generated text, predicted answer, gold answer, and correctness

2. Accuracy Summary: `{dataset}_{template}_{model_size}_total_accuracy.txt`
   - Contains overall accuracy statistics
   - Shows total correct answers and accuracy percentage

## Debug Mode

When running with `--debug`, the script provides detailed output for each example:
- Input question
- Generated model response
- Extracted answer
- Correct answer
- Accuracy status
- Progress updates every 10 examples

## Prompting Strategies

The code implements four different prompting strategies:

1. **Simple**: Direct question-answer format
2. **Chain**: Step-by-step reasoning approach (based on "Large Language Models are Zero-Shot Reasoners")
3. **Role**: Teacher-student interaction format (based on NAACL 2024 paper)
4. **Plan**: Structured planning and solving approach (based on "Plan-and-Solve Prompting")

## File Structure

```
.
├── models/
│   └── reasoning_methods/
│       └── evaluate_llama_prompting.py
├── results/
│   ├── {dataset}_{template}_{model_size}_results.csv
│   └── {dataset}_{template}_{model_size}_total_accuracy.txt
├── requirements.txt
└── README.md
```

## Notes

- Ensure sufficient GPU memory for model loading
- Results are saved automatically in the `results` directory
- Debug mode might slow down execution but provides detailed insights
- Different datasets might require different evaluation metrics

## License

This project is licensed under the MIT License.

