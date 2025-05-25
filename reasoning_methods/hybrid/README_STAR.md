# STaR (Self-Taught Reasoner) Implementation

This directory contains a modular implementation of the STaR algorithm as described in the paper "STaR: Bootstrapping Reasoning With Reasoning" by Zelikman et al.

## Overview

The implementation is split into separate modules following the paper's algorithm exactly:

1. **Rationale Collection** (`collect_rationales.py`) - Steps 3-6 from the paper
2. **Fine-tuning** (`finetune.py`) - Step 7 from the paper  
3. **Main Orchestrator** (`star_main.py`) - Coordinates the full STaR process

## Key Features

- **Exact Paper Implementation**: Follows the STaR algorithm precisely
- **Modular Design**: Each phase can be run independently
- **Data Persistence**: Rationales are saved to files for inspection and reuse
- **Multiple Datasets**: Supports CommonsenseQA, GSM8K, and arithmetic problems
- **Comprehensive Logging**: Detailed statistics and progress tracking

## File Structure

```
reasoning_methods/hybrid/
├── star_main.py              # Main orchestrator
├── collect_rationales.py     # Rationale collection (Steps 3-6)
├── finetune.py              # Fine-tuning (Step 7)
├── prompting.py             # Generation and rationalization functions
├── prepare_datasets.py      # Dataset loading utilities
├── main.py                  # Original monolithic implementation
├── prompts/                 # Few-shot prompt files
│   ├── cqa_few_shot.txt
│   ├── gsm8k_few_shot.txt
│   └── arithmetic_few_shot.txt
├── collected_rationales/    # Generated rationale data
│   └── iteration_X/
│       ├── {dataset}_generated_rationales.jsonl
│       ├── {dataset}_rationalized_rationales.jsonl
│       └── {dataset}_collection_stats.json
└── star_models/            # Fine-tuned models
    └── iteration_X/
        ├── model/          # Saved model files
        └── training_metadata.json
```

## Usage

### Complete STaR Process

Run the full STaR algorithm with automatic iteration:

```bash
# Run STaR on CommonsenseQA
python star_main.py --dataset cqa --num_iterations 5

# Run STaR on GSM8K with debugging
python star_main.py --dataset gsm8k --num_iterations 3 --debug

# Run STaR with limited samples (for testing)
python star_main.py --dataset arithmetic --max_samples 100 --num_iterations 2
```

### Individual Phases

You can also run each phase separately:

#### 1. Collect Rationales Only

```bash
python star_main.py --collect_only --dataset cqa --iteration 1 --model_path meta-llama/Llama-3.2-1B
```

#### 2. Fine-tune Only

```bash
python star_main.py --finetune_only --dataset cqa --iteration 1
```

#### 3. Evaluate Only

```bash
python star_main.py --eval_only --dataset cqa --model_path ./star_models/iteration_1/model
```

### Direct Module Usage

You can also call the modules directly:

```bash
# Collect rationales
python collect_rationales.py --model_path meta-llama/Llama-3.2-1B --dataset cqa --iteration 1 --output_dir ./collected_rationales

# Fine-tune model
python finetune.py --rationales_dir ./collected_rationales --dataset cqa --iteration 1 --output_dir ./star_models

# The main orchestrator handles both automatically
```

## Command Line Arguments

### Main Arguments

- `--dataset`: Dataset type (`cqa`, `gsm8k`, `arithmetic`)
- `--num_iterations`: Number of STaR iterations (default: 5)
- `--max_samples`: Limit samples per iteration (useful for testing)
- `--debug`: Enable detailed debug output
- `--rationales_dir`: Directory to save collected rationales
- `--models_dir`: Directory to save fine-tuned models
- `--no_eval`: Skip evaluation after each iteration

### Individual Phase Arguments

- `--collect_only`: Only run rationale collection
- `--finetune_only`: Only run fine-tuning
- `--eval_only`: Only run evaluation
- `--iteration`: Iteration number (for individual phases)
- `--model_path`: Model path (for collection or evaluation)

## Algorithm Details

### STaR Algorithm (from the paper)

1. **Initialize**: Start with base model M₀
2. **For each iteration n = 1 to N**:
   - **Step 3**: Generate rationales using M_{n-1}
   - **Step 4**: Rationalize failures (generate explanations with correct answer hints)
   - **Step 5**: Filter successful self-generated rationales (D_n^gen)
   - **Step 6**: Filter successful rationalized explanations (D_n^rat)
   - **Step 7**: Fine-tune M₀ on D_n^gen ∪ D_n^rat → M_n

### Key Implementation Details

- **Always fine-tune from M₀**: Each iteration starts fine-tuning from the original base model, not the previous iteration's model
- **Verification**: Rationalized explanations are verified to ensure they lead to correct answers
- **Multiple attempts**: Up to 5 attempts for rationalization with different random seeds
- **Proper filtering**: Only rationales that lead to correct answers are kept

## Data Format

### Collected Rationales

Each rationale file contains JSONL with the following structure:

```json
{
  "example_id": 123,
  "question": {...},
  "rationale": "Step-by-step reasoning...",
  "answer": "correct_answer",
  "formatted_text": "Q: ... A: ... Therefore, the answer is ...",
  "source": "generated" | "rationalized",
  "attempt_number": 1
}
```

### Statistics

Collection statistics are saved for each iteration:

```json
{
  "total_processed": 1000,
  "initial_correct": 300,
  "rationalized_correct": 200,
  "final_failures": 500,
  "generated_rationales_count": 300,
  "rationalized_rationales_count": 200
}
```

## Configuration

Key parameters can be modified in the respective files:

- **Model**: `BASE_MODEL_ID` in each file
- **Training steps**: `INITIAL_TRAIN_STEPS` and `STEP_INCREASE_FACTOR` in `finetune.py`
- **Generation parameters**: `TEMPERATURE`, `TOP_P`, etc. in `prompting.py`
- **Batch sizes**: `PER_DEVICE_TRAIN_BATCH_SIZE` in `finetune.py`

## Requirements

- PyTorch
- Transformers
- TRL (for SFTTrainer)
- Datasets
- tqdm

## Differences from Original Implementation

The new modular implementation:

1. **Separates concerns**: Collection and fine-tuning are independent
2. **Persists data**: All rationales are saved to files
3. **Better error handling**: Each phase can fail independently
4. **More flexible**: Individual phases can be run separately
5. **Follows paper exactly**: Fine-tuning always starts from base model M₀

This allows for better debugging, data inspection, and experimental flexibility while maintaining the exact algorithm from the paper. 