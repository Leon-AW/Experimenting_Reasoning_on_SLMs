# Fine-Tuning Module for Orca-Style Reasoning

This directory contains the implementation for the **Multi-Stage Finetuning** reasoning approach, as outlined in the main project exposé. The goal is to replicate the methodology from the Orca 2 paper to enhance the reasoning capabilities of Small Language Models (SLMs) like Llama 3.2.

## Overview

The core idea, inspired by Orca 2, is to teach an SLM complex reasoning skills by training it on a high-quality, diverse dataset of problems and explanations. A key technique from the paper, "prompt erasure," is used to force the model to learn the underlying reasoning strategy rather than simply memorizing the format of a specific prompt.

This process is broken down into two main steps:
1.  **Dataset Creation**: A specialized training dataset is created from a source like FLAN-v2, where system prompts are "erased" and mixed with various task instructions.
2.  **Supervised Fine-Tuning (SFT)**: The base SLM is then fine-tuned on this newly created dataset.

## File Structure

- `create_mixed_dataset.py`: A script to process a source dataset (e.g., from Hugging Face), apply prompt erasure, and create a mixed dataset suitable for fine-tuning.
- `sft.py`: The main script for running the supervised fine-tuning process on a model using the dataset generated above.
- `README_FINETUNING.md`: This documentation file.

## Workflow and Usage

The process is a two-step pipeline. You must first create the dataset and then run the fine-tuning.

### Step 1: Create the Mixed Dataset

The `create_mixed_dataset.py` script prepares the data. It downloads, processes, and combines several reasoning-focused datasets into a single collection suitable for fine-tuning. The script applies the "prompt erasure" technique to the `Open-Orca/SlimOrca-Dedup` dataset within this mix.

All configuration, including the list of datasets and their respective proportions, is currently hardcoded within the script itself.

**Example Usage:**
To run the script, simply execute it directly:
```bash
python reasoning_methods/fine-tuning/create_mixed_dataset.py
```
This will save the final dataset to the hardcoded location: `reasoning_methods/fine-tuning/mixed_finetuning_dataset`.

### Step 2: Run Supervised Fine-Tuning

Once the dataset is created, the `sft.py` script fine-tunes the base model. The script is built using the TRL library and accepts a wide range of training arguments.

**Example Usage:**
A typical command for full fine-tuning on a single GPU would look like this:
```bash
python reasoning_methods/fine-tuning/sft.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_name "reasoning_methods/fine-tuning/mixed_finetuning_dataset" \
    --dataset_test_split validation \
    --output_dir "./models/llama-1b-orca-finetuned" \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --packing True \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --logging_steps 25
```
- `--model_name_or_path`: The base SLM to be fine-tuned.
- `--dataset_name`: Path to the dataset created in Step 1. Note this is a directory path.
- `--output_dir`: Where to save the final fine-tuned model.
- The script also supports multi-GPU training with `accelerate launch` and PEFT methods like LoRA. Refer to the docstring in `sft.py` for more advanced examples. 