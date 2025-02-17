import time
import sys
from datasets import load_dataset
import torch


def load_custom_dataset(dataset_config, max_retries=3):
    """
    Generic loader that tries up to max_retries times to load the dataset
    based on the config.
    """
    for attempt in range(max_retries):
        try:
            if dataset_config.get("is_mmlu", False):
                try:
                    # Load the dataset and select the correct split
                    dataset = load_dataset(dataset_config["name"])
                    dataset = dataset[dataset_config["split"]]
                except ValueError as e:
                    # If invalid subject, show available ones and exit
                    from datasets import get_dataset_config_names
                    available_subjects = get_dataset_config_names("cais/mmlu")
                    print(f"Error: Invalid MMLU subject. Available subjects are:\n{', '.join(available_subjects)}")
                    print(f"\nUsage example: python script.py --dataset mmlu --mmlu_subject high_school_mathematics")
                    sys.exit(1)
            else:
                dataset = load_dataset(dataset_config["name"], dataset_config["split"])
                if dataset_config["subset"]:
                    dataset = dataset[dataset_config["subset"]]
            return dataset
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to load dataset after {max_retries} attempts: {str(e)}")
                raise
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(5)


def configure_hardware():
    """
    Detect available hardware and return the device along with optimal hyperparameters.
    
    Returns:
       device (str): "cuda", "mps", or "cpu"
       batch_size (int): Optimal batch size based on hardware.
       num_gpus (int): Number of GPUs to use.
       max_memory (str): Maximum memory allocation per GPU.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "a100" in gpu_name:
            # NVIDIA A100 80GB
            return device, 96, 3, "75GB"
        elif "a6000" in gpu_name:
            # NVIDIA RTX A6000
            return device, 64, 4, "40GB"
        else:
            # Default CUDA configuration if GPU is not A100 or RTX A6000
            return device, 32, 1, "8GB"
    elif torch.backends.mps.is_available():
        device = "mps"
        # Hypothetical optimal hyperparameters for M4 Pro Chip
        return device, 16, 1, "8GB"
    else:
        # Fallback CPU settings
        return "cpu", 1, 1, "4GB"
