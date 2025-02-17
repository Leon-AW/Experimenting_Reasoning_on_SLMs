import time
import sys
from datasets import load_dataset
import torch
import random
from .config import SEED  # Import the seed from config


def load_custom_dataset(dataset_config, max_retries=3):
    """
    Generic loader that tries up to max_retries times to load the dataset
    based on the config.
    """
    random.seed(SEED)  # Set the random seed for reproducibility

    for attempt in range(max_retries):
        try:
            if dataset_config.get("is_mmlu", False):
                from datasets import get_dataset_config_names, disable_progress_bar
                disable_progress_bar()  # Disable the progress bar for cleaner output
                
                available_subjects = get_dataset_config_names("cais/mmlu")
                random.shuffle(available_subjects)  # Randomize subject order
                
                # Collect samples from multiple subjects
                all_samples = []
                samples_per_subject = 100  # Maximum samples to take from each subject
                
                for subject in available_subjects:
                    if len(all_samples) >= 1000:
                        break
                        
                    try:
                        # Load dataset for this subject quietly
                        subject_dataset = load_dataset("cais/mmlu", subject, split="test")
                        # Take up to 25 samples from this subject
                        subject_samples = list(subject_dataset)[:samples_per_subject]
                        all_samples.extend(subject_samples)
                        print(f"Loaded {len(subject_samples)} samples from {subject}")
                    except Exception as e:
                        continue
                
                # Ensure we have exactly 1000 samples
                all_samples = all_samples[:1000]
                if len(all_samples) == 1000:
                    from datasets import Dataset
                    return Dataset.from_list(all_samples)
                else:
                    continue  # Try again if we didn't get enough samples
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
            return device, 192, 3, "75GB"
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
