import time
import sys
from datasets import load_dataset
import torch
import random
from .config import SEED  # Import the seed from config
import psutil  # For non-CUDA memory info


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
    Dynamically detect available hardware and configure parameters to maximize computational capacity.
    
    Returns:
       device (str): "cuda", "mps", or "cpu"
       batch_size (int): Dynamically calculated batch size based on available memory.
       num_gpus (int): Number of available GPUs.
       max_memory (dict): Maximum memory allocation per device in the format expected by transformers.
                           For GPUs, keys should be integers.
    """
    if torch.cuda.is_available():
        device = "cuda"
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA device(s)")
        
        # Initialize variables to store total and per-GPU memory (in MB)
        total_memory_mb = 0
        memory_per_gpu = {}
        
        # Check each GPU
        for i in range(num_gpus):
            # Get GPU properties
            gpu_properties = torch.cuda.get_device_properties(i)
            gpu_name = gpu_properties.name
            gpu_memory_mb = gpu_properties.total_memory / (1024 * 1024)  # Total memory in MB
            free_memory_mb = torch.cuda.mem_get_info(i)[0] / (1024 * 1024)  # Free memory in MB
            
            # Store memory info - use 85-90% of available memory to be safe for overhead
            memory_per_gpu[i] = min(gpu_memory_mb * 0.85, free_memory_mb * 0.9)  # Memory in MB
            total_memory_mb += memory_per_gpu[i]
            
            print(f"GPU {i}: {gpu_name}, Total Memory: {gpu_memory_mb/1024:.2f} GB, Free Memory: {free_memory_mb/1024:.2f} GB")
        
        # Build max_memory dictionary with keys for GPUs as integers
        max_memory = {}
        for i in range(num_gpus):
            # Convert available MB to GB and ensure at least 10GB per GPU to avoid offloading to disk
            available_gb = max(10, int(memory_per_gpu[i] / 1024))
            max_memory[i] = f"{available_gb}GB"
        
        # For CPU offloading, assign a small fixed amount (to avoid excessive offload)
        max_memory["cpu"] = "2GB"
        
        # Calculate batch size based on total available memory (heuristic: ~2GB per batch item)
        memory_per_item_gb = 2  # Change this value if you know the model requires more/less memory per item
        total_memory_gb = total_memory_mb / 1024
        
        estimated_batch_size = max(1, int(total_memory_gb / (memory_per_item_gb * 1.5)))  # using a 33% buffer
        
        # Set an upper cap for batch size
        batch_size = min(256, estimated_batch_size)
        
        print(f"Dynamically configured batch_size={batch_size}, using {num_gpus} GPUs with max_memory={max_memory}")
        
        return device, batch_size, num_gpus, max_memory
        
    elif torch.backends.mps.is_available():
        device = "mps"
        total_mem = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # in GB
        
        ml_memory = total_mem * 0.4
        batch_size = max(1, int(ml_memory))
        batch_size = min(32, batch_size)
        
        max_memory = {"mps": f"{int(ml_memory)}GB"}
        
        print(f"Running on Apple Silicon (MPS) with {total_mem:.1f}GB RAM, batch_size={batch_size}")
        
        return device, batch_size, 1, max_memory
    else:
        # Fallback for CPU only
        device = "cpu"
        total_mem = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # in GB
        
        ml_memory = total_mem * 0.3
        batch_size = max(1, int(ml_memory / 4))
        batch_size = min(8, batch_size)
        
        max_memory = {"cpu": f"{int(ml_memory)}GB"}
        
        print(f"Running on CPU with {total_mem:.1f}GB RAM, batch_size={batch_size}")
        
        return device, batch_size, 1, max_memory
