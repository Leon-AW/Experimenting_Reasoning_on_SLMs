import time
import sys
from datasets import load_dataset
import torch
import random
import os
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
    Optimized for high-end GPUs like A100s with 80GB VRAM.
    
    Returns:
       device (str): "cuda", "mps", or "cpu"
       batch_size (int): Dynamically calculated batch size based on available memory.
       num_gpus (int): Number of available GPUs.
       max_memory (dict): Maximum memory allocation per device in the format expected by transformers.
    """
    if torch.cuda.is_available():
        device = "cuda"
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA device(s)")
        
        # Initialize variables to store total and per-GPU memory (in MB)
        total_memory_mb = 0
        memory_per_gpu = {}
        
        # GPU model detection - to better optimize for A100s
        gpu_models = []
        has_a100 = False
        
        # Check each GPU
        for i in range(num_gpus):
            # Get GPU properties
            gpu_properties = torch.cuda.get_device_properties(i)
            gpu_name = gpu_properties.name
            gpu_models.append(gpu_name)
            
            gpu_memory_mb = gpu_properties.total_memory / (1024 * 1024)  # Total memory in MB
            free_memory_mb = torch.cuda.mem_get_info(i)[0] / (1024 * 1024)  # Free memory in MB
            
            # Check if this is an A100 GPU
            if "A100" in gpu_name:
                has_a100 = True
                # For A100s, we can use a higher percentage of memory safely
                utilization_factor = 0.95  # Use 95% of available memory for A100s
            else:
                utilization_factor = 0.85  # Use 85% for other GPUs
                
            # Store memory info with GPU-specific utilization factor
            memory_per_gpu[i] = min(gpu_memory_mb * utilization_factor, free_memory_mb * utilization_factor)
            total_memory_mb += memory_per_gpu[i]
            
            print(f"GPU {i}: {gpu_name}, Total Memory: {gpu_memory_mb/1024:.2f} GB, Free Memory: {free_memory_mb/1024:.2f} GB")
        
        # Build max_memory dictionary with keys for GPUs as integers
        max_memory = {}
        for i in range(num_gpus):
            # Convert available MB to GB and ensure appropriate minimum based on GPU type
            min_gb = 75 if "A100" in gpu_models[i] and "80GB" in gpu_models[i] else 10
            available_gb = max(min_gb, int(memory_per_gpu[i] / 1024))
            max_memory[i] = f"{available_gb}GB"
        
        # For CPU offloading, we can allocate more if we have A100s
        if has_a100:
            # More CPU memory for A100 setups to balance the workflow
            max_memory["cpu"] = "8GB"
        else:
            max_memory["cpu"] = "2GB"
        
        # Calculate batch size based on available memory and model size
        # The heuristic is adjusted based on detected GPU type
        if has_a100:
            # For A100 with 80GB, we can process many more samples per batch
            # These values are optimized for LLaMA-3 (3B and 1B) models specifically
            memory_per_item_gb = 0.75  # LLaMA models need less memory per item with optimization
            buffer_factor = 1.1  # Only 10% buffer needed with A100s
            
            # Further optimize based on environment variables if present
            if os.environ.get('MODEL_SIZE') == '1b':
                memory_per_item_gb = 0.5  # 1B model needs even less memory per item
            elif os.environ.get('MODEL_SIZE') == '3b':
                memory_per_item_gb = 1.0  # 3B model needs a bit more
        else:
            # Default values for other GPUs
            memory_per_item_gb = 2.0
            buffer_factor = 1.5  # 50% buffer for other GPUs
        
        total_memory_gb = total_memory_mb / 1024
        
        estimated_batch_size = max(1, int(total_memory_gb / (memory_per_item_gb * buffer_factor)))
        
        # Set batch size caps based on GPU type
        if has_a100:
            max_batch_size = 512  # Much higher for A100s
        else:
            max_batch_size = 256  # Default cap
            
        batch_size = min(max_batch_size, estimated_batch_size)
        
        # Enable tensor cores for faster computation on compatible GPUs
        if has_a100:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 precision for faster computation on A100 GPUs")
        
        # Configure PyTorch for maximum performance
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        
        print(f"Optimized configuration: batch_size={batch_size}, {num_gpus} GPUs with max_memory={max_memory}")
        
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
