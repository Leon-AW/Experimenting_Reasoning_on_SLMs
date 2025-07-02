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


def configure_hardware(args=None):
    """
    Dynamically detect available hardware and configure parameters to maximize computational capacity.
    This function now dynamically excludes GPUs with low free memory.
    Optimized for high-end GPUs like A100s with 80GB VRAM.
    
    Returns:
       device (str): "cuda", "mps", or "cpu"
       batch_size (int): Dynamically calculated batch size based on available memory.
       num_gpus (int): Number of available GPUs.
       max_memory (dict): Maximum memory allocation per device in the format expected by transformers.
       usable_gpus_indices (list): Indices of usable GPUs for CUDA devices.
    """
    if torch.cuda.is_available():
        device = "cuda"
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA device(s)")
        
        # GPU model detection and memory info collection
        gpu_models = []
        has_a100 = False
        
        # Memory threshold to consider a GPU "busy" (in GB)
        BUSY_GPU_THRESHOLD_GB = 20
        
        # Collect info about each GPU
        gpus_info = []
        for i in range(num_gpus):
            gpu_properties = torch.cuda.get_device_properties(i)
            free_memory_mb = torch.cuda.mem_get_info(i)[0] / (1024 * 1024)
            gpus_info.append({
                "name": gpu_properties.name,
                "total_memory_mb": gpu_properties.total_memory / (1024 * 1024),
                "free_memory_mb": free_memory_mb
            })
            if "A100" in gpu_properties.name:
                has_a100 = True
            gpu_models.append(gpu_properties.name)
            print(f"GPU {i}: {gpus_info[i]['name']}, Total Memory: {gpus_info[i]['total_memory_mb']/1024:.2f} GB, Free Memory: {gpus_info[i]['free_memory_mb']/1024:.2f} GB")

        # Build max_memory dictionary and identify usable GPUs
        max_memory = {}
        usable_gpus_indices = []
        total_usable_memory_mb = 0
        
        for i in range(num_gpus):
            free_gb = gpus_info[i]['free_memory_mb'] / 1024
            if free_gb < BUSY_GPU_THRESHOLD_GB:
                print(f"GPU {i} is heavily utilized (Free Memory: {free_gb:.2f}GB). Excluding it from use.")
                max_memory[i] = "1MiB"  # Allocate minimal memory to effectively exclude
            else:
                usable_gpus_indices.append(i)
                # For A100s, we can use a higher percentage of memory safely
                utilization_factor = 0.95 if "A100" in gpus_info[i]['name'] else 0.85
                
                # Allocate memory based on what's free
                allocatable_memory_mb = gpus_info[i]['free_memory_mb'] * utilization_factor
                total_usable_memory_mb += allocatable_memory_mb
                max_memory[i] = f"{int(allocatable_memory_mb / 1024)}GiB"
        
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
            model_size = os.environ.get('MODEL_SIZE')
            if model_size == '1b':
                memory_per_item_gb = 0.5  # 1B model needs even less memory per item
            elif model_size == '3b':
                memory_per_item_gb = 1.0  # 3B model needs a bit more

            # Dynamically adjust for memory-intensive tasks
            if args:
                if args.self_consistency:
                    # SC with confidence scoring is very memory heavy
                    print("Self-consistency enabled, adjusting batch size for higher memory usage.")
                    memory_per_item_gb *= 4
                if args.dataset == 'drop':
                    # DROP dataset has long contexts, requiring more memory
                    print("DROP dataset detected, adjusting batch size for longer sequences.")
                    memory_per_item_gb *= 2
        else:
            # Default values for other GPUs
            memory_per_item_gb = 2.0
            buffer_factor = 1.5  # 50% buffer for other GPUs
        
        total_usable_memory_gb = total_usable_memory_mb / 1024
        
        if total_usable_memory_gb > 0:
            estimated_batch_size = max(1, int(total_usable_memory_gb / (memory_per_item_gb * buffer_factor)))
        else:
            estimated_batch_size = 1

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
        
        num_usable_gpus = len(usable_gpus_indices)
        print(f"Optimized configuration: batch_size={batch_size}, {num_usable_gpus} usable GPUs with max_memory={max_memory}")
        
        return device, batch_size, num_usable_gpus, max_memory, usable_gpus_indices
        
    elif torch.backends.mps.is_available():
        device = "mps"
        total_mem = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # in GB
        
        ml_memory = total_mem * 0.4
        batch_size = max(1, int(ml_memory))
        batch_size = min(32, batch_size)
        
        max_memory = {"mps": f"{int(ml_memory)}GB"}
        
        print(f"Running on Apple Silicon (MPS) with {total_mem:.1f}GB RAM, batch_size={batch_size}")
        
        return device, batch_size, 1, max_memory, None
    else:
        # Fallback for CPU only
        device = "cpu"
        total_mem = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # in GB
        
        ml_memory = total_mem * 0.3
        batch_size = max(1, int(ml_memory / 4))
        batch_size = min(8, batch_size)
        
        max_memory = {"cpu": f"{int(ml_memory)}GB"}
        
        print(f"Running on CPU with {total_mem:.1f}GB RAM, batch_size={batch_size}")
        
        return device, batch_size, 1, max_memory, None
