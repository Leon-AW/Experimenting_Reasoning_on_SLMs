# optimization_config.py
"""
Advanced optimization settings for maximizing inference speed on high-end hardware.
These settings can be adjusted based on the specific hardware configuration.
"""

import os
import torch

# Inference Precision Settings
# Using bfloat16 for A100 GPUs provides the best balance of speed and accuracy
USE_BFLOAT16 = True  # Set to False to use float16 instead, which may be faster but less stable

# FlashAttention Settings (requires PyTorch 2.0+ and a compatible GPU)
# Flash Attention 2 dramatically speeds up the attention mechanism when available
USE_FLASH_ATTENTION = True

# Memory Optimization Settings
# These control the aggressive memory usage to maximize batch sizes
# Higher values = more aggressive memory usage (faster but risk of OOM)
A100_MEMORY_UTILIZATION = 0.95  # Use 95% of A100 memory
OTHER_GPU_MEMORY_UTILIZATION = 0.85  # Use 85% of other GPU memory
CPU_MEMORY_ALLOCATION = "8GB"  # CPU memory for model offloading

# Batch Size Optimization
# Memory per item estimates for different model sizes (in GB)
MEM_PER_ITEM_1B = 0.5  # 1B model memory per sample estimate
MEM_PER_ITEM_3B = 1.0  # 3B model memory per sample estimate
BATCH_SIZE_BUFFER = 1.1  # Safety buffer (lower = larger batches but risk of OOM)
MAX_BATCH_SIZE_A100 = 512  # Maximum allowed batch size for A100s
MAX_BATCH_SIZE_OTHER = 256  # Maximum allowed batch size for other GPUs

# Self-Consistency Optimization
# Controls how many SC paths to generate in parallel
SC_PATHS_PER_BATCH_A100 = 5  # Process this many SC paths at once on A100s
SC_PATHS_PER_BATCH_OTHER = 2  # Process this many SC paths at once on other GPUs

# Multi-Processing Optimization
# Controls the number of worker processes for data loading
NUM_WORKERS = min(4, os.cpu_count() // 2) if os.cpu_count() else 2

# KV Cache Optimization
# Experimental options to optimize KV cache usage
OPTIMIZE_KV_CACHE = True  # Enable experimental KV cache optimizations
KV_CACHE_BLOCK_SIZE = 64  # Tune this based on model size if needed

# Function to detect if we're running on A100 GPUs
def is_running_on_a100():
    """Detects if the code is running on A100 GPUs"""
    if not torch.cuda.is_available():
        return False
    
    for i in range(torch.cuda.device_count()):
        if "A100" in torch.cuda.get_device_properties(i).name:
            return True
    return False

# Function to get optimized model loading parameters
def get_optimized_model_params():
    """Returns optimized model loading parameters based on detected hardware"""
    params = {
        "low_cpu_mem_usage": True,
    }
    
    # Set precision (bfloat16 preferred for A100 GPUs if available)
    if USE_BFLOAT16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        params["torch_dtype"] = torch.bfloat16
    else:
        params["torch_dtype"] = torch.float16
        
    # Set Flash Attention if supported
    if USE_FLASH_ATTENTION and torch.cuda.is_available():
        try:
            # Attempt to use Flash Attention 2
            params["attn_implementation"] = "flash_attention_2"
        except:
            # Fallback to standard attention
            pass
            
    return params

# Function to get optimized memory settings
def get_optimized_memory_config():
    """Returns memory configuration optimized for the detected hardware"""
    if is_running_on_a100():
        return {
            "memory_utilization": A100_MEMORY_UTILIZATION,
            "cpu_memory": CPU_MEMORY_ALLOCATION,
            "max_batch_size": MAX_BATCH_SIZE_A100,
            "sc_paths_per_batch": SC_PATHS_PER_BATCH_A100,
            "memory_per_item": MEM_PER_ITEM_1B if os.environ.get('MODEL_SIZE') == '1b' else MEM_PER_ITEM_3B,
            "batch_size_buffer": BATCH_SIZE_BUFFER
        }
    else:
        return {
            "memory_utilization": OTHER_GPU_MEMORY_UTILIZATION,
            "cpu_memory": "2GB",
            "max_batch_size": MAX_BATCH_SIZE_OTHER,
            "sc_paths_per_batch": SC_PATHS_PER_BATCH_OTHER,
            "memory_per_item": MEM_PER_ITEM_1B * 1.5 if os.environ.get('MODEL_SIZE') == '1b' else MEM_PER_ITEM_3B * 1.5,
            "batch_size_buffer": BATCH_SIZE_BUFFER * 1.5
        } 