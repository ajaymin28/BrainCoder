from utils.logman import logger
import torch
import psutil
import os

def memory_stats(get_dict=False, print_mem_usage=True, device=None):
    """
    Provides memory stats for, cpu%, ram% for process, 
    """
    stats = {
            "cpu": "",
            "ram": "",
            "cuda_free": .0,
            "cuda_total": .0,
            "cuda_allocated": .0,
            "cuda_reserved": .0,
            "peak_vram_allocated_mb": .0,
    }

    cuda_freeMem = 0
    cuda_total = 0
    cuda_allocated = 0
    cuda_reserved = 0
    peak_vram_allocated_bytes = 0
    peak_vram_allocated_mb = 0
    MB_eval_exp = 1024**2
    
    if torch.cuda.is_available():
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            cuda_freeMem, cuda_total  = torch.cuda.mem_get_info()
            stats["cuda_total"] = cuda_total = round(cuda_total/MB_eval_exp,2)
            stats["cuda_free"] = cuda_freeMem = round(cuda_freeMem/MB_eval_exp,2)
        except: pass
            
        try:
            stats["cuda_allocated"] = cuda_allocated = round(torch.cuda.memory_allocated()/MB_eval_exp, 3)
            stats["cuda_reserved"] = cuda_reserved = round(torch.cuda.memory_reserved()/MB_eval_exp, 3)
        except: pass

        try:
            peak_vram_allocated_bytes = torch.cuda.max_memory_allocated(device)
            stats["peak_vram_allocated_mb"] = peak_vram_allocated_mb  = peak_vram_allocated_bytes / (MB_eval_exp)
        except: pass


    process = psutil.Process(os.getpid())
    ram_mem_perc = process.memory_percent()
    cpu_usage = psutil.cpu_percent()

    stats["cpu"] = cpu_usage
    stats["ram"] = ram_mem_perc

    if print_mem_usage:
        logger.info(f"CPU: {cpu_usage:.2f}% RAM: {ram_mem_perc:.2f}% GPU memory Total: [{cuda_total:.2f}] Available: [{cuda_freeMem:.2f}]  Allocated: [{cuda_allocated:.2f}] Reserved: [{cuda_reserved:.2f}] Cuda Peak Mem: [{peak_vram_allocated_mb:.2f}]")

    if get_dict:
        return stats