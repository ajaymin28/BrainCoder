import torch
import torch.distributed as dist
import os

# --- Data Parallel/Distributed Setup ---
def setup_distributed():
    if 'RANK' in os.environ:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend='nccl')


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_ddp():
    return 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1

def setup_ddp():
    if is_ddp():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        return local_rank
    else:
        return 0

def cleanup_ddp():
    if is_ddp():
        dist.destroy_process_group()

