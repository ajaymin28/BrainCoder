import os
import torch
import torch.nn as nn
import uuid
from datetime import datetime
from collections import OrderedDict
import torch.nn.init as init
import gc
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

import os

class GlobalConfig:
    """
    Global configuration settings for the EEG-Image project.
    Centralizes paths and other high-level settings.
    """
    # --- Base Directories ---
    # Adjust these paths based on your project structure and environment
    PROJECT_ROOT = "/home/jbhol/EEG/gits/BrainCoder"
    DATA_BASE_DIR = "/home/jbhol/EEG/gits/NICE-EEG"
    MODEL_BASE_DIR = os.path.join(PROJECT_ROOT, "model", "grok") # Base directory for saving models and checkpoints

    # --- Specific File/Directory Paths ---
    EEG_DATA_PATH = os.path.join(DATA_BASE_DIR, "Data", "Things-EEG2", "Preprocessed_data_250Hz")
    TEST_CENTER_PATH = os.path.join(DATA_BASE_DIR, "dnn_feature")
    IMG_DATA_PATH = os.path.join(DATA_BASE_DIR, "dnn_feature") 

    # --- Wandb Settings ---
    WANDB_PROJECT_NAME = "Transformer_VAE"
    WANDB_CODE_DIR = PROJECT_ROOT # Directory containing the code to be logged

    # --- Other Global Settings (optional) ---
    # Add any other settings that are constant across different training runs
    # e.g., default device, logging level, etc.
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Requires torch import if used here

# Instantiate the global configuration
global_config = GlobalConfig()


class TrainConfig:
    global_config = global_config  # Access the global config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 100 # subject data will be trained for local_epochs
    global_epochs = 1  # subject will be repeated for global_epochs*local_epochs
    current_epoch = 1
    current_global_epoch = 1
    current_itr = 0
    current_subject = 1
    batch_size = 1024

    channels = 63
    time_points = 250
    sessions = 4
    mean_eeg_data = False # Important for VAE input shape
    keep_dim_after_mean = False # Important for VAE input shape
    cache_data = True # Enable memmapping
    image_feature_dim = 768 # should be same as encoder_output_dim
    num_subjects = 10 # Number of subjects in the dataset

    train_batch_data = None
    val_batch_data = None
    batch_idx = 0
    val_batch_idx = 0
    runId = None

    load_pre_trained_models = False

    learning_rate = 0.002
    discriminator_lr = 0.002

    weight_decay = 1e-5
    latent_dim = 768 # should be same as image features
    dropout_rate = 0.1

    init_features = 32
    only_AE = False
    add_img_proj = True
    add_eeg_proj = False
    use_pre_trained_encoder = False

    # Populate TrainConfig with paths from GlobalConfig
    eeg_data_path = global_config.EEG_DATA_PATH
    test_center_path = global_config.TEST_CENTER_PATH
    model_save_base_dir = global_config.MODEL_BASE_DIR
    data_base_dir = global_config.DATA_BASE_DIR # Used by EEG_Dataset3
    
    Contrastive_augmentation = True
    nSub = 1
    nSub_Contrastive = 2
    
    EEG_Augmentation = False
    Total_Subjects = 2
    MultiSubject = True
    TestSubject = 1
    dnn = "clip"

    # adv training
    lambda_adv = 0.1 # If adv_loss dominates, reduce lambda_adv; if task performance(image features) suffers, increase it.
    max_lambda_adv = 0.1
    enable_adv_training = True

    #otho loss between cls feat and subj feat
    lambda_ortho = 0.0001

    alpha = 0.0
    
    log_test_data = True
    Train = True

    #debug
    profile_code = False

    encoder_output_dim = 768

import torch.distributed as dist
# --- Distributed Training Setup (if needed) ---
# This setup is for single-host multi-GPU. For multi-host, more complex setup is needed.
def setup_distributed(rank: int, world_size: int) -> None:
    """Sets up the distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost' # For single host
    os.environ['MASTER_PORT'] = '12355'     # Use a free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # NCCL for GPU

def cleanup_distributed() -> None:
    """Cleans up the distributed training environment."""
    dist.destroy_process_group()

def weights_init_normal(m):
    """
    Code used from : # https://github.com/eeyhsong/NICE-EEG/
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def freeze_model(model, train=False):
    for name, param in model.named_parameters():
        param.requires_grad = train
    return model

class RunManager:
    def __init__(self, run_id=None):
        timestamp_ymd = datetime.now().strftime("%Y%m%d")
        timestamp_hms = datetime.now().strftime("%H%M%S")
        short_uuid = str(uuid.uuid4()).split('-')[0]  # Shortened UUID
        self.run_id = run_id or os.path.join(timestamp_ymd,timestamp_hms,short_uuid)

    def getRunID(self):
        return self.run_id

class CheckpointManager:
    def __init__(self, prefix="checkpoint", base_dir="checkpoints"):
        """
        Initialize the CheckpointManager.

        Args:
            base_dir (str): Base directory to store all checkpoints.
            run_id (str, optional): Unique identifier for the run. If None, a new run_id is generated.
        """
        self.base_dir = base_dir
        self.run_dir = os.path.join(self.base_dir)
        self.prefix = prefix
        os.makedirs(self.run_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch):
        """
        Save the model and optimizer state as a checkpoint.

        Args:
            model (torch.nn.Module): Model to save.
            optimizer (torch.optim.Optimizer): Optimizer to save.
            epoch (int): Current epoch number.
        """
        checkpoint_path = os.path.join(self.run_dir, f"{self.prefix}_epoch_{epoch}.pth")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, checkpoint_path)
        # print(f"Checkpoint saved at: {checkpoint_path}")

    def load_checkpoint(self, model, optimizer, epoch, strict=True):
        """
        Load a checkpoint into the model and optimizer.

        Args:
            model (torch.nn.Module): Model to load state into.
            optimizer (torch.optim.Optimizer): Optimizer to load state into.
            epoch (int): Epoch number of the checkpoint to load.

        Returns:
            int: The epoch number of the loaded checkpoint.
        """
        checkpoint_path = os.path.join(self.run_dir, f"{self.prefix}_epoch_{epoch}.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        
        state_dict = checkpoint["model_state_dict"]  # Or checkpoint itself if it's just state_dict
        # Add 'module.' prefix if necessary
        if isinstance(model, nn.DataParallel):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[f"module.{k}"] = v
            model.load_state_dict(new_state_dict,strict=strict)
        else:
            model.load_state_dict(state_dict,strict=strict)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # print(f"Checkpoint loaded from: {checkpoint_path}")

        return checkpoint["epoch"]

    def list_checkpoints(self):
        """
        List all checkpoints for the current run_id.

        Returns:
            list: List of checkpoint filenames.
        """
        if not os.path.exists(self.run_dir):
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        return [f for f in os.listdir(self.run_dir) if f.endswith(".pth")]

# Example Usage
if __name__ == "__main__":
    model = torch.nn.Linear(10, 1)  # Example model
    optimizer = torch.optim.Adam(model.parameters())
    manager = CheckpointManager()

    print(f"Run ID: {manager.run_id}")

    # Save a checkpoint
    manager.save_checkpoint(model, optimizer, epoch=1)

    # List checkpoints
    print("Available Checkpoints:", manager.list_checkpoints())

    # Load the checkpoint
    manager.load_checkpoint(model, optimizer, epoch=1)

import os
import psutil

def memory_stats(get_dict=False, print_mem_usage=True):
    try:
        freeMem, total  = torch.cuda.mem_get_info()
        total = total/1024**2
        freeMem = freeMem/1024**2
    except:
        freeMem = 0
        total = 0

    try:
        cuda_allocated = torch.cuda.memory_allocated()/1024**2
        cuda_reserved = torch.cuda.memory_reserved()/1024**2
    except:
        cuda_allocated = 0
        cuda_reserved = 0

    process = psutil.Process(os.getpid())
    ram_mem_perc = process.memory_percent()
    cpu_usage = psutil.cpu_percent()

    if print_mem_usage:
        print(f"CPU: {cpu_usage:.2f}% RAM: {ram_mem_perc:.2f}% GPU memory Total: [{total:.2f}] Available: [{freeMem:.2f}] Allocated: [{cuda_allocated:.2f}] Reserved: [{cuda_reserved:.2f}]")

    if get_dict:
        return {
            "cpu": cpu_usage,
            "ram": ram_mem_perc,
            "cuda_free": freeMem,
            "cuda_total": total,
            "cuda_allocated": round(cuda_allocated,2),
            "cuda_reserved": round(cuda_reserved,2),
        }
    
memory_stats()


def tryDel(obj_name):
    """
    Deletes the given object from the global space.
    """
    try:
        globals()[obj_name]  # Check if the global variable exists
        del globals()[obj_name]
        print(f"Deleted: {obj_name}")  # Optional: Confirmation message
    except KeyError:
        pass
        print(f"Object '{obj_name}' not found in the global environment.")
    except Exception as e:
        print(f"An error occurred while trying to delete '{obj_name}': {e}")

def clean_mem(objects_to_del:list[str]):
    """
    Jaimin: This function helps free up CUDA memory for loading other models
    """
    for obj_name in objects_to_del:
        tryDel(obj_name)

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except:
        pass


def config_to_dict(config_class: TrainConfig) -> Dict[str, Any]:
    """Converts a TrainConfig object to a dictionary for logging."""
    # Convert torch.device to string for serialization
    return {k: (v if not isinstance(v, torch.device) else str(v))
            for k, v in vars(config_class).items() if not k.startswith("__")}

def compute_alpha(config: TrainConfig, dataloader_len: int) -> float:
    """
    Computes the adversarial loss weight (alpha) based on training progress.
    Uses a logistic function to schedule alpha from -1 towards 1 (or 0 to 1 if desired range).
    The original code computes 2./(1. + np.exp(-5 * p)) - 1, which ranges from -1 to 1.
    If alpha is used as a weight, it might be intended to range from 0 to 1.
    Assuming the original intent is to schedule from a lower value to a higher value.
    Let's keep the original calculation for now but add a note.
    """
    # Ensure current training state is tracked in the config
    if not all([hasattr(config, 'current_global_epoch'), hasattr(config, 'current_subject'),
                hasattr(config, 'current_epoch'), hasattr(config, 'batch_idx')]):
        print("Warning: Training state attributes not found in config. Alpha calculation may be incorrect.")
        return 0.0 # Return a default or raise an error

    # Total steps across all global epochs, subjects, local epochs, and batches
    total_steps = (config.global_epochs * config.Total_Subjects *
                   config.local_epochs * dataloader_len)

    # Calculate overall batch index that continuously increases
    overall_batch_idx = (
        (config.current_global_epoch - 1) * config.Total_Subjects * config.local_epochs * dataloader_len +
        (config.current_subject - 1) * config.local_epochs * dataloader_len +
        (config.current_epoch - 1) * dataloader_len +
        (config.batch_idx - 1)
    )

    # Progress ratio (0 to 1)
    p = float(overall_batch_idx) / total_steps

    # Compute alpha using the logistic function (ranges from -1 to 1)
    # If a 0-1 range is needed, consider 1. / (1. + np.exp(-k * p)) or similar.
    alpha = 2. / (1. + np.exp(-5 * p)) - 1
    return alpha