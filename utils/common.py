import os
import torch
import torch.nn as nn
import uuid
from datetime import datetime
from collections import OrderedDict
import torch.nn.init as init
import gc

class TrainConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 100 # subject data will be trained for local_epochs
    global_epochs = 1  # subject will be repeated for global_epochs*local_epochs
    current_epoch = 1
    current_global_epoch = 1
    current_subject = 1
    batch_size = 4096

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

    keepDimAfterAvg = False
    encoder_output_dim = 768

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

def memory_stats():
    freeMem, total  = torch.cuda.mem_get_info()
    process = psutil.Process(os.getpid())
    ram_mem_perc = process.memory_percent()
    cpu_usage = psutil.cpu_percent()
    print(f"CPU: {cpu_usage:.2f}% RAM: {ram_mem_perc:.2f}% GPU memory Total: [{total/1024**2:.2f}] Available: [{freeMem/1024**2:.2f}] Allocated: [{torch.cuda.memory_allocated()/1024**2:.2f}] Reserved: [{torch.cuda.memory_reserved()/1024**2:.2f}]")
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