import os
import torch
import uuid
from datetime import datetime

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

    def load_checkpoint(self, model, optimizer, epoch):
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
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        # if optimizer:
        #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from: {checkpoint_path}")

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
