import gc
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb # Assuming wandb is installed and configured
from typing import List, Tuple, Dict, Any, Optional

# Assuming these are custom modules
from archs.TEncoder import EEGTransformerVAE
from archs.nice import Proj_eeg, Proj_img, Enc_eeg
from archs.CNNTrans import SubjectDiscriminator
from utils.common import (CheckpointManager, RunManager, TrainConfig, config_to_dict,
                          freeze_model, memory_stats, weights_init_normal)
# Assuming get_eeg_data is in utils.eeg_utils and uses paths passed to it
from utils.eeg_utils import get_eeg_data
from utils.datasets import EEG_Dataset3 # Assuming EEG_Dataset3 uses paths passed to it
# from utils.losses import vae_loss # Assuming get_contrastive_loss is implemented


# Import the new global configuration
from utils.common import global_config, clean_mem

# --- Configuration and Setup ---

# Define GPU usage
# Use global_config if you want to configure GPUs there, otherwise keep local
GPUS: List[int] = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, GPUS))

# Define tensor types for convenience
Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

def vae_loss(recon_x, x, mu, logvar, projected_z, image_features, subject_logits, subject_labels, logit_scale, use_feature_confusion, beta=1.0, alpha=1.0, gamma=0):
    if recon_x is not None:
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        recon_loss = torch.tensor(0.0, device=x.device)
        kl_loss = torch.tensor(0.0, device=x.device)

    proj_z_normalized = F.normalize(projected_z, dim=-1)
    image_features_normalized = F.normalize(image_features, dim=-1)
    logits = torch.matmul(proj_z_normalized, image_features_normalized.T) * logit_scale.exp()
    labels = torch.arange(proj_z_normalized.shape[0], device=proj_z_normalized.device)
    contrastive_loss = F.cross_entropy(logits, labels)

    if use_feature_confusion:
        probs = F.softmax(subject_logits, dim=-1)
        uniform = torch.full_like(probs, 1.0 / probs.size(1))
        subject_loss = F.kl_div(probs.log(), uniform, reduction='batchmean')
        # subject_loss_clamped = torch.clamp(subject_loss, max=1.0)
        total_loss = recon_loss + beta * kl_loss + alpha * contrastive_loss + gamma * subject_loss
    else:
        subject_loss = F.cross_entropy(subject_logits, subject_labels)
        # subject_loss_clamped = torch.clamp(subject_loss, max=1.0)
        total_loss = recon_loss + beta * kl_loss + alpha * contrastive_loss - gamma * subject_loss

    return total_loss, recon_loss, kl_loss, contrastive_loss, subject_loss

# --- Helper Functions ---

def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Can slow down training but ensures determinism

# --- Core Training and Testing Logic ---

def process_batch(models: Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]],
                  optimizers: Tuple[optim.Optimizer, Optional[optim.Optimizer]],
                  data,
                  config: TrainConfig,
                  dataloader_len: int,
                  is_validation: bool = False) -> Tuple[float, float, Optional[float], TrainConfig, Tuple[optim.Optimizer, Optional[optim.Optimizer]]]:
    """Processes a single batch for training or validation."""
    model, model_img_proj, model_eeg_proj, subject_discriminator = models
    optimizer, subject_optimizer = optimizers

    # Unpack batch data
    # Assuming batch data structure is consistent with EEG_Dataset3 output
    (eeg, image_features, cls_label_id, subid), \
    (eeg_2, image_features_2, cls_label_id2, subid2), \
    (eeg_neg, image_features_neg, cls_label_id_neg, subid1_neg) = data # Use config to pass batch data

    # Move data to GPU and set data types
    eeg = eeg.cuda().type(Tensor)
    # Handle optional augmentation data
    if config.Contrastive_augmentation and eeg_2!=[]:
        eeg_2 = eeg_2.cuda().type(Tensor) 
    else:
        eeg_2 = None
    image_features = image_features.cuda().type(Tensor)
    # Ensure labels and subids are on GPU and correct type
    # labels = torch.arange(eeg.shape[0]).cuda().type(LongTensor)
    subid_gpu = subid.cuda().type(LongTensor)
    # subid2_gpu = subid2.cuda().type(LongTensor) if subid2 is not None else None

    total_loss: Optional[torch.Tensor] = None
    contrastive_loss: Optional[torch.Tensor] = None
    
    if not is_validation:
        optimizer.zero_grad()
        if config.enable_adv_training and subject_optimizer:
            subject_optimizer.zero_grad()

    if torch.isnan(eeg).any():
        raise ValueError("NaN detected in eeg tensor")
    with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):

        recon, mu, logvar, z, subj_logits, proj, logit_scale, use_feature_confusion = model(eeg)

        # Apply projection layers if configured
        if config.add_img_proj and model_img_proj:
            image_features = model_img_proj(image_features)
        if config.add_eeg_proj and model_eeg_proj:
            proj = model_eeg_proj(proj)
        
        total_loss, recon_loss, kl_loss, contrastive_loss, subject_loss = vae_loss(
            recon,
            eeg, # Original input x
            mu, # mu_averaged_for_heads from model.forward
            logvar, # logvar_averaged_for_heads from model.forward
            proj, # z_session_specific from model.forward
            image_features, # You would need to provide these
            subj_logits, # subject_logits from model.forward
            subid_gpu, # You would need to provide these
            logit_scale, # You would need to provide these
            use_feature_confusion
        )

        if not is_validation:
            if not torch.isnan(total_loss).any():
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)                  
                optimizer.step()

        data_dump = {
            "data": data,
            "model_outputs": [recon, mu, logvar, z, subj_logits, recon, mu, logvar, z, subj_logits, proj, logit_scale, use_feature_confusion]
        }
        if torch.isnan(total_loss).any():
            print(f"NaN detected in loss: {total_loss}")

            print(f"Recon: {recon_loss.item():.4f}")
            print(f"KL: {kl_loss.item():.4f}")
            print(f"Contrastive: {contrastive_loss.item():.4f}")
            print(f"Subject: {subject_loss.item():.4f}")
            print(f"Batch Loss: {total_loss.item():.4f}")

            # Save the tensor to a file
            if not os.path.exists('/home/jbhol/EEG/gits/BrainCoder/eeg_data_dump_NAN.pt'):
                torch.save(data_dump, '/home/jbhol/EEG/gits/BrainCoder/eeg_data_dump_NAN.pt')
        else:
            if not os.path.exists('/home/jbhol/EEG/gits/BrainCoder/eeg_data_dump_good.pt'):
                torch.save(data_dump, '/home/jbhol/EEG/gits/BrainCoder/eeg_data_dump_good.pt')


        print(f"[{config.batch_idx}/{dataloader_len}]  Batch Loss: {total_loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
              f"KL: {kl_loss.item():.4f}, Contrastive: {contrastive_loss.item():.4f}, "
              f"Subject: {subject_loss.item():.4f}")
    
        

        
        
    # Return losses and updated config/optimizers
    # Return scalar loss values by calling .item()
    return (total_loss.item() if total_loss is not None else 0.0,
            contrastive_loss.item() if contrastive_loss is not None else 0.0,
            recon_loss.item() if recon_loss is not None else 0.0,
            kl_loss.item() if kl_loss is not None else 0.0,
            subject_loss.item() if subject_loss is not None else 0.0,
            logit_scale,
            config,
            optimizers)


def train(models: Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]],
          optimizers: Tuple[optim.Optimizer, Optional[optim.Optimizer]],
          dataloaders: Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]],
          config: TrainConfig,
          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Tuple[Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]], Tuple[optim.Optimizer, Optional[optim.Optimizer]], Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Trains the models for a specified number of local epochs."""

    runId = config.runId
    print(f"Training model for sub: {config.nSub} contrastive: {config.nSub_Contrastive}")

    model, model_img_proj, model_eeg_proj, subject_discriminator = models
    optimizer, subject_optimizer = optimizers
    dataloader, val_dataloader = dataloaders

    # Initialize logit scale parameter for contrastive loss
    # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda() # Ensure logit_scale is on GPU

    # Criterion for adversarial training
    # adv_criterion = nn.CrossEntropyLoss() if config.enable_adv_training else None

    # Checkpoint managers - Use path from config
    cpm_AEnc_eeg = CheckpointManager(prefix="AEnc_eeg", base_dir=os.path.join(config.model_save_base_dir, runId))
    cpm_SubDisc = CheckpointManager(prefix="SubDisc", base_dir=os.path.join(config.model_save_base_dir, runId)) if config.enable_adv_training else None
    cpm_Proj_img = CheckpointManager(prefix="Proj_img", base_dir=os.path.join(config.model_save_base_dir, runId)) if config.add_img_proj else None
    cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg", base_dir=os.path.join(config.model_save_base_dir, runId)) if config.add_eeg_proj else None

    # Load pre-trained models if configured
    if config.use_pre_trained_encoder:
        print("Loading pre-trained encoder...")
        cpm_AEnc_eeg.load_checkpoint(model=model, optimizer=None, epoch="best")

    # Load checkpoints for resuming training if configured
    if config.load_pre_trained_models:
        print("Loading models for resuming training...")
        cpm_AEnc_eeg.load_checkpoint(model=model, optimizer=optimizer, epoch="best")
        if config.add_img_proj and cpm_Proj_img:
            cpm_Proj_img.load_checkpoint(model=model_img_proj, optimizer=optimizer, epoch="best")
        if config.add_eeg_proj and cpm_Proj_eeg:
            cpm_Proj_eeg.load_checkpoint(model=model_eeg_proj, optimizer=optimizer, epoch="best")
        if config.enable_adv_training and cpm_SubDisc and subject_optimizer:
            cpm_SubDisc.load_checkpoint(model=subject_discriminator, optimizer=subject_optimizer, epoch="best")

    best_val_loss = float('inf')

    for local_epoch in range(config.local_epochs):
        config.current_epoch = local_epoch + 1 # Update current local epoch in config

        
        metric_keys = ["total", "img", "recon", "kl", "sub"]

        # Initialize epoch loss trackers
        train_losses = {}
        val_losses= {}
        for mkey in metric_keys:
            train_losses[mkey] = 0.0
            val_losses[mkey] = 0.0

        # Set models to training mode and enable gradients as per config
        model = freeze_model(model=model, train=True)
        if config.add_img_proj and model_img_proj:
            model_img_proj = freeze_model(model=model_img_proj, train=True)
        if config.add_eeg_proj and model_eeg_proj:
            model_eeg_proj = freeze_model(model=model_eeg_proj, train=True)
        if config.enable_adv_training and subject_discriminator:
            subject_discriminator = freeze_model(model=subject_discriminator, train=True)

        # Training Loop
        dataloader_len = len(dataloader)
        for idx, batch_data in enumerate(dataloader):
            config.batch_idx = idx + 1 # Update current batch index in config
            # config.train_batch_data = batch_data # Pass batch data via config

            # Process the training batch
            total_loss,contrastive_loss, recon_loss, kl_loss, subject_loss,logit_scale, config, optimizers = process_batch(
                models, optimizers, 
                batch_data,
                config, dataloader_len, is_validation=False
            )

            # Accumulate training losses
            train_losses["total"] += total_loss
            train_losses["img"] += contrastive_loss
            train_losses["recon"] += recon_loss
            train_losses["kl"] += kl_loss
            train_losses["sub"] += subject_loss


            # Log batch-level metrics if needed (optional)
            # if config.log_batch_metrics:
            #     wandb.log({"batch_total_loss": total_loss, ...})

        # Validation Loop
        if val_dataloader is not None:
            # Set models to evaluation mode and disable gradients as per config
            model = freeze_model(model=model, train=False)
            if config.add_img_proj and model_img_proj:
                model_img_proj = freeze_model(model=model_img_proj, train=False)
            if config.add_eeg_proj and model_eeg_proj:
                model_eeg_proj = freeze_model(model=model_eeg_proj, train=False)
            if config.enable_adv_training and subject_discriminator:
                subject_discriminator = freeze_model(model=subject_discriminator, train=False)

            val_dataloader_len = len(val_dataloader)
            with torch.no_grad():
                for vidx, vbatch_data in enumerate(val_dataloader):
                    config.batch_idx = vidx + 1 # Update batch index for validation logging
                    # config.train_batch_data = vbatch_data # Pass batch data via config

                    # Process the validation batch
                    total_loss,contrastive_loss, recon_loss, kl_loss, subject_loss,logit_scale, config, optimizers = process_batch(
                        models, optimizers, 
                        vbatch_data,
                        config,val_dataloader_len, is_validation=True
                    )

                    # Accumulate validation losses
                    val_losses["total"] += total_loss
                    val_losses["img"] += contrastive_loss
                    val_losses["recon"] += recon_loss
                    val_losses["kl"] += kl_loss
                    val_losses["sub"] += subject_loss

                    

        # Calculate average epoch losses
        avg_train_total_loss = train_losses["total"] / dataloader_len
        avg_train_img_loss = train_losses["img"] / dataloader_len
        avg_train_recon_loss = train_losses["recon"] / dataloader_len
        avg_train_kl_loss = train_losses["kl"] / dataloader_len
        avg_train_sub_loss = train_losses["sub"] / dataloader_len


        avg_val_total_loss = val_losses["total"] / val_dataloader_len if val_dataloader is not None else 0.0
        avg_val_img_loss = val_losses["img"] / val_dataloader_len if val_dataloader is not None else 0.0
        avg_val_recon_loss = val_losses["recon"] / val_dataloader_len if val_dataloader is not None else 0.0
        avg_val_kl_loss = val_losses["kl"] / val_dataloader_len if val_dataloader is not None else 0.0
        avg_val_sub_loss = val_losses["sub"] / val_dataloader_len if val_dataloader is not None else 0.0



        # Log epoch-level metrics to wandb
        log_dict = {
            "epoch": config.current_global_epoch * config.local_epochs + config.current_epoch, # Global epoch count
            "total_loss": avg_train_total_loss,
            "img_loss": avg_train_img_loss,
            "recon_loss": avg_train_recon_loss,
            "kl_loss": avg_train_kl_loss,
            "sub_loss": avg_train_sub_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "logit_scale": logit_scale.item(),
        }
        if val_dataloader is not None:
             log_dict.update({
                "val_total_loss": avg_val_total_loss,
                "val_img_loss": avg_val_img_loss,
                "val_recon_loss": avg_val_recon_loss,
                "val_kl_loss": avg_val_kl_loss,
                "val_sub_loss": avg_val_sub_loss,
             })

        wandb.log(log_dict)

        # Print epoch summary
        subject_print = f"Sub:[{config.nSub}]" if not config.MultiSubject else f"Sub:[{config.nSub}][c-{config.nSub_Contrastive}]"
        print_stmt = (
            f"{subject_print} Global Epoch {config.current_global_epoch}/{config.global_epochs}, "
            f"Local Epoch {config.current_epoch}/{config.local_epochs}, "
            f"Total Loss: {avg_train_total_loss:.4f}, Recon: {avg_train_recon_loss:.4f}, "
            f"KL: {avg_train_kl_loss:.4f}, Contrastive: {avg_train_img_loss:.4f}, "
            f"Subject: {avg_train_sub_loss:.4f}"
        )

        print(print_stmt)

        # if config.enable_adv_training:
        #      print_stmt += f", Train Adv: {avg_train_adv_loss:.4f}"
        if val_dataloader is not None:
            print_stmt_val = (
                f"{subject_print} Global Epoch {config.current_global_epoch}/{config.global_epochs}, "
                f"Local Epoch {config.current_epoch}/{config.local_epochs}, "
                f"vTotal Loss: {avg_val_total_loss:.4f}, vRecon: {avg_val_recon_loss:.4f}, "
                f"vKL: {avg_val_kl_loss:.4f}, vContrastive: {avg_val_img_loss:.4f}, "
                f"vSubject: {avg_val_sub_loss:.4f}"
            )
            print(print_stmt_val)
           

        # Checkpoint saving based on validation loss
        if val_dataloader is not None and avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            print(f"Validation loss improved. Saving checkpoint for epoch {config.current_epoch}.")

            # Save best models. Use .module if using DataParallel.
            if config.enable_adv_training and cpm_SubDisc and subject_discriminator:
                cpm_SubDisc.save_checkpoint(model=subject_discriminator.module if isinstance(subject_discriminator, nn.DataParallel) else subject_discriminator,
                                            optimizer=subject_optimizer, epoch="best")
            # Only save the encoder if it's being trained in this run
            if not config.use_pre_trained_encoder and cpm_AEnc_eeg:
                 cpm_AEnc_eeg.save_checkpoint(model=model.module if isinstance(model, nn.DataParallel) else model,
                                             optimizer=optimizer, epoch="best")
            if config.add_img_proj and cpm_Proj_img and model_img_proj:
                 cpm_Proj_img.save_checkpoint(model=model_img_proj.module if isinstance(model_img_proj, nn.DataParallel) else model_img_proj,
                                             optimizer=optimizer, epoch="best")
            if config.add_eeg_proj and cpm_Proj_eeg and model_eeg_proj:
                 cpm_Proj_eeg.save_checkpoint(model=model_eeg_proj.module if isinstance(model_eeg_proj, nn.DataParallel) else model_eeg_proj,
                                             optimizer=optimizer, epoch="best")

        # Step the scheduler if provided (e.g., ReduceLROnPlateau on validation loss)
        if scheduler and val_dataloader is not None:
            scheduler.step(avg_val_total_loss)
        elif scheduler: # Step based on training loss if no validation set
             scheduler.step(avg_train_total_loss)


    print("Training complete for this subject/global epoch.")
    return models, optimizers, scheduler

def test(runId: str,
         models: Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]],
         subject_id: int,
         dnn: str = "clip", # Assuming dnn refers to the image feature type
         batch_size: int = 8,
         with_img_projection: bool = False, # Not used in original test function, but kept for clarity
         with_eeg_projection: bool = False,
         config: Optional[TrainConfig] = None) -> Tuple[float, float, float]:
    """Evaluates the model on the test set for a single subject."""

    print(f"Testing model for Subject: {subject_id}")

    # Load test data - Use path from config
    _, _, test_eeg, test_label = get_eeg_data(eeg_data_path=config.eeg_data_path, nSub=subject_id, subset="test")

    # Load test centers (image features) - Use path from config
    test_center_path = os.path.join(config.test_center_path, f'center_{dnn}.npy')
    if not os.path.exists(test_center_path):
        print(f"Error: Test center file not found at {test_center_path}")
        return 0.0, 0.0, 0.0 # Return zero accuracy if centers are missing

    test_center = np.load(test_center_path, allow_pickle=True)

    # Convert data to PyTorch tensors
    test_eeg_tensor = torch.from_numpy(test_eeg).type(Tensor) # Move to GPU immediately
    test_center_tensor = torch.from_numpy(test_center).type(Tensor) # Move to GPU immediately
    test_label_tensor = torch.from_numpy(test_label).type(LongTensor) # Move to GPU immediately

    # Create test dataset and dataloader
    test_dataset = torch.utils.data.TensorDataset(test_eeg_tensor, test_label_tensor)
    # Use a larger batch size for testing if memory allows
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Get models
    model, model_img_proj, model_eeg_proj, _ = models # Subject discriminator not needed for testing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) # Ensure model is on the correct device

    # Load the best trained checkpoint for the encoder - Use path from config
    cpm_AEnc_eeg = CheckpointManager(prefix="AEnc_eeg", base_dir=os.path.join(config.model_save_base_dir, runId))
    try:
        # Load only the model state dictionary, optimizer is not needed for inference
        cpm_AEnc_eeg.load_checkpoint(model=model, optimizer=None, epoch="best", strict=True)
    except FileNotFoundError:
        print(f"Error: Best checkpoint for AEnc_eeg not found for run {runId}. Cannot perform test.")
        return 0.0, 0.0, 0.0
    except Exception as e:
        print(f"Error loading AEnc_eeg checkpoint: {e}")
        return 0.0, 0.0, 0.0


    # Load EEG projection model if used during training
    if with_eeg_projection:
        if model_eeg_proj is None:
             # Re-initialize the projection model if it wasn't created in __main__
             # This might be necessary if test is run independently
             if config is None:
                 print("Error: Config is required to initialize Proj_eeg for testing.")
                 return 0.0, 0.0, 0.0
             model_eeg_proj = Proj_eeg(embedding_dim=config.encoder_output_dim, proj_dim=768).to(device)
             print("Initialized Proj_eeg for testing.")

        model_eeg_proj = model_eeg_proj.to(device) # Ensure on device
        # Use path from config
        cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg", base_dir=os.path.join(config.model_save_base_dir, runId))
        try:
            cpm_Proj_eeg.load_checkpoint(model=model_eeg_proj, optimizer=None, epoch="best", strict=True)
        except FileNotFoundError:
            print(f"Error: Best checkpoint for Proj_eeg not found for run {runId}. Testing without EEG projection.")
            model_eeg_proj = None # Disable projection if checkpoint is missing
        except Exception as e:
            print(f"Error loading Proj_eeg checkpoint: {e}. Testing without EEG projection.")
            model_eeg_proj = None # Disable projection if checkpoint is missing


    # Set models to evaluation mode
    model.eval()
    if model_eeg_proj:
        model_eeg_proj.eval()

    total_samples = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    # Testing loop
    with torch.no_grad():
        for teeg_batch, tlabel_batch in test_dataloader:

            # Encode EEG data
            recon, mu, logvar, z, subj_logits, proj, logit_scale, use_feature_confusion = model(teeg_batch)

            # Apply EEG projection if used during training
            if model_eeg_proj:
                proj = model_eeg_proj(z)

            # Compute similarity (using original approach)
            similarity = (100.0 * proj @ test_center_tensor.t()).softmax(dim=-1)
            _, indices = similarity.topk(5, dim=-1) # Get top 5 indices

            # Reshape true labels for comparison
            tlabel_batch_reshaped = tlabel_batch.view(-1, 1)

            # Count correct predictions
            total_samples += tlabel_batch.size(0)
            top1_correct += (tlabel_batch_reshaped == indices[:, :1]).sum().item()
            top3_correct += (tlabel_batch_reshaped == indices[:, :3]).sum().item()
            # For top-5, check if the true label is present anywhere in the top 5 indices
            top5_correct += (tlabel_batch_reshaped == indices).sum().item()


    # Calculate accuracies
    top1_acc = float(top1_correct) / float(total_samples) if total_samples > 0 else 0.0
    top3_acc = float(top3_correct) / float(total_samples) if total_samples > 0 else 0.0
    top5_acc = float(top5_correct) / float(total_samples) if total_samples > 0 else 0.0

    # Print results for the subject
    print(f'Subject: [{subject_id}] Test Top1-{top1_acc:.6f}, Top3-{top3_acc:.6f}, Top5-{top5_acc:.6f}')

    # Log test data to wandb if configured
    if config and config.log_test_data:
        wandb.log({
            f"test_subject_{subject_id}/top-1": top1_acc,
            f"test_subject_{subject_id}/top-3": top3_acc,
            f"test_subject_{subject_id}/top-5": top5_acc,
        })

    return top1_acc, top3_acc, top5_acc

# --- Main Execution Block ---

if __name__ == "__main__":

    set_seed(42)

    # --- Configuration ---
    t_config = TrainConfig()

    # Hyperparameters
    t_config.learning_rate = 1e-4
    t_config.discriminator_lr = 1e-4
    # Note: Batch size of 16000 is very large and might require significant GPU memory.
    # Ensure your hardware can handle this.
    t_config.batch_size = 512
    t_config.local_epochs = 100
    t_config.global_epochs = 1 # Number of passes through the entire training subject list

    # General Config
    t_config.Train = True # Set to False to only run testing
    t_config.log_test_data = t_config.Train # Log test results to wandb only if training
    t_config.Contrastive_augmentation = True # Enables subject pair EEG
    t_config.MultiSubject = True # Set to True for multi-subject training
    t_config.TestSubject = 1 # This subject will be used for testing, others for training
    # Note: If MultiSubject is False, this subject is used for both train/val/test (split handled by get_eeg_data/EEG_Dataset3)
    

    # Adversarial Training Config
    t_config.enable_adv_training = True
    # Alpha is scheduled, this initial value might not be used directly in compute_alpha
    # t_config.alpha = 1.0 # This is scheduled across training steps from 0 to 1
    t_config.lambda_adv = 0.01 # Weight for the adversarial loss

    # Model Architecture Config (related to reproducing NICE)
    t_config.add_img_proj = False
    t_config.add_eeg_proj = False
    # Set encoder output dim based on model/task (1440 for NICE, 768 for others like CLIP)
    t_config.encoder_output_dim = 768

    # Data Loading Config
    t_config.cache_data = False # Cache data in memory (requires significant RAM for large datasets)
    t_config.mean_eeg_data = False # Apply mean normalization to EEG data (check dataset implementation)
    t_config.keep_dim_after_mean = False # for NICE model use true else False
    # Assuming these attributes are needed by EEGVAE and EEG_Dataset3
    t_config.channels = 63 # Example value
    t_config.time_points = 250 # Example value
    t_config.sessions = 4 # Example value
    t_config.image_feature_dim = 768 # Example value (e.g., CLIP feature dimension)
    t_config.latent_dim = 768 # Example value (encoder output dimension before projection)
    t_config.num_subjects = 2 # Total number of subjects available

    # Checkpoint and Run Management
    # Set run_id_to_test if you want to test a specific pre-trained run
    run_id_to_test: Optional[str] = None # Example: "your_previous_run_id"
    # If Train is False, run_id_to_test *must* be provided
    if not t_config.Train and run_id_to_test is None:
        raise ValueError("run_id_to_test must be provided when t_config.Train is False.")
    # If Train is True, run_id_to_test should generally be None for a new run
    if t_config.Train and run_id_to_test is not None:
         print("Warning: run_id_to_test is set but t_config.Train is True. Starting a new run.")
         run_id_to_test = None # Ensure a new run ID is generated

    # Option to load pre-trained models for fine-tuning or testing
    t_config.use_pre_trained_encoder = False # Load only the encoder checkpoint before training
    t_config.load_pre_trained_models = False # Load all model/optimizer states for resuming training

    # Option to profile code (currently commented out in train function)
    t_config.profile_code = False

    # Initialize RunManager to get/set run ID
    runMan = RunManager(run_id=run_id_to_test)
    runId = runMan.getRunID()
    t_config.runId = runId
    print(f"Run ID: {runId}")

    # --- Model Initialization ---
    # Initialize core EEG encoder 
    model = EEGTransformerVAE(
        channels= t_config.channels,
        time_points=t_config.time_points,
        sessions=t_config.sessions,
        latent_dim=t_config.latent_dim, # Latent dim might be different from encoder_output_dim depending on VAE structure
        image_feature_dim=t_config.image_feature_dim,
        num_subjects=t_config.num_subjects, # Needed for VAE if subject-conditional
        encoder_only=True,
        use_flash=True,
        use_feature_confusion=False,
        grl_alpha=0.1,
        mean_session=t_config.mean_eeg_data
    ).cuda()

    # Wrap with DataParallel if using multiple GPUs
    if len(GPUS) > 1:
        model = nn.DataParallel(model, device_ids=GPUS)
    model.apply(weights_init_normal) # Apply weight initialization

    # Initialize projection models if configured
    model_img_proj: Optional[nn.Module] = None
    if t_config.add_img_proj:
        model_img_proj = Proj_img(embedding_dim=t_config.image_feature_dim, proj_dim=t_config.encoder_output_dim).cuda()
        if len(GPUS) > 1:
            model_img_proj = nn.DataParallel(model_img_proj, device_ids=GPUS)
        model_img_proj.apply(weights_init_normal)

    model_eeg_proj: Optional[nn.Module] = None
    if t_config.add_eeg_proj:
        model_eeg_proj = Proj_eeg(embedding_dim=t_config.encoder_output_dim, proj_dim=768).cuda() # Assuming proj_dim is 768
        if len(GPUS) > 1:
            model_eeg_proj = nn.DataParallel(model_eeg_proj, device_ids=GPUS)
        model_eeg_proj.apply(weights_init_normal)

    # Initialize Subject Discriminator if adversarial training is enabled
    subj_discriminator: Optional[nn.Module] = None


    # --- Optimizer and Scheduler Setup ---
    # Define parameters to be optimized
    trainable_params: List[torch.Tensor] = []
    if not t_config.use_pre_trained_encoder:
        trainable_params += list(model.parameters())
    if t_config.add_img_proj and model_img_proj:
        trainable_params += list(model_img_proj.parameters())
    if t_config.add_eeg_proj and model_eeg_proj:
        trainable_params += list(model_eeg_proj.parameters())
    # Note: Discriminator parameters are optimized separately by subject_optimizer

    # Optimizer for the encoder and projection layers
    optimizer = optim.Adam(trainable_params, lr=t_config.learning_rate)

    # Optimizer for the subject discriminator
    subject_optimizer: Optional[optim.Optimizer] = None
    if t_config.enable_adv_training and subj_discriminator:
        subject_optimizer = optim.Adam(subj_discriminator.parameters(), lr=t_config.discriminator_lr)

    # Learning rate scheduler (optional)
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    # Example: ReduceLROnPlateau scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Bundle models and optimizers
    models = (model, model_img_proj, model_eeg_proj, subj_discriminator)
    optimizers = (optimizer, subject_optimizer)

    # --- Weights and Biases (wandb) Setup ---
    # Prepare config dictionary for wandb logging
    config_dict = config_to_dict(t_config)
    config_dict["runid"] = runId
    # Add any specific notes or changes to the config log
    config_dict["notes"] = "Cleaned and simplified training script with global config"

    wandb_run = None
    if t_config.Train:
        # Initialize wandb run - Use paths from global_config
        wandb_run = wandb.init(
            # id="your_resume_id", # Uncomment and set ID to resume a run
            project=global_config.WANDB_PROJECT_NAME, # Set your wandb project name
            config=config_dict, # Log hyperparameters and config
            settings=wandb.Settings(code_dir=global_config.WANDB_CODE_DIR) # Log code
        )
        # Log the code directory
        if wandb_run:
            wandb_run.log_code(global_config.WANDB_CODE_DIR)

    # --- Data Loading and Training Loop ---
    if t_config.Train:
        # Define the list of subjects to train on
        if t_config.MultiSubject:
            # Train on all subjects except the test subject
            total_subjects_list = list(range(1, t_config.num_subjects + 1)) # Assuming subjects are 1-indexed
            if t_config.TestSubject in total_subjects_list:
                total_subjects_list.remove(t_config.TestSubject)
            print("Subjects to be trained:", total_subjects_list)
            t_config.Total_Subjects = len(total_subjects_list) # Update total subjects in config
        else:
            # Train only on the specified test subject (for single-subject experiments)
            total_subjects_list = [t_config.TestSubject]
            print("Training on single subject:", total_subjects_list)
            t_config.Total_Subjects = 1 # Update total subjects in config

        # Global Epochs Loop (iterating through the list of training subjects)
        for g_epoch_idx in range(t_config.global_epochs):
            t_config.current_global_epoch = g_epoch_idx + 1 # Update current global epoch

            print(f"\n--- Global Epoch {t_config.current_global_epoch}/{t_config.global_epochs} ---")
            memory_stats() # Print memory usage before loading data

            # Create and load dataset for the current global epoch - Use path from config
            # Note: In multi-subject training, this dataset might load data for all subjects
            # in `total_subjects_list`. If memory is an issue, consider loading data
            # subject-by-subject within the global epoch loop or using a dataset
            # that loads data on-the-fly. The current implementation loads all data
            # for the list of subjects into memory if cache_data is True.
            dataset = EEG_Dataset3(
                args=t_config,
                nsubs=total_subjects_list, # Pass the list of subjects for this global epoch
                subset="train",
                agument_data=t_config.Contrastive_augmentation,
                cache_data=t_config.cache_data,
                mean_eeg_data=t_config.mean_eeg_data,
                keep_dim_after_mean=t_config.keep_dim_after_mean,
                saved_data_path=t_config.data_base_dir, # Use path from config
                eeg_data_shape=(t_config.sessions, t_config.channels, t_config.time_points),
                eeg_data_dtype=np.float32
            )

            val_dataset = EEG_Dataset3(
                args=t_config,
                nsubs=total_subjects_list, # Use the same subjects for validation or a separate validation set list
                subset="val", # Or "test" if using test set as validation
                agument_data=False, # Usually no augmentation in validation
                cache_data=t_config.cache_data,
                mean_eeg_data=t_config.mean_eeg_data,
                keep_dim_after_mean=t_config.keep_dim_after_mean,
                saved_data_path=t_config.data_base_dir, # Use path from config
                eeg_data_shape=(t_config.sessions, t_config.channels, t_config.time_points),
                eeg_data_dtype=np.float32
            )

            # Create dataloaders
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=t_config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=t_config.batch_size, shuffle=False, num_workers=0, pin_memory=True) # Usually no shuffle for validation

            # Perform training for local epochs on the current dataset
            models, optimizers, scheduler = train(
                models=models,
                dataloaders=(dataloader, val_dataloader),
                optimizers=optimizers,
                scheduler=scheduler,
                config=t_config
            )

            # Clean up memory after processing data for the global epoch
            print("Cleaning up data loaders and datasets...")
            del dataset, val_dataset, dataloader, val_dataloader
            print("After deleting data objects:")
            memory_stats()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Memory cleanup complete.")

    # --- Testing ---
    print("\n--- Starting Testing ---")

    # Determine which subjects to test
    subjects_to_test_list: List[int] = []
    if t_config.Train:
        # If trained, test on the specified TestSubject
        subjects_to_test_list = [t_config.TestSubject]
        print(f"Testing on specified test subject: {t_config.TestSubject}")
    else:
        # If not training, test on all subjects (or a specified list)
        # Assuming you want to test on subjects 1 to 10 if not training
        subjects_to_test_list = list(range(1, t_config.num_subjects + 1))
        print(f"Testing on subjects: {subjects_to_test_list}")


    # Store test results
    all_subjects_results: Dict[int, Dict[str, float]] = {}

    # Perform testing for each subject in the test list
    for sub in subjects_to_test_list:
        t_config.current_subject = sub # Update current subject in config for logging

        top1_acc, top3_acc, top5_acc = test(
            runId=runId,
            models=models,
            subject_id=sub,
            batch_size=t_config.batch_size, # Use same batch size for consistency or larger for test
            with_eeg_projection=t_config.add_eeg_proj,
            config=t_config # Pass config for mean reduction and logging, and paths
        )
        all_subjects_results[sub] = {"top-1": top1_acc, "top-3": top3_acc, "top-5": top5_acc}

    # Print summary of test results
    print(f"\n--- Test Results Summary for Run ID: {runId} ---")
    top1_scores = []
    top3_scores = []
    top5_scores = []

    for sub, scores in all_subjects_results.items():
        print(f"Subject: [{sub}] "
              f"Top1-{scores['top-1']:.6f}, "
              f"Top3-{scores['top-3']:.6f}, "
              f"Top5-{scores['top-5']:.6f}")
        top1_scores.append(scores['top-1'])
        top3_scores.append(scores['top-3'])
        top5_scores.append(scores['top-5'])

    # Calculate and print mean accuracies across tested subjects
    if top1_scores: # Check if the list is not empty
        mean_top1 = np.mean(top1_scores)
        mean_top3 = np.mean(top3_scores)
        mean_top5 = np.mean(top5_scores)
        print(f"\nMean Top1 Accuracy: {mean_top1:.6f}")
        print(f"Mean Top3 Accuracy: {mean_top3:.6f}")
        print(f"Mean Top5 Accuracy: {mean_top5:.6f}")

        # Log mean results to wandb if training was done
        if t_config.Train and wandb_run:
             wandb_run.log({
                 "test/mean_top-1": mean_top1,
                 "test/mean_top-3": mean_top3,
                 "test/mean_top-5": mean_top5,
             })


    # --- Finish Wandb Run ---
    if t_config.Train and wandb_run:
        wandb_run.finish()
        print("Wandb run finished.")