"""
DINOv2 Multimodal (EEG + Image) Training Pipeline
-------------------------------------------------
Supports flexible crop configurations: EEG+EEG, EEG+IMG, IMG+IMG, etc.
Includes all essentials:
  - EEG encoder (your design, with global pooling)
  - Image encoder (ViT for real use, can swap for ResNet)
  - Shared projection head
  - DINOv2 loss (with centering, temperature, teacher scheduling)
  - Student/Teacher EMA update
  - Multi-crop, multi-modal batch handling
  - Single/multi-GPU/Slurm friendly

Author: OpenAI ChatGPT, Jaimin Bhoi request
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
import os
from timm.models.vision_transformer import vit_base_patch16_224

import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils.common import TrainConfig
from utils.DINODatasets import DINOV2EEGDataset,DINOV2EEGDatasetKaggle, eeg_global_aug, eeg_local_aug
from archs.EEGDinoEncoder import DINOV2EEGEncoder, DynamicEEG2DEncoder
from archs.EEG1DAutoEncoder import TSConv1DResNet
from utils.logman import logger
from utils.gpu_utils import memory_stats
from tqdm import tqdm
import wandb

# def load_model_for_inference(ckpt_path, eeg_encoder, img_encoder, device='cuda'):
#     # Load your encoders
#     model = MultiModalEncoder(eeg_encoder, img_encoder).to(device)
#     ckpt = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(ckpt['model_student'])
#     model.eval()
#     return model

# def extract_features(model, input_data, input_type='eeg', batch_size=32, device='cuda'):
#     """
#     input_data: torch.Tensor or list of tensors (EEG: [B, C, T], IMG: [B, 3, H, W])
#     input_type: 'eeg' or 'img'
#     Returns: features [N, feat_dim]

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     eeg_encoder = EEGEncoder(proj_dim=768).to(device)
#     img_encoder = ImageEncoder(proj_dim=768).to(device)
#     model = load_model_for_inference('dinov2_ckpt_epoch90.pth', eeg_encoder, img_encoder, device)

#     # Suppose you have eeg_data: [N, C, T]
#     eeg_features = extract_features(model, eeg_data, input_type='eeg', batch_size=64, device=device)
#     print(eeg_features.shape)  # (N, 768)

#     img_features = extract_features(model, img_data, input_type='img', batch_size=64, device=device)
#     print(img_features.shape)  # (N, 768)


#     """
#     features = []
#     model.eval()
#     with torch.no_grad():
#         if isinstance(input_data, list):
#             # list of tensors (chunks/batches)
#             for crop in input_data:
#                 crop = crop.to(device)
#                 out = model([crop], [input_type])[0]  # [B, feat_dim]
#                 features.append(out.cpu())
#         else:
#             # single big tensor, slice in batches
#             N = input_data.shape[0]
#             for i in range(0, N, batch_size):
#                 batch = input_data[i:i+batch_size].to(device)
#                 out = model([batch], [input_type])[0]
#                 features.append(out.cpu())
#     features = torch.cat(features, dim=0)
#     return features  # [N, feat_dim]


def config_to_dict(obj):
    # Get all relevant attributes (class + instance, skip dunder and callables)
    keys = [
        k for k in set(obj.__class__.__dict__.keys()).union(vars(obj).keys())
        if not k.startswith("__") and not callable(getattr(obj, k, None))
    ]
    def to_serializable(v):
        if isinstance(v, torch.device):
            return str(v)
        return v
    return {k: to_serializable(getattr(obj, k)) for k in keys}


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



# --- Good Image Encoder (ViT, DINOv2 friendly) ---
class ImageEncoder(nn.Module):
    def __init__(self, proj_dim=768, vit_ckpt=None):
        super().__init__()
        # Use ViT-Base (DINOv2's default backbone). Requires timm
        self.vit = vit_base_patch16_224(pretrained=True if vit_ckpt is None else False)
        if vit_ckpt is not None:
            state = torch.load(vit_ckpt, map_location='cpu')
            self.vit.load_state_dict(state, strict=False)
        self.proj = nn.Sequential(
            nn.Linear(self.vit.embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
    def forward(self, x):
        # x: (B, 3, H, W), any H/W
        features = self.vit.forward_features(x)  # (B, embed_dim)
        features = self.proj(features)
        return features

def trunc_normal_(tensor, mean=0., std=1.):
    # Timm/trunc_normal compatible with recent PyTorch
    with torch.no_grad():
        return tensor.normal_(mean=mean, std=std)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=65536):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # weight_norm wrapper
        self.last_layer = torch.nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        # Fill only the weight_g parameter with 1 (per DINO)
        if hasattr(self.last_layer, "weight_g"):
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

# --- Shared Multi-modal Wrapper ---
class MultiModalEncoder(nn.Module):
    def __init__(self, eeg_encoder, img_encoder, dino_head):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.img_encoder = img_encoder
        self.dino_head = dino_head

    def forward(self, crops, crop_types=None):
        outs = []
        for i, crop in enumerate(crops):
            ctype = None
            if crop_types is not None:
                ctype = crop_types[i]
            else:
                if crop.dim() == 3:
                    ctype = 'eeg'
                elif crop.dim() == 4 and crop.shape[1] == 3:
                    ctype = 'img'
                else:
                    raise ValueError(f"Unknown crop shape: {crop.shape}")
            if ctype == 'eeg':
                feat = self.eeg_encoder(crop)
            elif ctype == 'img':
                feat = self.img_encoder(crop)
            else:
                raise ValueError(f"Unknown crop type: {ctype}")
            out = self.dino_head(feat)
            outs.append(out)
        return outs

# --- DINO Loss ---
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp=0.04, teacher_temp=0.07,
                 warmup_epochs=30, nepochs=100, student_temp=0.1, center_momentum=0.996):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
            np.ones(nepochs - warmup_epochs) * teacher_temp
        ))
        self.ncrops = ncrops
    
    def getCurrentTemp(self, epoch):
        return self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule)-1)]
    
    def forward(self, student_output, teacher_output, epoch):

        # student_out = [so / self.student_temp for so in student_output]
        student_out = torch.stack(student_output)
        student_out = student_out / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        teacher_temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule)-1)]
        # teacher_out = [F.softmax((tout - self.center) / teacher_temp, dim=-1).detach() for tout in teacher_output]
        teacher_out = torch.stack(teacher_output)
        teacher_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1).detach()
        teacher_out = teacher_out.chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v, s in enumerate(student_out):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(s, dim=-1), dim=-1).mean()
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        teacher_output = torch.stack(teacher_output, dim=0)
        teacher_output = teacher_output.view(-1, teacher_output.size(-1))
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        # batch_center = batch_center / len(teacher_output)
        if torch.isnan(batch_center).any():
            logger.info("NaN detected in batch_center. Skipping center update!")
            return
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)



# --- EMA Teacher Update ---
@torch.no_grad()
def update_teacher(student, teacher, momentum=0.996):
    student_mod = student.module if hasattr(student, 'module') else student
    teacher_mod = teacher.module if hasattr(teacher, 'module') else teacher
    with torch.no_grad():
        for param_q, param_k in zip(student_mod.parameters(), teacher_mod.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

# --- Data Parallel/Distributed Setup ---
def setup_distributed():
    if 'RANK' in os.environ:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend='nccl')

def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

import torch.nn.init as init
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def main_worker():

    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Model

    """
    [FOR EEG + EEG]
    eeg_encoder = EEGEncoder(proj_dim=768)
    img_encoder = None  # Not used, but required by init
    model = MultiModalEncoder(eeg_encoder, img_encoder=None)  # img_encoder can be None, just don't pass any 'img' crops

    # Example: 2 global, 4 local EEG crops per sample
    crops = [torch.randn(B, random.randint(16, 63), random.randint(50, 250)) for _ in range(6)]
    crop_types = ['eeg'] * 6
    out = model(crops, crop_types)

    [for img + eeg]
    eeg_encoder = EEGEncoder(proj_dim=768).to(device)
    img_encoder = ImageEncoder(proj_dim=768).to(device)
    model_student = MultiModalEncoder(eeg_encoder, img_encoder).to(device)
    """

    class HyperParams:

        batch_size = 512
        epochs = 100

        GLOBAL_CROPS = 2
        LOCAL_CROPS = 6
        MIXED_PREC_TRAINING = True
        freeze_last_layer = 1
        clip_grad = 1.0

        DINO_Head_Dim = 32768
        DINO_Head_bottleneck_dim = 768
        DINO_Head_hidden_dim = 2048
        DINO_Head_nlayers = 3
        DINO_Head_norm_last_layer = False  # in orginal False with vit_small and True with vit_base.
        DINO_Head_use_bn = False

        # learning_rate = 5e-4
        learning_rate = 5e-4
        learning_rate_warmpup_epochs = 10
        learning_rate_final = 1e-6

        warmup_teacher_temp=0.04
        teacher_temp=0.04
        warmup_teacher_temp_epochs=30  # 0.07 --> 0.2 over 30 epochs
        student_temp=0.2
        center_momentum=0.996   # higher for smaller batches i.e 0.9995 scheduled to being 1.0 by end of the training

        wd_base_value=0.04
        wd_final_value=0.4
    
        momentum_base_value=0.996
        momentum_final_value=1.0


    # eeg_encoder = EEGEncoder(proj_dim=768).to(device)
    teacher_eeg_encoder = TSConv1DResNet(proj_dim=1696)
    student_eeg_encoder = TSConv1DResNet(proj_dim=1696)

    # teacher_eeg_encoder = DINOV2EEGEncoder(proj_dim=768)  # or your DINO head dimension
    # student_eeg_encoder = DINOV2EEGEncoder(proj_dim=768)  # or your DINO head dimension
    # img_encoder = ImageEncoder(proj_dim=768).to(device)

    

    # head
    # default args in dino (in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256)
    teacher_dino_head = DINOHead(
        in_dim=1696,                   # Feature dim from both encoders
        out_dim=HyperParams.DINO_Head_Dim,        # Embedding dim for DINO loss
        use_bn=HyperParams.DINO_Head_use_bn,
        norm_last_layer=HyperParams.DINO_Head_norm_last_layer,
        nlayers=HyperParams.DINO_Head_nlayers,
        hidden_dim=HyperParams.DINO_Head_hidden_dim,
        bottleneck_dim=HyperParams.DINO_Head_bottleneck_dim
    )

    student_dino_head = DINOHead(
        in_dim=1696,                   # Feature dim from both encoders
        out_dim=HyperParams.DINO_Head_Dim,        # Embedding dim for DINO loss
        use_bn=HyperParams.DINO_Head_use_bn,
        norm_last_layer=HyperParams.DINO_Head_norm_last_layer,
        nlayers=HyperParams.DINO_Head_nlayers,
        hidden_dim=HyperParams.DINO_Head_hidden_dim,
        bottleneck_dim=HyperParams.DINO_Head_bottleneck_dim
    )

    from utils.losses import cosine_scheduler

    model_student = MultiModalEncoder(teacher_eeg_encoder, img_encoder=None, dino_head=student_dino_head).to(device)
    model_teacher = MultiModalEncoder(student_eeg_encoder, img_encoder=None, dino_head=teacher_dino_head).to(device)

    weights_init_normal(model_student)
    weights_init_normal(model_teacher)

    for p in model_student.parameters():
        p.requires_grad = True

    # # Initialize teacher with student's weights
    model_teacher.load_state_dict(model_student.state_dict())

    # Set teacher to eval mode and disable gradients
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    # DDP wrap if needed
    if is_ddp():
        model_student = DDP(model_student, device_ids=[local_rank], output_device=local_rank)
        model_teacher = DDP(model_teacher, device_ids=[local_rank], output_device=local_rank)

    # Dataset, Sampler, DataLoader
    args = TrainConfig()
    args.WANDB_PROJECT_NAME = "DinoV2EEG_MultiSub"

    dataset = DINOV2EEGDataset(
        args=args,
        subject_ids=[1],  # or [1] or up to [1,2,...,10]
        session_ids=[0,1,2,3],    # select sessions you want for train only 4 session are avialable for Things EEG2
        subset="train",
        n_global_crops=HyperParams.GLOBAL_CROPS-1,
        n_local_crops=HyperParams.LOCAL_CROPS+1,
        transform_global=eeg_global_aug,
        transform_local=eeg_local_aug,
        max_cache_size=5
    )

    if is_ddp():
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=HyperParams.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=HyperParams.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Loss, optimizer
    dino_loss = DINOLoss(out_dim=HyperParams.DINO_Head_Dim, 
                         ncrops=HyperParams.LOCAL_CROPS,
                         warmup_teacher_temp=HyperParams.warmup_teacher_temp,
                         teacher_temp=HyperParams.teacher_temp,
                         warmup_epochs=HyperParams.warmup_teacher_temp_epochs,
                         nepochs=HyperParams.epochs,
                         student_temp=HyperParams.student_temp,
                         center_momentum=HyperParams.center_momentum
                         ).to(device)
    
    # Initialize your schedulers (call this after defining your args/data_loader)
    lr_schedule = cosine_scheduler(
        base_value=HyperParams.learning_rate * (HyperParams.batch_size * get_world_size() ) / 256.,
        final_value=HyperParams.learning_rate_final, # 0.001 | 0.0005 -> 0.000001
        epochs=HyperParams.epochs,
        niter_per_epoch=len(dataloader),
        warmup_epochs=HyperParams.learning_rate_warmpup_epochs,
        start_warmup_value=0  # can be set to args.start_warmup_lr if you wish
    )

    wd_schedule = cosine_scheduler(
        base_value=HyperParams.wd_base_value,
        final_value=HyperParams.wd_final_value,
        epochs=HyperParams.epochs,
        niter_per_epoch=len(dataloader),
    )

    momentum_schedule = cosine_scheduler(
        base_value=HyperParams.momentum_base_value,
        final_value=HyperParams.momentum_final_value,
        epochs=HyperParams.epochs,
        niter_per_epoch=len(dataloader),
    )

    optimizer = optim.AdamW(model_student.parameters(), lr=HyperParams.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    if local_rank == 0:
        wandb.init(
            project=args.WANDB_PROJECT_NAME,          # your project name
            # mode="offline"
            config=config_to_dict(HyperParams()),             # log all config parameters
            notes=""""""
        )


    if local_rank == 0:
        memory_stats()

    best_loss = float('inf')
    it_global = 0
    for epoch in range(HyperParams.epochs):

        if is_ddp():
            dataloader.sampler.set_epoch(epoch)
        model_student.train()
        model_teacher.eval()

        torch.cuda.reset_peak_memory_stats("cuda") 
        for it, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            crops = batch['crops']   # 1 + ncrops

            it_global = epoch * len(dataloader) + it

            # Update optimizer LR and WD
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it_global]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[it_global]

            # crops = [c.cuda() for c in crops]
            crops = [c.cuda(non_blocking=True) for c in crops]
            global_crops = crops[:2]
            local_crops = crops

            # for i in range(HyperParams.GLOBAL_CROPS):
            #     global_crops.append(crops[:2])
            # for j in range(HyperParams.LOCAL_CROPS):
            #     local_crops.append(crops[j])     

            if HyperParams.MIXED_PREC_TRAINING:
                with torch.amp.autocast(device_type="cuda"):
                    teacher_out = model_teacher(global_crops, ['eeg'] * len(global_crops))  # only global crops
                    student_out = model_student(local_crops,  ['eeg'] * len(local_crops))
                    loss = dino_loss(student_out, teacher_out, epoch)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model_student.parameters(), HyperParams.clip_grad)
                cancel_gradients_last_layer(epoch, model_student, HyperParams.freeze_last_layer)  # for freeze_last_layer epochs cancels the gradient for last layer of dino head
                scaler.step(optimizer)
                scaler.update() 
            else:    
                teacher_out = model_teacher(global_crops, ['eeg'] * len(global_crops))  # only global crops
                student_out = model_student(local_crops,  ['eeg'] * len(local_crops))
                loss = dino_loss(student_out, teacher_out, epoch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_student.parameters(), HyperParams.clip_grad)
                cancel_gradients_last_layer(epoch, model_student, HyperParams.freeze_last_layer)
                optimizer.step()

            m = momentum_schedule[it_global]
            update_teacher(model_student, model_teacher,momentum=m)

            # Reduce loss for logging if DDP
            if is_ddp():
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / dist.get_world_size()

            
            if local_rank == 0:
                memory_stats()

        if local_rank == 0:
            logger.info(f"Epoch {epoch}: Loss {loss.item():.4f}")
            # logger.info("Student out mean/std", student_out[0].mean().item(), student_out[0].std().item())
            # logger.info("Teacher out mean/std", teacher_out[0].mean().item(), teacher_out[0].std().item())
            # logger.info("LR", optimizer.param_groups[0]["lr"])
            # logger.info("Center mean", dino_loss.center.mean().item(), "std", dino_loss.center.std().item())

            mem_stat_dict = memory_stats(get_dict=True)

            wandb.log({
                    "Epoch": epoch if epoch>0 else 1,
                
                    "train/loss": loss,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/teacher_temp": dino_loss.getCurrentTemp(epoch),

                    "schedules/learning_rate": optimizer.param_groups[0]["lr"],
                    "schedules/teacher_momentum": m,
                    "schedules/weight_decay": optimizer.param_groups[0]["weight_decay"],
                    
                    "centers/center_mean": dino_loss.center.mean().item(),
                    "centers/center_std": dino_loss.center.std().item(),
                    "centers/student_mean": student_out[0].mean().item(),
                    "centers/student_std": student_out[0].std().item(),
                    "centers/teacher_mean": teacher_out[0].mean().item(),
                    "centers/teacher_std": teacher_out[0].std().item(),
                
                    "resources_stats/": mem_stat_dict
            })


        if (not is_ddp() and epoch % 10 == 0) or (is_ddp() and local_rank == 0 and epoch % 10 == 0):

            if loss < best_loss:
                best_loss = loss

                student_to_save = model_student.module if hasattr(model_student, 'module') else model_student
                teacher_to_save = model_teacher.module if hasattr(model_teacher, 'module') else model_teacher

                checkpoint = {
                    'epoch': epoch,
                    'model_student': student_to_save.state_dict(),
                    'model_teacher': teacher_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }
                torch.save(checkpoint, f"dinov2_ckpt_epoch{epoch}.pth")
                logger.info(f"Checkpoint saved at epoch {epoch}")

    checkpoint = {
        'epoch': epoch,
        'model_student': student_to_save.state_dict(),
        'model_teacher': teacher_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
    }
    torch.save(checkpoint, f"dinov2_ckpt_epoch{epoch+1}_final.pth")
    cleanup_ddp()

if __name__ == '__main__':
    if is_ddp():
        # torchrun/SLURM/launch
        main_worker()
    else:
        # vanilla single-GPU or DataParallel
        main_worker()



