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

Author: OpenAI ChatGPT, Jaimin Bhoi request (I'm just gonna keep it this way, if something blows up you know whom to blame)
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
from utils.DINODatasets import DINOV2EEGDataset, dummy_dino_global_aug, dummy_dino_local_aug

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


# --- EEG Encoder with global pooling ---
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        reduction = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        attn = self.avg_pool(x)   # (B, C, 1)
        attn = self.fc(attn)      # (B, C, 1)
        return x * attn

class TemporalAttention(nn.Module):
    def __init__(self, in_time, reduction=8):
        super().__init__()
        reduction = max(1, in_time // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_time, reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduction, in_time, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_perm = x.permute(0, 2, 1)    # (B, T, C)
        attn = self.avg_pool(x_perm)   # (B, T, 1)
        attn = self.fc(attn)           # (B, T, 1)
        out = x_perm * attn            # (B, T, C)
        return out.permute(0, 2, 1)

class EEGEncoder(nn.Module):
    def __init__(self, proj_dim=768, drop_proj=0.5, use_layernorm=True):
        super().__init__()
        self.use_layernorm = use_layernorm
        # Flexible convs, attention
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Conv1d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )
        self.channel_attn = ChannelAttention(32)
        self.temporal_attn = TemporalAttention(32)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_head = nn.Sequential(
            nn.Linear(32, proj_dim),
            nn.BatchNorm1d(proj_dim),
            ResidualAdd(nn.Sequential(
                nn.SiLU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj)
            )),
        )
        if self.use_layernorm:
            self.ln = nn.LayerNorm(proj_dim)
    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = x.reshape(B, 1, C*T)
        x = self.spatial_conv(x)  # (B, 32, C*T)
        x = x.view(B, 32, -1)
        x = self.channel_attn(x)
        x = self.temporal_attn(x)
        x = self.global_pool(x).squeeze(-1)  # (B, 32)
        features = self.feature_head(x)      # (B, proj_dim)
        if self.use_layernorm:
            features = self.ln(features)
        return features

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
    # Copied from timm
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor



def trunc_normal_(tensor, mean=0., std=1.):
    # Timm/trunc_normal compatible with recent PyTorch
    with torch.no_grad():
        return tensor.normal_(mean=mean, std=std)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
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
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

# class DINOHead(nn.Module):
#     def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
#         super().__init__()
#         nlayers = max(nlayers, 1)
#         if nlayers == 1:
#             self.mlp = nn.Linear(in_dim, bottleneck_dim)
#         else:
#             layers = [nn.Linear(in_dim, hidden_dim)]
#             if use_bn:
#                 layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.GELU())
#             for _ in range(nlayers - 2):
#                 layers.append(nn.Linear(hidden_dim, hidden_dim))
#                 if use_bn:
#                     layers.append(nn.BatchNorm1d(hidden_dim))
#                 layers.append(nn.GELU())
#             layers.append(nn.Linear(hidden_dim, bottleneck_dim))
#             self.mlp = nn.Sequential(*layers)
#         self.apply(self._init_weights)
#         self.last_layer = torch.nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
#         self.last_layer.weight.data.fill_(1)
#         if norm_last_layer:
#             self.last_layer.weight.requires_grad = False

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.mlp(x)
#         x = nn.functional.normalize(x, dim=-1, p=2)
#         x = self.last_layer(x)
#         return x

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

# --- DINOv2 Loss ---
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp=0.04, teacher_temp=0.07,
                 warmup_epochs=30, nepochs=100, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
            np.ones(nepochs - warmup_epochs) * teacher_temp
        ))
        self.ncrops = ncrops
    def forward(self, student_output, teacher_output, epoch):
        student_out = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_output]
        teacher_temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule)-1)]
        teacher_out = [F.softmax((t - self.center) / teacher_temp, dim=-1).detach() for t in teacher_output]
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v, s in enumerate(student_out):
                if v == iq:
                    continue
                loss = torch.sum(-q * s, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(torch.cat(teacher_output, dim=0))
        return total_loss
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# --- EMA Teacher Update ---
def update_teacher(student, teacher, momentum=0.996):
    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
        param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

# --- Data Parallel/Distributed Setup ---
def setup_distributed():
    if 'RANK' in os.environ:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend='nccl')

# --- Training Loop ---
def train_dino(model_student, model_teacher, dino_loss, dataloader, optimizer, device, 
              epochs=100, ncrops=6, local_rank=0, is_ddp=False, use_amp=True):
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    for epoch in range(epochs):
        model_student.train()
        model_teacher.eval()
        if is_ddp:
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            # batch: List[Tensor], each is a crop (B, ...)
            crops = batch['crops']  # list of tensors
            crop_types = batch['crop_types'] if 'crop_types' in batch else None
            # Forward student and teacher
            with torch.cuda.amp.autocast(enabled=use_amp):
                student_out = model_student(crops, crop_types)  # list of (B, D)
                with torch.no_grad():
                    teacher_out = model_teacher(crops[:2], crop_types[:2] if crop_types else None)  # only global crops
                loss = dino_loss(student_out, teacher_out, epoch)
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            update_teacher(model_student, model_teacher)
        if local_rank == 0 or not is_ddp:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")



def load_model_for_inference(ckpt_path, eeg_encoder, img_encoder, device='cuda'):
    # Load your encoders
    model = MultiModalEncoder(eeg_encoder, img_encoder).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_student'])
    model.eval()
    return model

def extract_features(model, input_data, input_type='eeg', batch_size=32, device='cuda'):
    """
    input_data: torch.Tensor or list of tensors (EEG: [B, C, T], IMG: [B, 3, H, W])
    input_type: 'eeg' or 'img'
    Returns: features [N, feat_dim]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eeg_encoder = EEGEncoder(proj_dim=768).to(device)
    img_encoder = ImageEncoder(proj_dim=768).to(device)
    model = load_model_for_inference('dinov2_ckpt_epoch90.pth', eeg_encoder, img_encoder, device)

    # Suppose you have eeg_data: [N, C, T]
    eeg_features = extract_features(model, eeg_data, input_type='eeg', batch_size=64, device=device)
    print(eeg_features.shape)  # (N, 768)

    img_features = extract_features(model, img_data, input_type='img', batch_size=64, device=device)
    print(img_features.shape)  # (N, 768)


    """
    features = []
    model.eval()
    with torch.no_grad():
        if isinstance(input_data, list):
            # list of tensors (chunks/batches)
            for crop in input_data:
                crop = crop.to(device)
                out = model([crop], [input_type])[0]  # [B, feat_dim]
                features.append(out.cpu())
        else:
            # single big tensor, slice in batches
            N = input_data.shape[0]
            for i in range(0, N, batch_size):
                batch = input_data[i:i+batch_size].to(device)
                out = model([batch], [input_type])[0]
                features.append(out.cpu())
    features = torch.cat(features, dim=0)
    return features  # [N, feat_dim]


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

    eeg_encoder = EEGEncoder(proj_dim=768).to(device)
    # img_encoder = ImageEncoder(proj_dim=768).to(device)

    # shared head
    dino_head = DINOHead(
        in_dim=768,         # Feature dim from both encoders
        out_dim=768,        # Embedding dim for DINO loss
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256
    )


    model_student = MultiModalEncoder(eeg_encoder, img_encoder=None, dino_head=dino_head).to(device)
    model_teacher = MultiModalEncoder(eeg_encoder, img_encoder=None, dino_head=dino_head).to(device)

    # DDP wrap if needed
    if is_ddp():
        model_student = DDP(model_student, device_ids=[local_rank], output_device=local_rank)
        model_teacher = DDP(model_teacher, device_ids=[local_rank], output_device=local_rank)

    # Dataset, Sampler, DataLoader
    args = TrainConfig()

    args.global_config.WANDB_PROJECT_NAME = "DinIE"

    args.batch_size = 4
    args.learning_rate = 5e-4
    args.local_epochs = 100


    dataset = DINOV2EEGDataset(
        args=args,
        subject_ids=[1,2],  # or [1] or up to [1,2,...,10]
        session_ids=[0,1,2,3],    # select sessions you want for train only 4 session are avialable
        subset="train",
        n_global_crops=2,
        n_local_crops=6,
        transform_global=dummy_dino_global_aug,
        transform_local=dummy_dino_local_aug,
        max_cache_size=2
    )
    
    if is_ddp():
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Loss, optimizer
    dino_loss = DINOLoss(out_dim=768, ncrops=6).to(device)
    optimizer = optim.AdamW(model_student.parameters(), lr=args.learning_rate)

    scaler = torch.amp.GradScaler()
    epochs = args.local_epochs

    for epoch in range(epochs):

        if is_ddp():
            dataloader.sampler.set_epoch(epoch)
        model_student.train()
        model_teacher.eval()

        for batch in dataloader:
            crops = batch['crops']

            crop_types = ['eeg'] * crops.size(0) # all eeg here for img pass img string for respective sample index

            # crop_types = batch['crop_types']

            with torch.amp.autocast():
                student_out = model_student(crops, crop_types)
                with torch.no_grad():
                    teacher_out = model_teacher(crops[:2], crop_types[:2])  # only global crops
                loss = dino_loss(student_out, teacher_out, epoch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            update_teacher(model_student, model_teacher)

            # Reduce loss for logging if DDP
            if is_ddp():
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / dist.get_world_size()

        if local_rank == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

        if (not is_ddp() and epoch % 10 == 0) or (is_ddp() and local_rank == 0 and epoch % 10 == 0):
            student_to_save = model_student.module if hasattr(model_student, 'module') else model_student
            checkpoint = {
                'epoch': epoch,
                'model_student': student_to_save.state_dict(),
                'model_teacher': (model_teacher.module if hasattr(model_teacher, 'module') else model_teacher).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }
            torch.save(checkpoint, f"dinov2_ckpt_epoch{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")

    cleanup_ddp()

if __name__ == '__main__':
    if is_ddp():
        # torchrun/SLURM/launch
        main_worker()
    else:
        # vanilla single-GPU or DataParallel
        main_worker()


# # --- Usage Example (single GPU) ---
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # EEG crop dummy: (B, C, T) random
#     eeg_dataset = [torch.randn(32, random.randint(16, 63), random.randint(50, 250)) for _ in range(1000)]
#     # Image crop dummy: (B, 3, H, W) random
#     img_dataset = [torch.randn(32, 3, random.randint(64, 224), random.randint(64, 224)) for _ in range(1000)]
#     # Example: 2 global EEG, 0 global IMG, 2 local EEG, 2 local IMG



#     dataset = MultiModalDinoDataset(eeg_dataset, img_dataset, global_eeg=2, global_img=0, local_eeg=2, local_img=2)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

#     eeg_encoder = EEGEncoder(proj_dim=768)
#     img_encoder = ImageEncoder(proj_dim=768)
#     model_student = MultiModalEncoder(eeg_encoder, img_encoder).to(device)
#     model_teacher = copy.deepcopy(model_student).to(device)
#     dino_loss = DINOLoss(out_dim=768, ncrops=6)
#     optimizer = torch.optim.AdamW(model_student.parameters(), lr=5e-4)
#     train_dino(model_student, model_teacher, dino_loss, dataloader, optimizer, device, epochs=2)

#     # For multi-GPU/Slurm: see DDP setup/distributed notes above.
#     # For real training, use your real EEG and image dataloaders and set crop numbers as needed.
