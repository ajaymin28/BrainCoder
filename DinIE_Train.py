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
# import copy
# import random
import os
import itertools
import torch.optim as optim
# import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from utils.losses import SigLIPLoss, DINOLoss, update_teacher
from utils.common import TrainConfig, config_to_dict
from utils.DINODatasets import DINOV2EEGDataset
from utils.augment import EEGAug
# from utils.DINODatasets import DINOV2EEGDatasetKaggle
# from archs.EEGDinoEncoder import DINOV2EEGEncoder, DynamicEEG2DEncoder
from archs.EEG1DAutoEncoder import TSConv1DResNet
from archs.nice import Proj_img, Proj_eeg
from archs.DinIE import MultiModalEncoder, DINOHead
from utils.logman import logger
from utils.gpu_utils import memory_stats
from tqdm import tqdm
import wandb

from utils.dist_utils import is_ddp, setup_ddp
from utils.dist_utils import get_world_size, cleanup_ddp

OUTPUT_DIR = "/home/ja882177/EEG/gits/BrainCoder/checkpoints/sub1/single_sesssion_aug_only_dinoloss_localcrop_update"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

        batch_size = 2048
        epochs = 100

        GLOBAL_CROPS = 2
        LOCAL_CROPS = 4
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
        teacher_temp=0.1
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
    # img_proj = Proj_img(embedding_dim=768,proj_dim=768).to(device)
    # eeg_proj = Proj_eeg(embedding_dim=1696,proj_dim=768).to(device)

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
    args.dnn = "dinov2"
    args.WANDB_PROJECT_NAME = "DinoV2EEG_MultiSub"

    dataset = DINOV2EEGDataset(
        args=args,
        subject_ids=[1],  # or [1] or up to [1,2,...,10]
        session_ids=[0,1,2],    # select sessions you want for train only 4 session are avialable for Things EEG2
        subset="train",
        n_global_crops=HyperParams.GLOBAL_CROPS-1,
        n_local_crops=HyperParams.LOCAL_CROPS+1,
        transform_global=None,
        transform_local=None,
        max_cache_size=5
    )

    global_aug = EEGAug(noise_std=0.05, ft_surrogate_prob=1.0, min_len=200, max_len=230).cuda()
    local_aug = EEGAug(noise_std=0.09, ft_surrogate_prob=1.0, min_len=180, max_len=200).cuda()
    global_aug.eval()
    local_aug.eval()


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

    # SigLip Loss
    siglip_loss = SigLIPLoss().cuda()
    
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

    optimizer = optim.AdamW(itertools.chain(model_student.parameters(),
                                            # eeg_proj.parameters(),
                                            # img_proj.parameters(),
                                            # siglip_loss.parameters()
                                            ), lr=HyperParams.learning_rate)

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
        running_loss_global =  0
        running_contrastive_loss_global = 0
        running_siglip_loss_global = 0
        for it, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            eeg = batch['eeg'].cuda(non_blocking=True)
            image_features = batch['image_features'].cuda(non_blocking=True)
            it_global = epoch * len(dataloader) + it

            # Update optimizer LR and WD
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it_global]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[it_global]

            # crops = [c.cuda() for c in crops]
            # crops = [c.cuda(non_blocking=True) for c in crops]

            # global_crops = crops[:2]
            # local_crops = crops
            # for i in range(HyperParams.GLOBAL_CROPS):
            #     global_crops.append(crops[:2])
            # for j in range(HyperParams.LOCAL_CROPS):
            #     local_crops.append(crops[j])     

            global_crops = []
            local_crops = []

            for i in range(HyperParams.GLOBAL_CROPS):
                global_crops.append(global_aug(eeg).type(torch.cuda.FloatTensor))
            for j in range(HyperParams.LOCAL_CROPS):
                local_crops.append(local_aug(eeg).type(torch.cuda.FloatTensor))     

            running_contrastive_loss = 0
            running_siglip_loss = 0
            if HyperParams.MIXED_PREC_TRAINING:
                with torch.amp.autocast(device_type="cuda"):
                    teacher_out, t_base_feats = model_teacher(global_crops, ['eeg'] * len(global_crops))  # only global crops
                    student_out, s_base_feats = model_student(local_crops,  ['eeg'] * len(local_crops))
                    loss = dino_loss(student_out, teacher_out, epoch)

                    # image_features = img_proj(image_features)
                    # for tb_f in t_base_feats:
                    #     # running_contrastive_loss += model_student.nice_contrastive_loss(tb_f, image_features)
                    #     tb_f = eeg_proj(tb_f)
                    #     running_siglip_loss += siglip_loss(tb_f, image_features)
                    # # for sb_f in s_base_feats:
                    # #     running_contrastive_loss += model_student.nice_contrastive_loss(sb_f, image_features)
                    # #     # running_siglip_loss += siglip_loss(sb_f, image_features)
                    
                    # # running_contrastive_loss = running_contrastive_loss/len(t_base_feats)
                    # running_siglip_loss= running_siglip_loss/len(t_base_feats)
                    # loss = loss + running_siglip_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model_student.parameters(), HyperParams.clip_grad)
                cancel_gradients_last_layer(epoch, model_student, HyperParams.freeze_last_layer)  # for freeze_last_layer epochs cancels the gradient for last layer of dino head
                scaler.step(optimizer)
                scaler.update()
            else:    
                teacher_out, t_base_feats = model_teacher(global_crops, ['eeg'] * len(global_crops))  # only global crops
                student_out, s_base_feats = model_student(local_crops,  ['eeg'] * len(local_crops))
                loss = dino_loss(student_out, teacher_out, epoch)

                # image_features = img_proj(image_features)
                # for tb_f in t_base_feats:
                #     # running_contrastive_loss += model_student.nice_contrastive_loss(tb_f, image_features)
                #     tb_f = eeg_proj(tb_f)
                #     running_siglip_loss += siglip_loss(tb_f, image_features)
                # # for sb_f in s_base_feats:
                # #     running_contrastive_loss += model_student.nice_contrastive_loss(sb_f, image_features)
                # #     # running_siglip_loss += siglip_loss(sb_f, image_features)

                # # running_contrastive_loss = running_contrastive_loss/len(t_base_feats)
                # # loss = loss + running_contrastive_loss
                # running_siglip_loss = running_siglip_loss/len(t_base_feats)
                # loss = loss + running_siglip_loss

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

            running_loss_global += loss.item()
            # running_contrastive_loss_global += running_contrastive_loss.item()
            # running_siglip_loss_global += running_siglip_loss.item()

            
            # if local_rank == 0:
            #     memory_stats()

        if local_rank == 0:
            mem_stat_dict = memory_stats(get_dict=True)

            running_loss_global = running_loss_global/len(dataloader)
            # running_contrastive_loss_global = running_contrastive_loss_global/len(dataloader)
            # running_siglip_loss_global = running_siglip_loss_global/len(dataloader)
            # siglip_params = siglip_loss.get_params()

            # logger.info(f"Epoch {epoch}: Loss {loss.item():.4f} contrastive loss: {running_contrastive_loss.item():.4f}")
            # logger.info(f"Epoch {epoch}: Loss {running_loss_global:.4f} siglip loss: {running_siglip_loss_global:.4f}")
            logger.info(f"Epoch {epoch}: Loss {running_loss_global:.4f}")

            wandb.log({
                    "Epoch": epoch if epoch>0 else 1,
                
                    "train/loss": running_loss_global,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/teacher_temp": dino_loss.getCurrentTemp(epoch),
                    # "train/contrastive": running_contrastive_loss_global,
                    # "train/siglip_temp": siglip_params["temp"],
                    # "train/siglip_bias": siglip_params["bias"],
                    # "train/siglip_loss": running_siglip_loss_global,

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


        # check loss on every epoch, in 5-10 epochs lot of jump happens 
        if (not is_ddp() and epoch % 1 == 0) or (is_ddp() and local_rank == 0 and epoch % 1 == 0):

            if running_loss_global < best_loss:
                best_loss = running_loss_global

                student_to_save = model_student.module if hasattr(model_student, 'module') else model_student
                teacher_to_save = model_teacher.module if hasattr(model_teacher, 'module') else model_teacher

                checkpoint = {
                    'epoch': epoch,
                    'model_student': student_to_save.state_dict(),
                    'model_teacher': teacher_to_save.state_dict(),
                    # 'img_proj': img_proj.state_dict(),
                    # 'eeg_proj': eeg_proj.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }
                torch.save(checkpoint, f"{OUTPUT_DIR}/dinov2_ckpt_epoch{epoch}.pth")
                logger.info(f"Checkpoint saved at epoch {epoch}")

    checkpoint = {
        'epoch': epoch,
        'model_student': student_to_save.state_dict(),
        'model_teacher': teacher_to_save.state_dict(),
        # 'img_proj': img_proj.state_dict(),
        # 'eeg_proj': eeg_proj.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
    }
    torch.save(checkpoint, f"{OUTPUT_DIR}/dinov2_ckpt_epoch{epoch+1}_final.pth")
    cleanup_ddp()

if __name__ == '__main__':
    memory_stats()
    if is_ddp():
        logger.info("Running in DDP mode")
        # torchrun/SLURM/launch
        main_worker()
    else:
        # vanilla single-GPU or DataParallel
        logger.info("Running in none DDP mode")
        main_worker()