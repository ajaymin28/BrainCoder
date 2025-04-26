import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
from utils.eeg_utils import EEG_Dataset2, get_eeg_data
from utils.eeg_utils import EEG_DATA_PATH,IMG_DATA_PATH,TEST_CENTER_PATH
from utils.common import CheckpointManager, RunManager, freeze_model, weights_init_normal
from utils.common import TrainConfig, clean_mem, memory_stats
# from archs.AE_Unet import SpatioTemporalEEGAutoencoder, EEGDataAugmentation

from archs.nice import Proj_img, Proj_eeg, Enc_eeg
from archs.CNNTrans import CNNTrans, SubjectDiscriminator
from archs.CNNTrans import grad_reverse
from torch.autograd import Variable
from utils.losses import get_contrastive_loss, orthogonality_loss
from torch.cuda.amp import autocast, GradScaler


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import os
import time
import itertools

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor


# highlight-next-line
wandb.login()

def setup(rank, world_size):
    # For single host, localhost is fine
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Any free port number
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # NCCL for GPU

def cleanup():
    dist.destroy_process_group()

def test(runId,models, subject_id, dnn="clip", batch_size=8, with_img_projection=False, with_eeg_projection=False, config:TrainConfig=None):
    _, _, test_eeg, test_label = get_eeg_data(eeg_data_path=EEG_DATA_PATH, nSub=subject_id, subset="test")

    model, model_img_proj, model_eeg_proj, subject_discriminator = models

    test_center = np.load(TEST_CENTER_PATH + 'center_' + dnn + '.npy', allow_pickle=True)
    test_eeg = torch.from_numpy(test_eeg)
    # test_img_feature = torch.from_numpy(test_img_feature)
    test_center = torch.from_numpy(test_center)
    test_label = torch.from_numpy(test_label)
    test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # model = SpatioTemporalEEGAutoencoder(latent_dim=latent_dim, dropout_rate=dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpm_AEnc_eeg = CheckpointManager(prefix="AEnc_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")
    # model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
    model = model.to(device)
    cpm_AEnc_eeg.load_checkpoint(model=model,optimizer=None,epoch="best", strict=True)

    # if with_img_projection:
    #     model_img_proj = Proj_img(embedding_dim=768,proj_dim=768)
    #     cpm_Proj_img = CheckpointManager(prefix="Proj_img",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")
    #     model_img_proj = model_img_proj.to(device)
    #     cpm_Proj_img.load_checkpoint(model=model_img_proj,optimizer=None,epoch="best")

    if with_eeg_projection:
        model_eeg_proj = Proj_eeg(embedding_dim=config.encoder_output_dim,proj_dim=768)
        model_eeg_proj = model_eeg_proj.to(device)
        cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")
        cpm_Proj_eeg.load_checkpoint(model=model_eeg_proj,optimizer=None,epoch="best", strict=True)

    all_center = test_center
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0


    with torch.no_grad():
        for i, (teeg, tlabel) in enumerate(test_dataloader):
            teeg = teeg.type(Tensor)
            tlabel = tlabel.type(LongTensor)
            all_center = all_center.type(Tensor)
            
            teeg = torch.mean(teeg,dim=1,keepdim=config.keepDimAfterAvg)

            Ec = model(teeg)
            if with_eeg_projection:
                Ec = model_eeg_proj(Ec)

            # tfea = Ec / Ec.norm(dim=1, keepdim=True)
            similarity = (100.0 * Ec @ all_center.t()).softmax(dim=-1)  # no use 100?
            _, indices = similarity.topk(5)

            tt_label = tlabel.view(-1, 1)
            total += tlabel.size(0)
            top1 += (tt_label == indices[:, :1]).sum().item()
            top3 += (tt_label == indices[:, :3]).sum().item()
            top5 += (tt_label == indices).sum().item()

        
        top1_acc = float(top1) / float(total)
        top3_acc = float(top3) / float(total)
        top5_acc = float(top5) / float(total)
    
    # print('Subject: [%d]  The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (subject_id,top1_acc, top3_acc, top5_acc))

    if config.log_test_data:
        wandb.log({
            "top-1": top1_acc,
            "top-3": top3_acc,
            "top-5": top5_acc,
        })
    
    
    return top1_acc, top3_acc, top5_acc


def do_forward(model, input, train=True):
    model = freeze_model(model, train=train)
    if not train:
        with torch.no_grad():
            return model(input)
    else:
        return model(input)



def config_to_dict(config_class):
    return {k: (v if not isinstance(v, torch.device) else str(v)) for k, v in vars(config_class).items() if not k.startswith("__")}


def compute_alpha(config: TrainConfig, data_loader_len):
    # Total steps across all iterations
    total_steps = config.Total_Subjects * config.global_epochs * config.local_epochs * data_loader_len
    
    # Calculate overall batch index that continuously increases
    overall_batch_idx = (
        (config.current_global_epoch - 1) * config.Total_Subjects * config.local_epochs * data_loader_len +
        (config.current_subject - 1) * config.local_epochs * data_loader_len +
        (config.current_epoch - 1) * data_loader_len +
        (config.batch_idx - 1)
    )
    
    # Progress ratio (0 to 1)
    p = float(overall_batch_idx) / total_steps
    
    # Compute alpha using the logistic function
    return 2. / (1. + np.exp(-5 * p)) - 1

# def compute_alpha(config:TrainConfig, data_loader_len):
#     # total_steps = config.global_epochs * config.local_epochs * data_loader_len
#     # p = float(config.batch_idx + (config.current_epoch * total_steps)) / (config.Total_Subjects * total_steps)
#     total_steps = config.Total_Subjects * config.global_epochs * config.local_epochs * data_loader_len
#     start_steps = config.current_subject * config.current_global_epoch * config.current_epoch * data_loader_len
#     p = float(config.batch_idx + start_steps) / total_steps
#     return 2. / (1. + np.exp(-3 * p)) - 1

def process_batch(models, 
                  optimizers,
                  config:TrainConfig, logit_scale, adv_criterion, dataloader_len, isvalidation=False):

    model, model_img_proj, model_eeg_proj, subject_discriminator = models
    optimizer, subject_optimizer = optimizers
    (eeg, image_features, cls_label_id, subid), \
    (eeg_2, image_features_2, cls_label_id2, subid2), \
    (eeg_neg, image_features_neg, cls_label_id_neg, subid1_neg) = config.train_batch_data

    eeg, eeg_2 = eeg.type(Tensor), eeg_2.type(Tensor) if config.Contrastive_augmentation else None
    image_features = image_features.cuda().type(Tensor)

    total_loss, contrastive_loss, adv_loss = None, None, None
    scaler = GradScaler()  # Mixed precision training scaler
    scaler_sub = GradScaler()  # Mixed precision training scaler

    with autocast():

        if not isvalidation:
            optimizer.zero_grad()
            if config.enable_adv_training:
                subject_optimizer.zero_grad()

        output0 = model(eeg)
        output1 = model(eeg_2) if config.Contrastive_augmentation else None
        if config.add_img_proj:
            image_features = model_img_proj(image_features)
        if config.add_eeg_proj:
            output0 = model_eeg_proj(output0)
            if output1 is not None:
                output1 = model_eeg_proj(output1)
        
        labels = torch.arange(eeg.shape[0]).cuda().type(LongTensor)
        contrastive_loss = get_contrastive_loss(output0, image_features, labels, logit_scale)
        if config.Contrastive_augmentation:
            contrastive_loss += get_contrastive_loss(output1, image_features, labels, logit_scale)
            # contrastive_loss /= 2

        
        if config.enable_adv_training and config.Contrastive_augmentation:
            alpha = compute_alpha(config,dataloader_len)
            config.alpha = alpha
            subject_pred0, _ = subject_discriminator(output0, alpha=alpha)
            subject_pred1, _ = subject_discriminator(output1, alpha=alpha)
            # adv_loss0 = adv_criterion(subject_pred0.squeeze(), torch.zeros(eeg.shape[0]).cuda().type(LongTensor))
            # adv_loss1 = adv_criterion(subject_pred1.squeeze(), torch.ones(eeg.shape[0]).cuda().type(LongTensor))
            adv_loss0 = adv_criterion(subject_pred0.squeeze(), subid.cuda().type(LongTensor))
            adv_loss1 = adv_criterion(subject_pred1.squeeze(), subid2.cuda().type(LongTensor))
            adv_loss = (adv_loss0 + adv_loss1) * config.alpha

            if not isvalidation:
                scaler_sub.scale(adv_loss).backward(retain_graph=True)
            total_loss = contrastive_loss + adv_loss 
        else:
            total_loss = contrastive_loss

        if config.enable_adv_training and isvalidation==False:
            scaler_sub.step(subject_optimizer)
            scaler_sub.update()
        
        if not isvalidation:
            # if config.enable_adv_training:
            #     contrastive_loss -= adv_loss  #TODO do i need this?
            scaler.scale(contrastive_loss).backward()
            scaler.step(optimizer)
            scaler.update()

    
    optimizers = (optimizer, subject_optimizer)

    return total_loss, contrastive_loss, adv_loss, config, optimizers

def train(models, 
          optimizers, 
          scheduler=None, 
          dataloaders=[None,None],
          config:TrainConfig=None):

    runId = config.runId

    # global logit_scale
    print(f"Training model for sub: {config.nSub} cotrastive: {config.nSub_Contrastive}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion_mse = nn.MSELoss()
    model, model_img_proj, model_eeg_proj, subject_discriminator = models
    optimizer, subject_optimizer = optimizers
    dataloader, val_dataloader = dataloaders

    # torch.autograd.set_detect_anomaly(True)
    # criterion = nn.MSELoss()
    # scaler = GradScaler()  # Mixed precision training scaler
    # scaler_sub = GradScaler()  # Mixed precision training scaler
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    adv_criterion = nn.CrossEntropyLoss()   # For subject discriminator
    # adv_criterion = nn.BCEWithLogitsLoss()   # For subject discriminator, assumes model applies sigmoid

    cpm_AEnc_eeg = CheckpointManager(prefix="AEnc_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")
    
    if config.enable_adv_training:
        cpm_SubDisc = CheckpointManager(prefix="SubDisc",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")

    if config.use_pre_trained_encoder:
        cpm_AEnc_eeg.load_checkpoint(model=model,optimizer=None,epoch="best")
    if config.add_img_proj:
        cpm_Proj_img = CheckpointManager(prefix="Proj_img",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")
    if config.add_eeg_proj:
        cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")

    if config.load_pre_trained_models:
        cpm_AEnc_eeg.load_checkpoint(model=model,optimizer=optimizer,epoch="best")

        if config.add_img_proj:
            cpm_Proj_img.load_checkpoint(model=model_img_proj,optimizer=optimizer,epoch="best")
        if config.add_eeg_proj:
            cpm_Proj_eeg.load_checkpoint(model=model_eeg_proj,optimizer=optimizer,epoch="best")
        if config.enable_adv_training:
            cpm_SubDisc.load_checkpoint(model=subject_discriminator,optimizer=subject_optimizer,epoch="best")


    best_loss = float('inf')
    for local_epoch in range(config.local_epochs):
        config.current_epoch = local_epoch + 1

        train_losses = {"total": 0,"img": 0, "adv": 0, "ortho": 0}
        val_losses = {"total": 0,"img": 0,"adv": 0,"ortho": 0}


        if config.profile_code: start_epoch = time.time()  # Start time for the epoch

        model = freeze_model(model=model,train=True)
        if config.add_img_proj:
            model_img_proj = freeze_model(model=model_img_proj,train=True)
        if config.add_eeg_proj:
            model_eeg_proj = freeze_model(model=model_eeg_proj,train=True)
        if config.enable_adv_training:
            subject_discriminator = freeze_model(model=subject_discriminator,train=True)


        
        for idx, (data1,data2,data3) in enumerate(dataloader):
            config.train_batch_data = (data1,data2,data3)
            config.batch_idx = idx + 1

            total_loss, contrastive_loss, adv_loss, config, optimizers = process_batch(models,optimizers, config, logit_scale, adv_criterion, 
                                                                                       len(dataloader), isvalidation=False)

            if config.profile_code: start_loss = time.time()

            # scaler.scale(total_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            if config.enable_adv_training:
                train_losses["adv"] += adv_loss.item()
            train_losses["total"] += total_loss.item()
            train_losses["img"] += contrastive_loss.item()

            if config.profile_code: end_loss = time.time()
            if config.profile_code: end_batch = time.time()

            # if config.profile_code:
            #     print(f"Batch Time: {end_batch - start_batch:.4f}s | "
            #         f"Data Load: {end_data - start_data:.4f}s | "
            #         f"Forward: {end_forward - start_forward:.4f}s | "
            #         f"Loss Computation: {end_loss - start_loss:.4f}s ")
            # ds_length = len(dataloader.dataset.loaded_indexes_features['1'])
            # print(f"ds length:{ds_length}")


        if val_dataloader is not None:

            model = freeze_model(model=model,train=False)
            if config.add_img_proj:
                model_img_proj = freeze_model(model=model_img_proj,train=False)
            if config.add_eeg_proj:
                model_eeg_proj = freeze_model(model=model_eeg_proj,train=False)
            if config.enable_adv_training:
                subject_discriminator = freeze_model(model=subject_discriminator,train=False)

            with torch.no_grad():
                for vidx, (vdata1,vdata2,vdata3) in enumerate(val_dataloader):
                    batch = (vdata1,vdata2,vdata3)
                    config.batch_idx = vidx + 1
                    config.train_batch_data = batch
                    with autocast():    
                        vtotal_loss, vcontrastive_loss, vadv_loss, config,optimizers = process_batch(models,optimizers, config, logit_scale, adv_criterion, 
                                                                                                     len(val_dataloader),isvalidation=True)
                        if config.enable_adv_training:
                            val_losses["adv"] += vadv_loss.item()
                        val_losses["total"] += vtotal_loss.item()
                        val_losses["img"] += vcontrastive_loss.item()
                

        avg_loss = train_losses["total"] / len(dataloader)
        avg_img_loss = train_losses["img"]  / len(dataloader)

        vavg_loss = val_losses["total"] / len(val_dataloader)
        vavg_img_loss = val_losses["img"]  / len(val_dataloader)
        
        subjectPrint = f"Sub:[{config.nSub}]" if not config.MultiSubject else f"Sub:[{config.nSub}][c-{config.nSub_Contrastive}]"
        prnt_stmt = f"{subjectPrint} Epoch {config.current_epoch}/{config.local_epochs}, Loss: {avg_loss:.4f}, img: {avg_img_loss:.4f} val: {vavg_loss:.4f}  v_img: {vavg_img_loss:.4f}"
        if config.enable_adv_training:
            try:
                avg_adv_loss = train_losses["adv"] / len(dataloader)
                vavg_adv_loss = val_losses["adv"] / len(val_dataloader)
            except Exception as e:
                pass
            prnt_stmt += f", Adv loss: {avg_adv_loss:.4f}"
    
        log_dict = {
                "loss" :avg_loss,
                "img_loss": avg_img_loss,
                "lr": round(optimizer.param_groups[0]["lr"],6),
                "vloss": vavg_loss,
                "vimg_loss": vavg_img_loss,
        }
        if config.enable_adv_training:
            log_dict["adv_loss"] = avg_adv_loss
            log_dict["vadv_loss"] = vavg_adv_loss
            log_dict["alpha"] = config.alpha

        wandb.log(log_dict)
        
        if (config.current_epoch) % 1 == 0:
            print(prnt_stmt)
            # print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            # print(f'Latent shape: {latent.shape}')
            
            if vavg_loss < best_loss:
                best_loss = vavg_loss

                if config.enable_adv_training:
                    cpm_SubDisc.save_checkpoint(model=subj_discriminator.module, optimizer=optimizer, epoch="best")
                if not config.use_pre_trained_encoder:
                    cpm_AEnc_eeg.save_checkpoint(model=model.module,optimizer=optimizer,epoch="best")
                if config.add_img_proj:
                    cpm_Proj_img.save_checkpoint(model=model_img_proj.module,optimizer=optimizer,epoch="best")
                if config.add_eeg_proj:
                    cpm_Proj_eeg.save_checkpoint(model=model_eeg_proj.module,optimizer=optimizer,epoch="best")

    # print("Training model (complete)")
    optimizers = [optimizer, subject_optimizer]

    return [model, model_img_proj, model_eeg_proj, subject_discriminator], optimizers, scheduler

import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can slow down training but ensures determinism

# Example Usage
if __name__ == "__main__":

    set_seed(42)

    t_config = TrainConfig()

    # Hyperparameters
    t_config.learning_rate = 2e-3
    t_config.discriminator_lr = 1e-3
    t_config.batch_size = 16000
    t_config.local_epochs = 50
    t_config.global_epochs = 3

    # config
    t_config.Train = True
    t_config.log_test_data = t_config.Train
    t_config.Contrastive_augmentation = True # enables subject pair EEG 
    t_config.MultiSubject = True
    t_config.TestSubject = 1  # this subject will be used to test and other subjects will be trained.
    t_config.keepDimAfterAvg = False # for NICE model use true else False

    # Adv training
    t_config.enable_adv_training = True
    t_config.alpha = 1.0 #  this is scheduled accorss training steps from 0 to 1
    t_config.lambda_adv = 0.01

    # reproduce NICE
    t_config.add_img_proj = True
    t_config.add_eeg_proj = True
    t_config.encoder_output_dim = 768  # 1440 for NICE and 768 for others


    run_id_to_test = None

    if t_config.Train:
        assert run_id_to_test==None
    else:
        assert run_id_to_test!=None

    runMan = RunManager(run_id=run_id_to_test)
    runId = runMan.getRunID()
    print(runId)

    t_config.runId = runId

    model_img_proj = None
    model_eeg_proj = None
    subj_discriminator = None

    # model = Enc_eeg().cuda()
    # model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
    # model.apply(weights_init_normal)

    model = CNNTrans(input_channels=63,time_samples=250,embed_dim=768).cuda()
    model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
    model.apply(weights_init_normal)

    if t_config.enable_adv_training:
        subj_discriminator = SubjectDiscriminator(embed_dim=768,num_subjects=11,alpha=t_config.alpha).cuda()
        subj_discriminator = nn.DataParallel(subj_discriminator, device_ids=[i for i in range(len(gpus))])
        subj_discriminator.apply(weights_init_normal)

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    trainable_params = []
    
    if not t_config.use_pre_trained_encoder:
        trainable_params += list(model.parameters())

    if t_config.add_img_proj and t_config.only_AE==False:
        model_img_proj = Proj_img(embedding_dim=768,proj_dim=768).cuda()
        model_img_proj = nn.DataParallel(model_img_proj, device_ids=[i for i in range(len(gpus))])
        model_img_proj.apply(weights_init_normal)
        trainable_params += list(model_img_proj.parameters())

    if t_config.add_eeg_proj:
        model_eeg_proj = Proj_eeg(embedding_dim=t_config.encoder_output_dim,proj_dim=768).cuda()
        model_eeg_proj = nn.DataParallel(model_eeg_proj, device_ids=[i for i in range(len(gpus))])
        model_eeg_proj.apply(weights_init_normal)
        trainable_params += list(model_eeg_proj.parameters())


    sub_optimizer = None
    scheduler = None

    if t_config.enable_adv_training:
        sub_optimizer = optim.Adam(subj_discriminator.parameters(), lr=t_config.discriminator_lr)
    #     trainable_params += list(subj_discriminator.parameters())
    optimizer = optim.Adam(trainable_params, lr=t_config.learning_rate)
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    models = [model, model_img_proj, model_eeg_proj, subj_discriminator]
    optimizers = [optimizer, sub_optimizer]

    config_dict = config_to_dict(t_config)
    config_dict["runid"] = runId
    config_dict["change"] = "seperated sub discriminator"

    
    if t_config.Train:
        run = wandb.init(
            # id="adv+_lambda_1.0",
            # Set the project where this run will be logged
            project="Experiement_Vanilla",
            # Track hyperparameters and run metadata
            config=config_dict,
            settings=wandb.Settings(code_dir="/home/jbhol/EEG/gits/BrainCoder")
        )

        wandb.run.log_code("/home/jbhol/EEG/gits/BrainCoder")

    if t_config.Train:
        if not t_config.MultiSubject:
            dataset = EEG_Dataset2(args=t_config,nsub=1,
                        agument_data=t_config.Contrastive_augmentation,
                        load_individual_files=False,
                        subset="train",
                        include_neg_sample=False,
                        preTraning=False,
                        cache_data=True,
                        mean_eeg_data=True,
                        keep_dim_after_mean=t_config.keepDimAfterAvg,
                        constrastive_subject=t_config.nSub_Contrastive,
                        saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")    
            val_dataset = EEG_Dataset2(args=t_config,nsub=1,
                        agument_data=t_config.Contrastive_augmentation,
                        load_individual_files=False,
                        subset="val",
                        include_neg_sample=False,
                        preTraning=False,
                        cache_data=True,
                        mean_eeg_data=True,
                        keep_dim_after_mean=t_config.keepDimAfterAvg,
                        constrastive_subject=t_config.nSub_Contrastive,
                        saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")    
        
            # dataset.preload_data()
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=t_config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=t_config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

            models, optimizers,scheduler = train(models=models,
                                        dataloaders=[dataloader,val_dataloader],
                                        optimizers=optimizers,
                                        scheduler=scheduler,
                                        config=t_config)


        else:
            dataloader = None
            val_dataloader = None

            total_subjects_list = [i for i in range(1,11)]
            total_subjects_list.pop(total_subjects_list.index(t_config.TestSubject)) # remove test subject
            print("Subjects to be trained:", total_subjects_list)

            # no repearing pairs
            # sub_pairs = [list(pair) for pair in itertools.combinations(total_subjects_list, 2)]
            # print(sub_pairs)

            t_config.Total_Subjects = len(total_subjects_list)

            for g_epoch_idx in range(t_config.global_epochs):
                # used for scenario where we want to train [1,2,3,1,2,3,1,2,3] subjects instead of [1,2,3] where each subject is trained for longer epochs
                t_config.current_global_epoch = g_epoch_idx + 1

                # just loop through all subjects
                # if not t_config.Contrastive_augmentation:
                #     sub_pairs = total_subjects_list

                for subIdx, subI in enumerate(total_subjects_list):
                    t_config.current_subject = subIdx + 1
                    t_config.nSub = subI

                    if t_config.Contrastive_augmentation:
                        # subI, contrastive_subI = subI
                        # t_config.nSub_Contrastive = contrastive_subI
                        t_config.nSub = subI

                        contrastive_sub_list = [i for i in total_subjects_list if i!=t_config.TestSubject and i!=subI]
                        # contrastive_sub_list.pop(contrastive_sub_list.index(subI)) # remove current subject for getting contrastive subject
                        t_config.nSub_Contrastive = random.sample(contrastive_sub_list, 1)[0]


                    print("before loading ds")
                    memory_stats()

                    dataset = EEG_Dataset2(args=t_config,nsub=subI,
                            agument_data=t_config.Contrastive_augmentation,
                            load_individual_files=False,
                            subset="train",
                            include_neg_sample=False,
                            preTraning=False,
                            cache_data=True,
                            mean_eeg_data=True,
                            keep_dim_after_mean=t_config.keepDimAfterAvg,
                            constrastive_subject=t_config.nSub_Contrastive,
                            saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")    
                    val_dataset = EEG_Dataset2(args=t_config,nsub=subI,
                                agument_data=t_config.Contrastive_augmentation,
                                load_individual_files=False,
                                subset="val",
                                include_neg_sample=False,
                                preTraning=False,
                                cache_data=True,
                                mean_eeg_data=True,
                                keep_dim_after_mean=t_config.keepDimAfterAvg,
                                constrastive_subject=t_config.nSub_Contrastive,
                                saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata") 
            
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=t_config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
                    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=t_config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

                    models, optimizers,scheduler = train(models=models,
                                        dataloaders=[dataloader,val_dataloader],
                                        optimizers=optimizers,
                                        scheduler=scheduler,
                                        config=t_config)

                    # clean_mem(objects_to_del=["dataset", "val_dataset", "dataloader", "val_dataloader"])
                    del dataset, val_dataset, dataloader, val_dataloader
                    print("after deleting ds")
                    memory_stats()
                    
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass


    subjects_to_test = 10
    if t_config.Train:
        t_config.log_test_data = True
        subjects_to_test = t_config.TestSubject
    results = {
        "top-1": [],
        "top-3": [],
        "top-5": []
    }
    for i in range(subjects_to_test):
        sub = i + 1
        top1_acc, top3_acc, top5_acc = test(runId=runId,models=models,subject_id=sub, with_eeg_projection=t_config.add_eeg_proj, config=t_config)
        results["top-1"].append(top1_acc)
        results["top-3"].append(top3_acc)
        results["top-5"].append(top5_acc)

    print(runId)

    for i in range(len(results["top-1"])):
        prnt_stmt = f"Subject: [{i}] Test "
        for top in [1,3,5]:
            score = results[f'top-{top}'][i]
            prnt_stmt += f" Top{top}-{score:.6f}"
        print(prnt_stmt)
    
    for top in [1,3,5]:
        mscore = np.array(results[f'top-{top}']).mean()
        print(f"mean top-{top}: {mscore}")

    # print(results)

    if t_config.Train:
        # Finish the run
        run.finish()