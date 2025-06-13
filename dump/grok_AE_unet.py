import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.eeg_utils import EEG_Dataset2, get_eeg_data
from utils.eeg_utils import EEG_DATA_PATH,IMG_DATA_PATH,TEST_CENTER_PATH
from utils.common import CheckpointManager, RunManager, freeze_model
from archs.AE_Unet import SpatioTemporalEEGAutoencoder, EEGDataAugmentation
from archs.nice import Proj_img, Proj_eeg
from utils.losses import get_contrastive_loss


gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor


import wandb

# highlight-next-line
wandb.login()

def test(runId,models, subject_id, dnn="clip", batch_size=8, with_img_projection=False, with_eeg_projection=False):
    _, _, test_eeg, test_label = get_eeg_data(eeg_data_path=EEG_DATA_PATH, nSub=subject_id, subset="test")

    model, model_img_proj, model_eeg_proj = models

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
    cpm_AEnc_eeg.load_checkpoint(model=model,optimizer=None,epoch="best")

    # if with_img_projection:
    #     model_img_proj = Proj_img(embedding_dim=768,proj_dim=768)
    #     cpm_Proj_img = CheckpointManager(prefix="Proj_img",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")
    #     model_img_proj = model_img_proj.to(device)
    #     cpm_Proj_img.load_checkpoint(model=model_img_proj,optimizer=None,epoch="best")

    if with_eeg_projection:
        model_eeg_proj = Proj_eeg(embedding_dim=768,proj_dim=768)
        model_eeg_proj = model_eeg_proj.to(device)
        cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")
        cpm_Proj_eeg.load_checkpoint(model=model_eeg_proj,optimizer=None,epoch="best")

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
            
            teeg = torch.mean(teeg,dim=1,keepdim=False)

            _, latent = model(teeg)
            if with_eeg_projection:
                latent  = model_eeg_proj(latent)
            # z = model.module.encoder(teeg)
            # p_z = model.module.projection(z)

            tfea = latent / latent.norm(dim=1, keepdim=True)
            similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # no use 100?
            _, indices = similarity.topk(5)

            tt_label = tlabel.view(-1, 1)
            total += tlabel.size(0)
            top1 += (tt_label == indices[:, :1]).sum().item()
            top3 += (tt_label == indices[:, :3]).sum().item()
            top5 += (tt_label == indices).sum().item()

        
        top1_acc = float(top1) / float(total)
        top3_acc = float(top3) / float(total)
        top5_acc = float(top5) / float(total)
    
    print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))

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
    

def fine_tune(models, dataloader, optimizer, scheduler, epochs=200, runId=None, 
        augment_eeg_data=False,add_img_proj=False, add_eeg_proj=False):

    global logit_scale

    model, model_img_proj, model_eeg_proj = models

    # Initialize augmenter
    augmenter = EEGDataAugmentation(
        probability=0.5,
        noise_level=0.09,
        time_shift_max=30,
        amplitude_scale_range=(0.9, 1.1),
        dropout_prob=0.5
    )


    cpm_AEnc_eeg = CheckpointManager(prefix="AEnc_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")
    cpm_AEnc_eeg.load_checkpoint(model=model,optimizer=None,epoch="best")

    if add_img_proj:
        cpm_Proj_img = CheckpointManager(prefix="Proj_img",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")
    if add_eeg_proj:
        cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")


    best_loss = float('inf')
    for epoch in range(epochs):

        total_loss = 0
        total_rcon_loss = 0
        total_img_loss = 0

        model = freeze_model(model, train=False)
        if add_img_proj:
            model_img_proj = freeze_model(model_img_proj, train=True)
        if add_eeg_proj:
            model_eeg_proj = freeze_model(model_eeg_proj, train=True)
        
        # step = 0
        for data1,data2,data3 in dataloader:
            # x, c = x.to(device), c.to(device)

            # optimizer = optimizer_scheduler(optimizer, p)
            optimizer.zero_grad()

            (eeg, image_features, cls_label_id, subid) = data1
            # (eeg_2, image_features_2, cls_label_id_2, subid2) = data2  # contrastive subject
            # (eeg_3, image_features_3, cls_label_id_3, subid3) = data3

            eegMean = torch.mean(eeg,dim=1,keepdim=False)
            
            if augment_eeg_data:
                x1 = augmenter.augment(eegMean).to(device).type(Tensor)
                _, latent = model(x1, encoder_only=True)
                eegMean = eegMean.to(device).type(Tensor)
            else:
                eegMean = eegMean.to(device).type(Tensor)
                _, latent = model(eegMean, encoder_only=True)

            if add_eeg_proj:
                latent = do_forward(model_eeg_proj, latent, train=add_eeg_proj)
            
            image_features = image_features.to(device).type(Tensor)
            if add_img_proj:
                image_features = model_img_proj(image_features)

            overall_loss = 0
            labels_align = torch.arange(latent.size(0)).to(device)
            img_loss = get_contrastive_loss(feat1=latent,feat2=image_features,labels=labels_align,logit_scale=logit_scale)
            total_img_loss += img_loss
            overall_loss += img_loss
            overall_loss.backward()

            if add_eeg_proj:
                torch.nn.utils.clip_grad_norm_(model_eeg_proj.parameters(), max_norm=1.0)
            if add_img_proj:
                torch.nn.utils.clip_grad_norm_(model_img_proj.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step(overall_loss)
            total_loss += overall_loss

            # print(f'step [{step+1}/{len(dataloader)}], Loss: {overall_loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            # step +=1
        

        total_loss = total_loss / len(dataloader)
        if not t_config.use_pre_trained_encoder:
            total_rcon_loss = total_rcon_loss / len(dataloader)
        
        if t_config.only_AE==False:
            total_img_loss = total_img_loss / len(dataloader)
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            # print(f'Latent shape: {latent.shape}')

            log_dict = {
                "loss" :total_loss.item(),
                "lr": round(optimizer.param_groups[0]["lr"],6),
            }

            log_dict["l_contrastive_align"] = total_img_loss.item()
            log_dict["logit_scale"]= logit_scale.item()

            wandb.log(log_dict)
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()

                if add_img_proj:
                    cpm_Proj_img.save_checkpoint(model=model_img_proj,optimizer=optimizer,epoch="best")

                if add_eeg_proj:
                    cpm_Proj_eeg.save_checkpoint(model=model_eeg_proj,optimizer=optimizer,epoch="best")

    print("Training model (complete)")

    return (model, model_img_proj, model_eeg_proj), optimizer, scheduler

class TrainConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    batch_size = 2048
    learning_rate = 0.001
    latent_dim = 768
    dropout_rate = 0.1
    init_features = 32
    add_img_proj = True
    add_eeg_proj = True
    use_pre_trained_encoder = True
    nSub = 1
    nSub_Contrastive = 2
    Contrastive_augmentation = False
    EEG_Augmentation = False
    only_AE = False
    Total_Subjects = 1
    dnn = "clip"


def train(models, dataloader,optimizer, 
          scheduler, epochs=200, 
          add_img_proj=True,
          add_eeg_proj=True,
          runId=None,augment_eeg_data=True,use_pre_trained_encoder=False,only_AE=False):

    global logit_scale

    print("Training model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion_mse = nn.MSELoss()
    # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    # trainable_params = [logit_scale]
    # if not use_pre_trained_encoder:
    #     trainable_params += list(model.parameters())

    # if add_img_proj and only_AE==False:
    #     model_img_proj = Proj_img(embedding_dim=768,proj_dim=768)
    #     model_img_proj = model_img_proj.to(device)
    #     trainable_params += list(model_img_proj.parameters())

    # if add_eeg_proj:
    #     model_eeg_proj = Proj_eeg(embedding_dim=768,proj_dim=768)
    #     model_eeg_proj = model_eeg_proj.to(device)
    #     trainable_params += list(model_eeg_proj.parameters())
    # optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-5)

    model, model_img_proj, model_eeg_proj = models

    # Initialize augmenter
    augmenter = EEGDataAugmentation(
        probability=0.7,
        noise_level=0.09,
        time_shift_max=30,
        amplitude_scale_range=(0.9, 1.1),
        dropout_prob=0.7
    )

    cpm_AEnc_eeg = CheckpointManager(prefix="AEnc_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")

    if use_pre_trained_encoder:
        cpm_AEnc_eeg.load_checkpoint(model=model,optimizer=None,epoch="best")

    if add_img_proj and only_AE==False:
        cpm_Proj_img = CheckpointManager(prefix="Proj_img",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")
    
    if add_eeg_proj:
        cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")


    best_loss = float('inf')
    for epoch in range(epochs):

        total_loss = 0
        total_rcon_loss = 0
        total_img_loss = 0
        
        # step = 0
        for data1,data2,data3 in dataloader:
            # x, c = x.to(device), c.to(device)

            # optimizer = optimizer_scheduler(optimizer, p)
            optimizer.zero_grad()

            (eeg, image_features, cls_label_id, subid) = data1
            (eeg_2, image_features_2, cls_label_id_2, subid2) = data2  # contrastive subject
            # (eeg_3, image_features_3, cls_label_id_3, subid3) = data3

            eegMean = torch.mean(eeg,dim=1,keepdim=False)
            eegMean2 = torch.mean(eeg_2,dim=1,keepdim=False)
            # eegMean = Variable(eegMean.cuda().type(self.Tensor))

            # eeg_2 = torch.mean(eeg_2,dim=1,keepdim=False)
            # eegMean2 = Variable(eegMean2.cuda().type(self.Tensor))


            overall_loss = 0

            for eeg_data in [eegMean,eegMean2]:

                if augment_eeg_data:
                    x1 = augmenter.augment(eeg_data).to(device).type(Tensor)
                    outputs, latent = do_forward(model, x1, train=not use_pre_trained_encoder)
                    if add_eeg_proj:
                        latent = do_forward(model_eeg_proj, latent, train=add_eeg_proj)
                    eeg_data = eeg_data.to(device).type(Tensor)
                else:
                    eeg_data = eeg_data.to(device).type(Tensor)
                    outputs, latent = do_forward(model, eeg_data, train=not use_pre_trained_encoder)
                
                

                if not use_pre_trained_encoder:
                    recon_loss = criterion_mse(outputs, eeg_data)
                    total_rcon_loss += recon_loss   
                    overall_loss += recon_loss 

                if only_AE==False:
                    image_features = image_features.to(device).type(Tensor)
                    if add_img_proj:
                        image_features = model_img_proj(image_features)

                    labels_align = torch.arange(latent.size(0)).to(device)
                    img_loss = get_contrastive_loss(feat1=latent,feat2=image_features,labels=labels_align,logit_scale=logit_scale)
                    total_img_loss += img_loss
                    overall_loss += img_loss
            
            overall_loss = overall_loss/2
            total_img_loss = total_img_loss/2

            overall_loss.backward()

            if only_AE:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if add_eeg_proj:
                torch.nn.utils.clip_grad_norm_(model_eeg_proj.parameters(), max_norm=1.0)
            if add_img_proj:
                torch.nn.utils.clip_grad_norm_(model_img_proj.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(overall_loss)
            total_loss += overall_loss

            # print(f'step [{step+1}/{len(dataloader)}], Loss: {overall_loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            # step +=1
        

        total_loss = total_loss / len(dataloader)
        if not use_pre_trained_encoder:
            total_rcon_loss = total_rcon_loss / len(dataloader)
        
        if only_AE==False:
            total_img_loss = total_img_loss / len(dataloader)
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            # print(f'Latent shape: {latent.shape}')

            log_dict = {
                "loss" :total_loss.item(),
                "lr": round(optimizer.param_groups[0]["lr"],6),
            }

            if not use_pre_trained_encoder:
                log_dict["recon"] = total_rcon_loss.item()

            if only_AE==False:
                log_dict["l_contrastive_align"] = total_img_loss.item()
                log_dict["logit_scale"]= logit_scale.item()

            wandb.log(log_dict)
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()

                if not use_pre_trained_encoder:
                    cpm_AEnc_eeg.save_checkpoint(model=model,optimizer=optimizer,epoch="best")

                if add_img_proj and only_AE==False:
                    cpm_Proj_img.save_checkpoint(model=model_img_proj,optimizer=optimizer,epoch="best")

                if add_eeg_proj:
                    cpm_Proj_eeg.save_checkpoint(model=model_eeg_proj,optimizer=optimizer,epoch="best")

    print("Training model (complete)")

    return (model, model_img_proj, model_eeg_proj), optimizer, scheduler


# Example Usage
if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_epochs = 30
    # batch_size = 2048
    # learning_rate = 0.001
    # latent_dim = 768
    # dropout_rate = 0.1
    # init_features = 32
    # add_img_proj = False
    # add_eeg_proj = False
    # use_pre_trained_encoder = False
    # nSub = 1
    # nSub_Contrastive = 2
    # Contrastive_augmentation = False
    # EEG_Augmentation = True
    # only_AE = True
    # Total_Subjects = 5

    t_config = TrainConfig()


    runMan = RunManager(run_id="20250306/132847/2a792265")
    runId = runMan.getRunID()
    print(runId)
    
    model = SpatioTemporalEEGAutoencoder(latent_dim=t_config.latent_dim, dropout_rate=t_config.dropout_rate, init_features=t_config.init_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    trainable_params = []
    model_img_proj = None
    model_eeg_proj = None
    if not t_config.use_pre_trained_encoder:
        trainable_params += list(model.parameters())

    if t_config.add_img_proj and t_config.only_AE==False:
        model_img_proj = Proj_img(embedding_dim=768,proj_dim=768)
        model_img_proj = model_img_proj.to(device)
        trainable_params += list(model_img_proj.parameters())

    if t_config.add_eeg_proj:
        model_eeg_proj = Proj_eeg(embedding_dim=768,proj_dim=768)
        model_eeg_proj = model_eeg_proj.to(device)
        trainable_params += list(model_eeg_proj.parameters())

    # optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
    learning_rate = 1e-4
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    models = (model, model_img_proj, model_eeg_proj)

    run = wandb.init(
        # Set the project where this run will be logged
        project="eeg_grok_unet_AE_only",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": t_config.learning_rate,
            "epochs": t_config.num_epochs,
            "change": "Unet initial experiments",
            "batch_size": t_config.batch_size,
            "dropout_rate": t_config.dropout_rate,
            "runId": runId,
            "init_features": t_config.init_features,
            "latent_dim": t_config.latent_dim,
            "add_img_proj": str(t_config.add_img_proj),
            "add_eeg_proj": str(t_config.add_eeg_proj),
            "nSub": t_config.nSub,
            "nSubContrastive": t_config.nSub_Contrastive if t_config.Contrastive_augmentation else "NA",
            "EEG_Augmentation": str(t_config.EEG_Augmentation),
            "only_AE": str(t_config.only_AE),
            "runid": runId,
            "Total_Subjects": t_config.Total_Subjects
        },
    )

    for i in range(t_config.Total_Subjects):
        t_config.nSub_Contrastive = i + 1

        dataset = EEG_Dataset2(args=t_config,nsub=1,
                    agument_data=t_config.Contrastive_augmentation,
                    load_individual_files=True,
                    subset="train",
                    include_neg_sample=False,
                    preTraning=True,
                    cache_data=True,
                    constrastive_subject=t_config.nSub_Contrastive,
                    saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=t_config.batch_size, shuffle=True)

        models, optimizer,scheduler = train(models=models, 
                            dataloader=dataloader,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epochs=t_config.num_epochs,
                            add_img_proj=t_config.add_img_proj, 
                            add_eeg_proj=t_config.add_eeg_proj,
                            runId=runId, 
                            only_AE=t_config.only_AE,
                            use_pre_trained_encoder=t_config.use_pre_trained_encoder,
                            augment_eeg_data=t_config.EEG_Augmentation)

    dataset = EEG_Dataset2(args=t_config,nsub=1,
                agument_data=t_config.Contrastive_augmentation,
                load_individual_files=True,
                subset="train",
                include_neg_sample=False,
                preTraning=True,
                cache_data=True,
                constrastive_subject=t_config.nSub_Contrastive,
                saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=t_config.batch_size, shuffle=True)

    models, optimizer,scheduler = fine_tune(models=models,
                        dataloader=dataloader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epochs=t_config.num_epochs,
                        runId=runId,
                        augment_eeg_data=t_config.EEG_Augmentation,
                        add_img_proj=t_config.add_img_proj,
                        add_eeg_proj=t_config.add_eeg_proj)
    
    top1_acc, top3_acc, top5_acc = test(runId=runId,models=models,subject_id=1, with_eeg_projection=t_config.add_eeg_proj)

    print(runId)