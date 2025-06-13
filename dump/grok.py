import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.eeg_utils import EEG_Dataset2, get_eeg_data
from utils.eeg_utils import EEG_DATA_PATH,IMG_DATA_PATH,TEST_CENTER_PATH
from utils.common import CheckpointManager, RunManager


from archs.nice import Enc_eeg

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor


import wandb
import random  # for demo script

# highlight-next-line
wandb.login()



# Encoder: Maps EEG data (batch, 63, 250) to latent space (batch, latent_dim)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(63, 128, kernel_size=5, stride=2, padding=2)  # (batch, 64, 125)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)  # (batch, 128, 63)
        self.pool = nn.AdaptiveAvgPool1d(1)  # (batch, 128, 1)
        self.fc = nn.Linear(256, latent_dim)  # (batch, latent_dim)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        z = self.fc(x)
        return z

# Decoder: Reconstructs EEG data from latent space
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 63)  # Expand to feature map size
        self.reshape = lambda x: x.view(-1, 256, 63)
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)  # (batch, 64, 126)
        self.deconv2 = nn.ConvTranspose1d(128, 63, kernel_size=5, stride=2, padding=2, output_padding=0)  # (batch, 63, 250)

    def forward(self, z):
        x = self.fc(z)
        x = self.reshape(x)
        x = F.gelu(self.deconv1(x))
        x = self.deconv2(x)[:, :, :250]  # Crop to exact size (batch, 63, 250)
        return x

# Projection Head: Maps latent features to CLIP embedding space
class Projection(nn.Module):
    def __init__(self, latent_dim, embedding_dim):
        super(Projection, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        p_z = self.fc2(h)
        return p_z

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    

class SubjectDiscriminator(nn.Module):
    def __init__(self, features_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            # nn.Sigmoid()
        )
    
    def forward(self, x, alpha=1):
        x = GradientReversal.apply(x, alpha)
        return self.net(x)
    

# Combined Model
class EEGModel(nn.Module):
    def __init__(self, latent_dim, embedding_dim, img_input_shape=768):
        super(EEGModel, self).__init__()
        # self.encoder = Encoder(latent_dim)
        self.encoder = Enc_eeg(emb_size=40)
        self.decoder = Decoder(latent_dim=1440)
        self.projection = Projection(latent_dim=1440, embedding_dim=embedding_dim)
        self.img_projection = Projection(img_input_shape, embedding_dim)
        self.dc = SubjectDiscriminator(features_dim=embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        p_z = self.projection(z)
        subid = self.dc(p_z)
        return z, x_recon, p_z, subid

def optimizer_scheduler(optimizer, p):
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75
    return optimizer

# Data Augmentation for EEG
def augment(x):
    """Apply noise and time shift to EEG data."""
    noise = torch.randn_like(x) * 0.1  # Gaussian noise with small magnitude
    x_aug = x + noise
    shift = torch.randint(-10, 10, (1,)).item()  # Random shift between -10 and 10
    x_aug = torch.roll(x_aug, shifts=shift, dims=2)
    return x_aug

# Training Step
def train_step(model, optimizer, data_loader, device,epoch,total_epochs,logit_scale, lambda1=1.0, lambda2=1.0, temperature=0.5):
    model.train()
    total_loss = 0

    total_rcon_loss = 0
    total_aug_loss = 0
    total_img_loss = 0

    total_steps = total_epochs * len(data_loader)
    i = 0
    p = 0
    alpha = 1 
    

    criterion = nn.CrossEntropyLoss()

    start_steps = epoch * len(data_loader)
    p = float(i + start_steps) / total_steps
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    for data1,data2,data3 in data_loader:
        # x, c = x.to(device), c.to(device)

        # optimizer = optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()

        (eeg, image_features, cls_label_id, subid) = data1
        # (eeg_2, image_features_2, cls_label_id_2, subid2) = data2  # contrastive subject
        # (eeg_3, image_features_3, cls_label_id_3, subid3) = data3


        eeg_in = torch.mean(eeg,dim=1,keepdim=True)
        # eegMean = Variable(eegMean.cuda().type(self.Tensor))

        # eeg_2 = torch.mean(eeg_2,dim=1,keepdim=False)
        # eegMean2 = Variable(eegMean2.cuda().type(self.Tensor))

        # eeg_2 = eeg_2.to(device).type(Tensor)
        eeg_in,image_features= eeg_in.to(device).type(Tensor), image_features.to(device).type(Tensor)
        
        # Create augmented views
        x1 = augment(eeg_in)
        # x2 = augment(eeg_2)

        # Forward pass
        # z1 = model.module.encoder(eeg)
        z1 = model.module.encoder(x1)
        x_recon = model.module.decoder(z1)
        p_z = model.module.projection(z1)

        # z2 = model.module.encoder(eeg_2)
        # p_z2 = model.module.projection(z2)

        # Losses
        # 1. Reconstruction Loss
        l_recon = F.mse_loss(x_recon, eeg_in[:,0,:,:])

        # 2. Contrastive Loss for Augmentation Invariance
        # z1_norm = F.normalize(z1, dim=1)
        # z2_norm = F.normalize(z2, dim=1)
        # sim_aug = torch.mm(z1_norm, z2_norm.t()) / temperature
        # labels_aug = torch.arange(z1.size(0)).to(device)
        # l_contrastive_aug = F.cross_entropy(sim_aug, labels_aug)

        # dc_z = model.module.dc(p_z,alpha=alpha)
        # dc_z2 = model.module.dc(p_z2,alpha=alpha)

        # s1 = torch.zeros(z.size(0)).cuda().type(LongTensor)  # Subject labels
        # s2 = torch.ones(z.size(0)).cuda().type(LongTensor)  # Subject labels

        # s1_loss = criterion(dc_z, s1)
        # s2_loss = criterion(dc_z2, s2)
        # s_loss = s1_loss + s2_loss


        # 3. Contrastive Loss for Alignment with CLIP Embeddings
        p_z_norm = F.normalize(p_z, dim=1)
        img_feat = model.module.img_projection(image_features)
        c_norm = F.normalize(img_feat, dim=1)
        sim_align = torch.mm(p_z_norm, c_norm.t()) * logit_scale
        labels_align = torch.arange(p_z.size(0)).to(device)
        l_contrastive_align = F.cross_entropy(sim_align, labels_align)

        # Total Loss
        l_total = lambda2 * l_contrastive_align + lambda1 * l_recon

        # Optimization
        # optimizer.zero_grad()
        l_total.backward()
        optimizer.step()

        total_loss += l_total.item()

        total_rcon_loss += l_recon.item()
        # total_aug_loss += s_loss.item()
        total_img_loss += l_contrastive_align.item()
    
    total_loss = total_loss / len(data_loader)
    total_rcon_loss = total_rcon_loss / len(data_loader)
    # total_aug_loss = total_aug_loss /  len(data_loader)
    total_img_loss = total_img_loss / len(data_loader)

    wandb.log({
        "recon": total_rcon_loss, 
        # "dc loss": total_aug_loss,
        "l_contrastive_align": total_img_loss,
        "loss" :total_loss,
        # "alpha": alpha,
        # "opt_sched":p 
        "logit_scale": logit_scale.item()
    })
    
    return total_loss, logit_scale


# Full Training Pipeline
def train_pipeline(data_loader,num_epochs, latent_dim, embedding_dim, device):
    # Dummy data for demonstration (replace with actual data)

    # Initialize model and optimizer
    model = EEGModel(latent_dim, embedding_dim).to(device)

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    best_loss = np.inf
    cpm_AEnc_eeg = CheckpointManager(prefix="AEnc_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")

    run = wandb.init(
        # Set the project where this run will be logged
        project="eeg_grok",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "change": "added aug"
        },
    )


    # Training loop
    for epoch in range(num_epochs):
        avg_loss, logit_scale = train_step(model, optimizer, data_loader, device,epoch, num_epochs,logit_scale)
        if avg_loss<best_loss:
            best_loss = avg_loss
            cpm_AEnc_eeg.save_checkpoint(model=model,optimizer=optimizer,epoch="best")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model, optimizer


def test(latent_dim, embedding_dim,runId, subject_id, dnn="clip", batch_size=8):
    _, _, test_eeg, test_label = get_eeg_data(eeg_data_path=EEG_DATA_PATH, nSub=subject_id, subset="test")


    test_center = np.load(TEST_CENTER_PATH + 'center_' + dnn + '.npy', allow_pickle=True)
    test_eeg = torch.from_numpy(test_eeg)
    # test_img_feature = torch.from_numpy(test_img_feature)
    test_center = torch.from_numpy(test_center)
    test_label = torch.from_numpy(test_label)
    test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    model = EEGModel(latent_dim, embedding_dim).to(device)
    cpm_AEnc_eeg = CheckpointManager(prefix="AEnc_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/grok/{runId}")

    model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])

    cpm_AEnc_eeg.load_checkpoint(model=model,optimizer=None,epoch="best")

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
            
            teeg = torch.mean(teeg,dim=1,keepdim=True)
            z = model.module.encoder(teeg)
            p_z = model.module.projection(z)

            tfea = p_z / p_z.norm(dim=1, keepdim=True)
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
    # self.log_write.write('The best epoch is: %d\n' % best_epoch)
    # self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))
    wandb.log({
        "top-1": top1_acc,
        "top-3": top3_acc,
        "top-5": top5_acc,
    })
    
    return top1_acc, top3_acc, top5_acc

# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 200
    batch_size = 1000
    latent_dim = 128
    embedding_dim = 768  # Match your CLIP embedding size

    class args:
        dnn = "clip"

    args_inst = args()

    dataset = EEG_Dataset2(args=args_inst,nsub=1,
                    agument_data=False,
                    load_individual_files=True,
                    subset="train",
                    include_neg_sample=False,
                    preTraning=True,
                    cache_data=True,
                    constrastive_subject=2,
                    saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    runMan = RunManager(run_id="20250226/115622/5bafe6e7") 
    runId = runMan.getRunID()
    print(runId)
    
    trained_model, optimizer = train_pipeline(dataloader,num_epochs,latent_dim, embedding_dim, device)
    top1_acc, top3_acc, top5_acc = test(latent_dim, embedding_dim,runId=runId,subject_id=1)

    print(runId)