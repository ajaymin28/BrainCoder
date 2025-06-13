"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""

import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange
import copy
from utils.eeg_utils import EEG_Dataset2
from utils.losses import cosine_dissimilarity_loss, kld_loss, SoftmaxThresholdLoss
from utils.common import CheckpointManager, RunManager
from archs.FD import FeatureDecomposerV2, ContrastiveLoss,orthogonality_loss,reconstruction_loss


gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
result_path = '/home/jbhol/EEG/gits/BrainCoder/results/' 
os.makedirs(result_path,exist_ok=True)
model_idx = 'sub1_nice_ssl'
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--model_output', default='/home/jbhol/EEG/gits/BrainCoder/model/', type=str)
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--cycles', default='5', type=int)
parser.add_argument('--pretrain_fd_epoch', default='25', type=int)
parser.add_argument('--pretrain_img_epoch', default='200', type=int)
parser.add_argument('--num_sub', default=2, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total'
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


import torch
import torch.nn.functional as F

def info_nce_loss_two_vectors(anchor, positive,negative,temperature=0.1):
    """
    Args:
        anchor: Tensor of shape (D,) or (1, D), the feature vector for the anchor.
        positive: Tensor of shape (D,) or (1, D), the feature vector for the positive pair.
        temperature: Temperature scaling factor for logits.
    Returns:
        loss: Scalar InfoNCE loss.
    """
    # Ensure both vectors are 2D (batch size = 1 if necessary)
    if anchor.dim() == 1:
        anchor = anchor.unsqueeze(0)
    if positive.dim() == 1:
        positive = positive.unsqueeze(0)
    if negative.dim() == 1:
        negative = negative.unsqueeze(0)

    # Normalize the feature vectors
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)

    # Compute similarity scores
    positive_similarity = torch.matmul(anchor, positive.T)  # Similarity between anchor and positive
    negative_similarity = torch.matmul(anchor, negative.T)    # Self-similarity for negatives (diagonal)

    # Combine similarities into logits
    logits = torch.cat([positive_similarity, negative_similarity], dim=1)
    logits /= temperature  # Scale by temperature

    # Create target labels (0 for positive pair)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, num_heads=4, output_dim=1440):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.attention = nn.MultiheadAttention(embed_dim=1440, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(1440, output_dim)


    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 


# Image2EEG
class IE():
    def __init__(self, args, nsub, runId=None):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 8
        self.batch_size_img = 500 
        self.n_epochs = args.epoch
        self.pretrain_fd_epoch = args.pretrain_fd_epoch
        self.pretrain_img_epoch = args.pretrain_img_epoch
        self.runId = runId

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub
        self.cycles = args.cycles

        self.start_epoch = 0
        self.eeg_data_path = '/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz'
        self.img_data_path = '/home/jbhol/EEG/gits/NICE-EEG/dnn_feature/'
        self.test_center_path = '/home/jbhol/EEG/gits/NICE-EEG/dnn_feature/'
        self.pretrain = False

        MODEL_SAVE_PATH = self.args.model_output
        os.makedirs(MODEL_SAVE_PATH,exist_ok=True)

        # self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.temperature = 0.1
        # self.commdiffloss = SoftmaxThresholdLoss(alpha=0.3,beta=0.7,temperature=self.temperature).cuda()

        self.Enc_eeg = Enc_eeg().cuda()
        # self.eeg_subj_emb = EEG_SubjectEmb().cuda()
        # self.EEGAlignment = EEGAlignmentModel(feature_dim=1440,embedding_dim=768).cuda()

        
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        # self.eeg_subj_emb = nn.DataParallel(self.eeg_subj_emb, device_ids=[i for i in range(len(gpus))])
        # self.EEGAlignment = nn.DataParallel(self.EEGAlignment, device_ids=[i for i in range(len(gpus))])

        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')



    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)

        train_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
        # train_data_original = copy.deepcopy(train_data['preprocessed_eeg_data'])
        train_data = train_data['preprocessed_eeg_data']
        train_data = np.mean(train_data, axis=1)
        train_data = np.expand_dims(train_data, axis=1)

        test_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
        test_data = test_data['preprocessed_eeg_data']
        test_data = np.mean(test_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)

        return train_data, train_label, test_data, test_label

    def get_image_data(self):
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_training.npy', allow_pickle=True)
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_test.npy', allow_pickle=True)

        train_img_feature = np.squeeze(train_img_feature)
        test_img_feature = np.squeeze(test_img_feature)

        return train_img_feature, test_img_feature
        
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_contrastive_loss(self, feat1, feat2, labels):
        feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_f1 = logit_scale * feat1 @ feat2.t()
        logits_f2 = logits_f1.t()
        loss_f1 = self.criterion_cls(logits_f1, labels)
        loss_f2 = self.criterion_cls(logits_f2, labels)
        loss = (loss_f1 + loss_f2) / 2
        return loss

    def feature_decompose(self, runId):
        
        self.Enc_eeg = Enc_eeg().cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        # cpm_Enc_eeg = CheckpointManager(prefix="Enc_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/sub1")
        cpm_Enc_eeg = CheckpointManager(prefix="Enc_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        # cpm_Enc_eeg.load_checkpoint(model=self.Enc_eeg,optimizer=None,epoch="best")

        # self.Proj_eeg = Proj_eeg().cuda()
        # self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        # # cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/sub1")
        # cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        # # cpm_Proj_eeg.load_checkpoint(model=self.Proj_eeg,optimizer=None,epoch="best")

        self.Proj_img = Proj_img().cuda()
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])
        # cpm_Proj_img = CheckpointManager(prefix="Proj_img",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/sub1")
        cpm_Proj_img = CheckpointManager(prefix="Proj_img",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        # cpm_Proj_img.load_checkpoint(model=self.Proj_img,optimizer=None,epoch="best")

        
        # Models that will be trained
        self.FD_model = FeatureDecomposerV2(1440, 768)
        self.FD_model = nn.DataParallel(self.FD_model, device_ids=[i for i in range(len(gpus))])
        cpm_FD_model = CheckpointManager(prefix="FD_model",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}/G")
        contrastive_criterion = ContrastiveLoss()
        lambda_orthojk = 0.1
        lambda_ortho = 0.2  # Weight for orthogonality loss
        lambda_recon = 0.1  # Weight for reconstruction loss
        lambda_img = 0.5
        lambda_commonality = 0.5

        # self.Enc_eeg_G = Enc_eeg().cuda()
        # self.Enc_eeg_G = nn.DataParallel(self.Enc_eeg_G, device_ids=[i for i in range(len(gpus))])
        
        # self.Proj_eeg_G = Proj_eeg().cuda()
        # self.Proj_eeg_G = nn.DataParallel(self.Proj_eeg_G, device_ids=[i for i in range(len(gpus))])
        
        # self.Proj_img_G = Proj_img().cuda()
        # self.Proj_img_G = nn.DataParallel(self.Proj_img_G, device_ids=[i for i in range(len(gpus))])
        

        # cpm_Enc_eeg_G = CheckpointManager(prefix="Enc_eeg_G",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/G")
        # cpm_Proj_eeg_G = CheckpointManager(prefix="Proj_eeg_G",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/G")
        # cpm_Proj_img_G = CheckpointManager(prefix="Proj_img_G",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/G")



        
        for sub_i in range(2,4):
            num = 0
            best_loss_val = np.inf
            best_loss_ortho_val = np.inf
            best_loss_img_val = np.inf
            best_loss_recon_val = np.inf

            dataset = EEG_Dataset2(args=self.args,nsub=self.args.num_sub,
                    agument_data=True,
                    load_individual_files=True,
                    subset="train",
                    include_neg_sample=False,
                    preTraning=True,
                    cache_data=False,
                    constrastive_subject=sub_i,
                    saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")

            self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
            self.margin = 1.0


            # self.Enc_eeg_subi = Enc_eeg().cuda()
            # self.Enc_eeg_subi = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
            # self.Proj_eeg_subi = Proj_eeg().cuda()
            # self.Proj_eeg_subi = nn.DataParallel(self.Proj_eeg_subi, device_ids=[i for i in range(len(gpus))])
            # self.Proj_img_subi = Proj_img().cuda()
            # self.Proj_img_subi = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])
            
            # cpm_Enc_eeg_subi = CheckpointManager(prefix="Enc_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/sub{sub_i}")
            # cpm_Enc_eeg_subi.load_checkpoint(model=self.Enc_eeg_subi,optimizer=None,epoch="best")

            # cpm_Proj_eeg_subi = CheckpointManager(prefix="Proj_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/sub{sub_i}")
            # cpm_Proj_eeg_subi.load_checkpoint(model=self.Proj_eeg_subi,optimizer=None,epoch="best")

            # cpm_Proj_img_subi = CheckpointManager(prefix="Proj_img",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/sub{sub_i}")
            # cpm_Proj_img_subi.load_checkpoint(model=self.Proj_img,optimizer=None,epoch="best")
        
           
            # subject J
            # self.freeze_model(self.Enc_eeg, train=False)
            # self.freeze_model(self.Proj_eeg, train=False)
            # self.freeze_model(self.Proj_img, train=False)

            # subject K
            # self.freeze_model(self.Enc_eeg_subi, train=False)
            # self.freeze_model(self.Proj_eeg_subi, train=False)
            # self.freeze_model(self.Proj_img_subi, train=False)
            

            self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(),
                                                              self.FD_model.parameters(),
                                                              self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))

            train_FD = False
            # for cycle_i in range(self.cycles):

                # if train_FD:
                #     epochs_to_train = self.pretrain_fd_epoch
                # else:
                #     epochs_to_train = self.pretrain_img_epoch

            for e in range(self.pretrain_img_epoch):
                # in_epoch = time.time()

                # self.freeze_model(self.FD_model, train=False)
                # self.freeze_model(self.Enc_eeg, train=False)
                # self.freeze_model(self.Proj_eeg, train=False)
                # self.freeze_model(self.Proj_img, train=False)

                self.freeze_model(self.FD_model, train=True)
                self.freeze_model(self.Enc_eeg, train=True)
                # self.freeze_model(self.Proj_eeg, train=True)
                self.freeze_model(self.Proj_img, train=True)

                for i, (data1,data2,data3) in enumerate(self.dataloader):

                    (eeg, image_features, cls_label_id) = data1
                    (eeg_2, image_features_2, cls_label_id_2) = data2  # contrastive subject
                    # (eeg_3, image_features_3, cls_label_id_3) = data3

                    image_features = Variable(image_features.cuda().type(self.Tensor))
                    image_features_2 = Variable(image_features_2.cuda().type(self.Tensor))

                    eegMean = torch.mean(eeg,dim=1,keepdim=True)
                    eegMean = Variable(eegMean.cuda().type(self.Tensor))

                    eegMean2 = torch.mean(eeg_2,dim=1,keepdim=True)
                    eegMean2 = Variable(eegMean2.cuda().type(self.Tensor))

                    labels = torch.arange(eegMean.shape[0])  # used for the loss
                    labels = Variable(labels.cuda().type(self.LongTensor))

                    # obtain the features
                    E_j = self.Enc_eeg(eegMean)
                    E_k = self.Enc_eeg(eegMean2)

                    # get initial subject variances for each subject
                    E_c_j, V_j, reconstructed_j = self.FD_model(E_j)
                    E_c_k, V_k, reconstructed_k = self.FD_model(E_k)
                    
                    image_features = self.Proj_img(image_features)
                    image_features_2 = self.Proj_img(image_features_2)

                    # # pass variance to cross subject and expect it to form the input like features,
                    # # this is to make E_c as pure as Image class features
                    # _, _, reconstructed_k_variance = self.FD_model(None, subject_variance=V_k, E_c_subject=E_c_j)
                    # _, _, reconstructed_j_variance = self.FD_model(None, subject_variance=V_j, E_c_subject=E_c_k)

                    # # cross reconstruction loss
                    # recon_loss_j_cross = self.criterion_l2(reconstructed_k_variance, E_k)  # SV_K + EC_J
                    # recon_loss_k_cross = self.criterion_l2(reconstructed_j_variance, E_j)  # SV_j + EC_k
                    # recon_cross_loss =  (recon_loss_j_cross + recon_loss_k_cross)

                    # commonality
                    commonality_loss = self.get_contrastive_loss(E_c_j, E_c_k,labels=labels)

                    # loss for FD model to seperaste variance
                    ortho_loss_j = orthogonality_loss(E_c_j, V_j)
                    ortho_loss_k = orthogonality_loss(E_c_k, V_k)
                    overall_ortho_loss = (ortho_loss_j + ortho_loss_k)

                    # recon loss
                    recon_loss_j = self.criterion_l2(reconstructed_j, E_j)
                    recon_loss_k = self.criterion_l2(reconstructed_k, E_k)

                    overall_recon_loss = (recon_loss_j + recon_loss_k)

                    contrastive_loss_img_j = self.get_contrastive_loss(E_c_j, image_features,labels=labels)
                    contrastive_loss_img_k = self.get_contrastive_loss(E_c_k, image_features_2,labels=labels)
                    overall_img_loss = (contrastive_loss_img_j+contrastive_loss_img_k)/2

                    loss = lambda_ortho * overall_ortho_loss + \
                    lambda_recon * overall_recon_loss + \
                    lambda_commonality * commonality_loss + \
                    overall_img_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                if (e + 1) % 1 == 0:

                    ortho_loss = overall_ortho_loss <=best_loss_ortho_val
                    img_loss = overall_img_loss <=best_loss_img_val
                    recon_loss = overall_recon_loss <=best_loss_recon_val

                    if recon_loss:
                        best_loss_recon_val = overall_recon_loss

                    if img_loss:
                        best_loss_img_val = overall_img_loss

                    if ortho_loss:
                        best_loss_ortho_val = overall_ortho_loss


                    if ortho_loss and img_loss and recon_loss:

                        best_epoch = e + 1

                        cpm_FD_model.save_checkpoint(model=self.FD_model.module,optimizer=self.fd_optimizer,epoch="best")
                        cpm_Enc_eeg.save_checkpoint(model=self.Enc_eeg,optimizer=self.fd_optimizer,epoch="best")
                        cpm_Proj_img.save_checkpoint(model=self.Proj_img,optimizer=self.fd_optimizer,epoch="best")
                        # cpm_Proj_eeg.save_checkpoint(model=self.Proj_eeg,optimizer=self.fd_optimizer,epoch="best")


                    print('Sub id', self.nSub, 
                    ' CrossSub', sub_i,
                    ' Epoch:', e,
                    ' best epoch: %d' % best_epoch,
                    ' loss: %.4f' % loss.detach().cpu().numpy(),
                    ' commonality_loss: %.4f' % commonality_loss.detach().cpu().numpy(),
                    ' img: %.4f' % overall_img_loss.detach().cpu().numpy(),  
                    ' recon: %.4f' % overall_recon_loss.detach().cpu().numpy(),
                    ' ortho: %.4f' % overall_ortho_loss.detach().cpu().numpy(),  
                    # ' cycle i', cycle_i
                    )

                    # self.log_write.write('Epoch %d: loss: %.4f, \n'%(e, loss.detach().cpu().numpy()))

        return 0,0,0

    def fine_tune(self, subjectId, runId, withFeatureDecompose=False, pretrained_subject_id=1):


        train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        train_img_feature, _ = self.get_image_data() 
        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:740])
        val_image = torch.from_numpy(train_img_feature[:740])

        train_eeg = torch.from_numpy(train_eeg[740:])
        train_image = torch.from_numpy(train_img_feature[740:])


        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_eeg = torch.from_numpy(test_eeg)
        # test_img_feature = torch.from_numpy(test_img_feature)
        self.test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)



        cpm_Enc_eeg = CheckpointManager(prefix="Enc_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/sub{pretrained_subject_id}")
        cpm_Enc_eeg.load_checkpoint(model=self.Enc_eeg, optimizer=None,epoch="best")

        cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/sub{pretrained_subject_id}")
        cpm_Proj_eeg.load_checkpoint(model=self.Proj_eeg, optimizer=None,epoch="best")
        
        cpm_Proj_img = CheckpointManager(prefix="Proj_img",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/sub{pretrained_subject_id}")
        cpm_Proj_img.load_checkpoint(model=self.Proj_img, optimizer=None,epoch="best")


        # Optimizers
        self.ft_optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(),self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))

        best_loss_val = np.inf
        for e in range(self.n_epochs):
            # in_epoch = time.time()

            self.freeze_model(self.Proj_eeg, train=True)
            self.freeze_model(self.Proj_img, train=True)
            self.freeze_model(self.Enc_eeg, train=True)

            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img) in enumerate(self.dataloader):

                eeg = Variable(eeg.cuda().type(self.Tensor))
                # img = Variable(img.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                # label = Variable(label.cuda().type(self.LongTensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                # Forward pass
                eeg_features = self.Enc_eeg(eeg)

                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features)
                img_features = self.Proj_img(img_features)

                if withFeatureDecompose:
                    # seperate variance
                    eeg_features, V = self.FD_model(eeg_features)

                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # total loss
                loss = loss_cos

                self.ft_optimizer.zero_grad()
                loss.backward()
                self.ft_optimizer.step()


            if (e + 1) % 1 == 0:
                self.freeze_model(self.Enc_eeg)
                self.freeze_model(self.Proj_eeg)
                self.freeze_model(self.Proj_img)

                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = Variable(veeg.cuda().type(self.Tensor))
                        vimg_features = Variable(vimg.cuda().type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                        # Forward pass
                        veeg_features = self.Enc_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            # best_epoch = e + 1

                            cpm_Enc_eeg.save_checkpoint(model=self.Enc_eeg.module,optimizer=self.ft_optimizer,epoch="best")
                            cpm_Proj_img.save_checkpoint(model=self.Proj_img.module,optimizer=self.ft_optimizer,epoch="best")
                            cpm_Proj_eeg.save_checkpoint(model=self.Proj_eeg.module,optimizer=self.ft_optimizer,epoch="best")

                            # torch.save(self.Enc_eeg.module.state_dict(), '/home/jbhol/EEG/gits/NICE-EEG/model/' + model_idx + 'Enc_eeg_cls.pth')
                            # torch.save(self.Proj_eeg.module.state_dict(), '/home/jbhol/EEG/gits/NICE-EEG/model/' + model_idx + 'Proj_eeg_cls.pth')
                            # torch.save(self.Proj_img.module.state_dict(), '/home/jbhol/EEG/gits/NICE-EEG/model/' + model_idx + 'Proj_img_cls.pth')

                print('Epoch:', e,
                      '  Cos eeg: %.4f' % loss_eeg.detach().cpu().numpy(),
                      '  Cos img: %.4f' % loss_img.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                      )
                # self.log_write.write('Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n'%(e, loss_eeg.detach().cpu().numpy(), loss_img.detach().cpu().numpy(), vloss.detach().cpu().numpy()))
        
        return 0, 0, 0

    
    def freeze_model(self, model, train=False):
        for name, param in model.named_parameters():
            param.requires_grad = train
    
    def test(self, subjectId, runId, withFeatureDecompose=False, pretrained_subject_id=1):


        self.nSub = subjectId

        _, _, test_eeg, test_label = self.get_eeg_data()
        # _, _ = self.get_image_data() 
        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        test_eeg = torch.from_numpy(test_eeg)
        # test_img_feature = torch.from_numpy(test_img_feature)
        self.test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        if withFeatureDecompose:

            self.FD_model = FeatureDecomposerV2(768, 768)
            self.FD_model = nn.DataParallel(self.FD_model, device_ids=[i for i in range(len(gpus))])
            cpm_FD_model = CheckpointManager(prefix="FD_model",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}/G")
            cpm_FD_model.load_checkpoint(model=self.FD_model,optimizer=None,epoch="best")
            self.freeze_model(self.FD_model)

            # cpm_Enc_eeg_G = CheckpointManager(prefix="Enc_eeg_G",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/G")
            # cpm_Proj_eeg_G = CheckpointManager(prefix="Proj_eeg_G",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/G")
            # cpm_Proj_img_G = CheckpointManager(prefix="Proj_img_G",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId or self.runId}/G")

            # cpm_Enc_eeg_G.load_checkpoint(model=self.Enc_eeg.module,optimizer=None,epoch="best")
            # cpm_Proj_eeg_G.load_checkpoint(model=self.Proj_eeg.module,optimizer=None,epoch="best")
            # cpm_Proj_img_G.load_checkpoint(model=self.Proj_img.module,optimizer=None,epoch="best")
        
        # else:
            
            cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",
                                         base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
            cpm_Proj_img = CheckpointManager(prefix="Proj_img",
                                            base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
            cpm_Enc_eeg = CheckpointManager(prefix="Enc_eeg",
                                            base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
            
            cpm_Proj_eeg.load_checkpoint(model=self.Proj_eeg.module,optimizer=None,epoch="best")
            cpm_Proj_img.load_checkpoint(model=self.Proj_img.module,optimizer=None,epoch="best")
            cpm_Enc_eeg.load_checkpoint(model=self.Enc_eeg.module,optimizer=None,epoch="best")


        # * test part
        all_center = self.test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.freeze_model(self.Proj_img)
        self.freeze_model(self.Proj_eeg)
        self.freeze_model(self.Enc_eeg)
        

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))   

                teeg_features = self.Enc_eeg(teeg)
                # tfea = self.Proj_eeg(teeg_features)

                if withFeatureDecompose:
                    tfea_fc, V, recon = self.FD_model(teeg_features)
                    # tfea = tfea - (tfea_fc + V)
                    tfea = (tfea + tfea_fc)/2


                tfea = tfea / tfea.norm(dim=1, keepdim=True)
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
        
        return top1_acc, top3_acc, top5_acc


def main():
    args = parser.parse_args()

    num_sub = args.num_sub   
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []

    # "20250122/063410/33ac8673" # trained 1,2,3
    # run_id="20250125/110006/602ab94f"
    # run_id="20250125/193853/df62e743"
    # 20250126/081918/8cf47f3f
    ## "20250126/081918/8cf47f3f" trained 2,5 subjects with 1 being cross.
    runMan = RunManager(run_id=None) 
    runId = runMan.getRunID()
    
    for i in range(num_sub):
        if i!=num_sub-1:
            continue

        cal_num += 1
        starttime = datetime.datetime.now()
        # seed_n = np.random.randint(args.seed)
        seed_n = args.seed

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i+1))
        ie = IE(args, i + 1, runId=runId)

        # Acc, Acc3, Acc5 = ie.pretrain()
        # print('THE BEST ACCURACY IS ' + str(Acc))
        # print("Pretraining Done")

        Acc, Acc3, Acc5 = ie.fine_tune(subjectId=i+1, runId=runId) #Stage 1
        # # print('THE BEST ACCURACY IS ' + str(Acc))

        # Acc, Acc3, Acc5 = ie.feature_decompose(runId=runId) #Stage 2

        # Acc, Acc3, Acc5 = ie.test(subjectId=i+1, runId=runId, withFeatureDecompose=True, pretrained_subject_id=1)
        # Acc, Acc3, Acc5 = ie.test(subjectId=2, runId=runId, withFeatureDecompose=True, pretrained_subject_id=1)

        print('THE BEST ACCURACY IS ' + str(Acc))

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num+1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(result_path + f'result_sub{i+1}.csv')

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))