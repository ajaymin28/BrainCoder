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
from utils.losses import cosine_dissimilarity_loss, kld_loss


gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
result_path = '/home/jbhol/EEG/gits/BrainCoder/results/' 
model_idx = 'sub1_nice_ssl'

MODEL_SAVE_PATH = '/home/jbhol/EEG/gits/BrainCoder/model/pretrain/'
os.makedirs(MODEL_SAVE_PATH,exist_ok=True)
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--pretrain_epoch', default='5', type=int)
parser.add_argument('--num_sub', default=1, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
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

        # x = x.contiguous().view(x.size(0), -1)

        # Multihead Attention
        # attention_output, _ = self.attention(x, x, x)

        # Output layer
        # x = self.fc(attention_output)

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

# Define the EEG feature alignment network
class EEGAlignmentModel(nn.Module):
    def __init__(self,feature_dim,embedding_dim=768):
        super(EEGAlignmentModel, self).__init__()
        
        # Linear layers for transforming EEG features and subject embeddings
        self.eeg_transform = nn.Linear(feature_dim + embedding_dim, 256)
        self.subject_transform = nn.Linear(embedding_dim, 256)
        self.output_layer = nn.Linear(256+256, embedding_dim)

    def forward(self, eeg_feature, subject_embedding):
        # Concatenate EEG features and subject embedding
        combined_input = torch.cat((eeg_feature, subject_embedding), dim=-1)
        
        # Transform the combined input
        transformed_eeg = F.relu(self.eeg_transform(combined_input))
        subject_adjusted = F.relu(self.subject_transform(subject_embedding))
        
        # Apply further processing to obtain aligned EEG features
        aligned_eeg = self.output_layer(torch.cat((transformed_eeg, subject_adjusted), dim=-1))
        
        return aligned_eeg

class EEG_SubjectEmb(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1440, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    #     self.fc_mu = nn.Linear(embedding_dim, proj_dim)
    #     self.fc_logvar = nn.Linear(embedding_dim, proj_dim)
    # def forward(self, input):
    #     x =  super().forward(input)
    #     mu = self.fc_mu(x)
    #     logvar = self.fc_mu(x)

    #     # Reparameterization trick
    #     z = self.reparameterize(mu, logvar)
    #     return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std







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
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 8
        self.batch_size_img = 500 
        self.n_epochs = args.epoch
        self.pretrain_epoch = args.pretrain_epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = '/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz'
        self.img_data_path = '/home/jbhol/EEG/gits/NICE-EEG/dnn_feature/'
        self.test_center_path = '/home/jbhol/EEG/gits/NICE-EEG/dnn_feature/'
        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Enc_eeg = Enc_eeg().cuda()
        self.eeg_subj_emb = EEG_SubjectEmb().cuda()
        self.EEGAlignment = EEGAlignmentModel(feature_dim=1440,embedding_dim=768).cuda()

        
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.eeg_subj_emb = nn.DataParallel(self.eeg_subj_emb, device_ids=[i for i in range(len(gpus))])
        self.EEGAlignment = nn.DataParallel(self.EEGAlignment, device_ids=[i for i in range(len(gpus))])

        # self.Proj_eeg = Proj_eeg().cuda()
        # self.Proj_img = Proj_img().cuda()
        # self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

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

    def train(self):
        
        self.Enc_eeg.apply(weights_init_normal)
        self.eeg_subj_emb.apply(weights_init_normal)
        # self.EEGAlignment.apply(weights_init_normal)
        # self.Proj_eeg.apply(weights_init_normal)
        # self.Proj_img.apply(weights_init_normal)

        # train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        # train_img_feature, _ = self.get_image_data() 
        # test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        # shuffle the training data
        # train_shuffle = np.random.permutation(len(train_eeg))
        # train_eeg = train_eeg[train_shuffle]
        # train_img_feature = train_img_feature[train_shuffle]

        # val_eeg = torch.from_numpy(train_eeg[:740])
        # val_image = torch.from_numpy(train_img_feature[:740])

        # train_eeg = torch.from_numpy(train_eeg[740:])
        # train_image = torch.from_numpy(train_img_feature[740:])

        # dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        # val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        # val_dataset = EEG_Dataset2(args=self.args,nsub=1,
        #           agument_data=True,
        #           load_individual_files=True,
        #           subset="val",
        #           include_neg_sample=False,
        #           saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")
        
        # self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        # test_eeg = torch.from_numpy(test_eeg)
        # test_img_feature = torch.from_numpy(test_img_feature)
        # test_center = torch.from_numpy(test_center)
        # test_label = torch.from_numpy(test_label)
        # test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        # self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        # Optimizers
        # self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(),self.eeg_subj_emb.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        # self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters()), lr=self.lr, betas=(self.b1, self.b2))

        
        for sub_i in range(2,5):
            num = 0
            best_loss_val = np.inf
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


            # train_subject_embeddings_after_n_epochs = self.pretrain_epoch - (self.pretrain_epoch//4)
            # ReinitDone = False

            for e in range(self.pretrain_epoch):
                in_epoch = time.time()

                self.freeze_model(self.eeg_subj_emb, train=True)
                self.freeze_model(self.Enc_eeg, train=True)


                for i, (data1,data2,data3) in enumerate(self.dataloader):

                    (eeg, image_features, cls_label_id) = data1
                    (eeg_2, image_features_2, cls_label_id_2) = data2  # contrastive subject
                    # (eeg_3, image_features_3, cls_label_id_3) = data3

                    eegMean = torch.mean(eeg,dim=1,keepdim=True)
                    eegMean = Variable(eegMean.cuda().type(self.Tensor))

                    eegMean2 = torch.mean(eeg_2,dim=1,keepdim=True)
                    eegMean2 = Variable(eegMean2.cuda().type(self.Tensor))

                    labels = torch.arange(eegMean.shape[0])  # used for the loss
                    labels = Variable(labels.cuda().type(self.LongTensor))

                    # obtain the features
                    Lk = self.Enc_eeg(eegMean)
                    Lj = self.Enc_eeg(eegMean2)

                    SEk = self.eeg_subj_emb(Lk)
                    SEj = self.eeg_subj_emb(Lj)

                    # # Forward pass through the model
                    # aligned_eeg_1 = self.EEGAlignment(eeg_featuresMean, subject1_embeddings)
                    # aligned_eeg_2 = self.EEGAlignment(eeg_featuresMean2, subject2_embeddings)

                    # aligned_eeg_1 = aligned_eeg_1 / aligned_eeg_1.norm(dim=1, keepdim=True)
                    # aligned_eeg_2 = aligned_eeg_2 / aligned_eeg_2.norm(dim=1, keepdim=True)

                    # subject1_embeddings = subject1_embeddings / subject1_embeddings.norm(dim=1, keepdim=True)
                    # subject2_embeddings = subject2_embeddings / subject2_embeddings.norm(dim=1, keepdim=True)

                    # sub1_combined_features = sub1_combined_features / sub1_combined_features.norm(dim=1, keepdim=True)
                    # sub2_combined_features = sub2_combined_features / sub2_combined_features.norm(dim=1, keepdim=True)

                    # positive_distance = torch.nn.functional.pairwise_distance(sub1_combined_features, sub2_combined_features)
                    # negative_distance = torch.nn.functional.pairwise_distance(subject1_embeddings, subject2_embeddings)

                    # # Compute the Triplet Loss
                    # loss = torch.mean(torch.clamp(positive_distance - negative_distance + self.margin, min=0.0))

                    # loss = cosine_dissimilarity_loss(u=subject1_embeddings,v=subject2_embeddings)

                    # Lk = eeg_featuresMean / eeg_featuresMean.norm(dim=1, keepdim=True)
                    # Lj = eeg_featuresMean2 / eeg_featuresMean2.norm(dim=1, keepdim=True)

                    # Z =  torch.mean(torch.stack([Lk,Lj]), dim=0)
                    # Zj =  torch.mean(torch.stack([eeg_featuresMean2,subject2_embeddings]), dim=0)

                    # Zk =  torch.mean(torch.stack([Lk,SEk]), dim=0)
                    # Zj =  torch.mean(torch.stack([Lj,SEj]), dim=0)

                    # Zk =  torch.add(Lk,SEk)
                    # Zj =  torch.add(Lj,SEj)

                    # Zk = Zk / Zk.norm(dim=1, keepdim=True)
                    # Zj = Zj / Zj.norm(dim=1, keepdim=True)

                    # loss = self.get_contrastive_loss(feat1=Zk,feat2=Zj,labels=labels)

                    # loss += self.get_contrastive_loss(feat1=Zk,feat2=Lk,labels=labels)
                    # loss += self.get_contrastive_loss(feat1=Zj,feat2=Lj,labels=labels)

                    # SEdk =  torch.diff(torch.stack([SEk,Lk]), dim=0)
                    # SEdj =  torch.diff(torch.stack([SEj,Lj]), dim=0)

                    # SEk = SEk / SEk.norm(dim=1, keepdim=True)
                    # SEj = SEj / SEj.norm(dim=1, keepdim=True)

                    # mu_k =  torch.mean(Lk,dim=0)
                    # mu_j =  torch.mean(Lj,dim=0)
                    # std_k =  torch.std(Lk,dim=0)
                    # std_j =  torch.std(Lj,dim=0)


                    # SEd_Mean =  torch.mean(torch.stack([Lk,Lj]), dim=0)

                    SEdk_Mean =  torch.mean(Lk, dim=0)
                    SEdj_Mean =  torch.mean(Lj, dim=0)

                    # centering removing subject related info
                    # SEdk =  Lk - SEdk_Mean
                    # SEdj =  Lj - SEdj_Mean

                    # norm
                    Ek = Lk - SEdk_Mean
                    Ej = Lj - SEdj_Mean

                    SEjemb = Lk - Ek
                    SEkemb = Lj - Ej

                    # pairs are similar (label = 1) or dissimilar (label = 0).
                    # labels0 = torch.zeros(eegMean.shape[0])  # used for the loss
                    # labels0 = Variable(labels0.cuda().type(self.LongTensor))

                    # labels1 = torch.arange(eegMean.shape[0])  # used for the loss
                    # labels1 = Variable(labels1.cuda().type(self.LongTensor))

                    loss = self.get_contrastive_loss(feat1=SEk,feat2=SEkemb,labels=labels)
                    loss += self.get_contrastive_loss(feat1=SEj,feat2=SEjemb,labels=labels)
                    loss += self.get_contrastive_loss(feat1=Ek,feat2=Ej,labels=labels)


                    # loss = self.get_contrastive_loss(feat1=Ek,feat2=Lk,labels=labels)
                    # loss = self.get_contrastive_loss(feat1=Ej,feat2=Lj,labels=labels)
                    # loss += self.get_contrastive_loss(feat1=SEdk,feat2=SEdj,labels=labels0)  
                    # loss += cosine_dissimilarity_loss(u=SEdk,v=SEdj)

                    # distance = F.pairwise_distance(SEk, SEj, p=2)
                    # loss = 0.5 * torch.mean(distance ** 2)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if (e + 1) % 1 == 0:
                    if loss <=best_loss_val:
                        best_loss_val = loss
                        best_epoch = e + 1
                        # torch.save(self.Enc_eeg.module.state_dict(), os.path.join(MODEL_SAVE_PATH, model_idx + 'Enc_eeg_cls.pth') )
                        # if e>=train_subject_embeddings_after_n_epochs:
                        #     torch.save(self.eeg_subj_emb.module.state_dict(), os.path.join(MODEL_SAVE_PATH, model_idx + 'eeg_subj_emb.pth') )
                        # else:
                        #     torch.save(self.Enc_eeg.module.state_dict(), os.path.join(MODEL_SAVE_PATH, model_idx + 'Enc_eeg_cls.pth') )

                        # torch.save(self.EEGAlignment.module.state_dict(), os.path.join(MODEL_SAVE_PATH, model_idx + 'EEGAlignment.pth') )
                        # torch.save(self.eeg_subj_emb.module.state_dict(), os.path.join(MODEL_SAVE_PATH, model_idx + 'eeg_subj_emb.pth') )
                        torch.save(self.Enc_eeg.module.state_dict(), os.path.join(MODEL_SAVE_PATH, model_idx + 'Enc_eeg_cls.pth') )

                    print('Sub id', self.nSub, 
                          'Cross Sub id', sub_i,
                          'Epoch:', e,
                          '  best epoch: %d' % best_epoch,
                          '  loss: %.4f' % loss.detach().cpu().numpy(),
                        )
                    self.log_write.write('Epoch %d: loss: %.4f, \n'%(e, loss.detach().cpu().numpy()))

        return 0,0,0

    def fine_tune(self):

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


        self.Enc_eeg.load_state_dict(torch.load('/home/jbhol/EEG/gits/BrainCoder/model/pretrain/' + model_idx + 'Enc_eeg_cls.pth'), strict=False)
        # self.eeg_subj_emb.load_state_dict(torch.load('/home/jbhol/EEG/gits/BrainCoder/model/pretrain/' + model_idx + 'eeg_subj_emb.pth'), strict=False)
        # self.EEGAlignment.load_state_dict(torch.load('/home/jbhol/EEG/gits/BrainCoder/model/pretrain/' + model_idx + 'EEGAlignment.pth'), strict=False)

        self.freeze_model(self.Enc_eeg, train=False)
        # self.freeze_model(self.eeg_subj_emb, train=False)

        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()

        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        # Optimizers
        self.ft_optimizer = torch.optim.Adam(itertools.chain(self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))


        best_loss_val = np.inf
        for e in range(self.n_epochs):
            in_epoch = time.time()

            self.freeze_model(self.Proj_eeg, train=True)
            self.freeze_model(self.Proj_img, train=True)

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
                # subject_embeddings = self.eeg_subj_emb(eeg_features)

                # SEdk =  subject_embeddings - eeg_features
                # E = eeg_features - SEdk
                # sub_combined_features = torch.add(eeg_features, SEdk)

                # Z =  torch.mean(torch.stack([eeg_features,subject_embeddings]), dim=0)
                # Z =  torch.add(eeg_features,subject_embeddings)

                # Forward pass through the model
                # aligned_eeg_1 = self.EEGAlignment(eeg_features, subject_embeddings)
                # aligned_eeg_1 = aligned_eeg_1 / aligned_eeg_1.norm(dim=1, keepdim=True)
                # sub_combined_features = torch.add(eeg_features, subject_embeddings)

                mu_eeg =  torch.mean(eeg_features,dim=0)
                # std_eeg =  torch.std(eeg_features,dim=0)
                # centering and norm
                E =  eeg_features - mu_eeg
                # E = SEdk/std_eeg

                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(E)
                img_features = self.Proj_img(img_features)

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
                # self.freeze_model(self.eeg_subj_emb)
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
                        # vsubject_embeddings = self.eeg_subj_emb(veeg_features)

                        # vSEdk =  vsubject_embeddings - veeg_features
                        # E = veeg_features - vSEdk
                        # vsub_combined_features = torch.add(veeg_features, vSEdk)

                        # vZ =  torch.mean(torch.stack([veeg_features,vsubject_embeddings]), dim=0)
                        # vZ =  torch.add(veeg_features,vsubject_embeddings)

                        # vsub_combined_features = torch.add(veeg_features, vsubject_embeddings)
                        # valigned_eeg_1 = self.EEGAlignment(veeg_features, vsubject_embeddings)
                        # valigned_eeg_1 = valigned_eeg_1 / valigned_eeg_1.norm(dim=1, keepdim=True)

                        vmu_eeg =  torch.mean(veeg_features,dim=0)
                        # vstd_eeg =  torch.std(veeg_features,dim=0)
                        # centering and norm
                        vE =  veeg_features - vmu_eeg
                        # vE = vSEdk/vstd_eeg

                        veeg_features = self.Proj_eeg(vE)
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
                            best_epoch = e + 1
                            # torch.save(self.Enc_eeg.module.state_dict(), '/home/jbhol/EEG/gits/NICE-EEG/model/' + model_idx + 'Enc_eeg_cls.pth')
                            torch.save(self.Proj_eeg.module.state_dict(), '/home/jbhol/EEG/gits/NICE-EEG/model/' + model_idx + 'Proj_eeg_cls.pth')
                            torch.save(self.Proj_img.module.state_dict(), '/home/jbhol/EEG/gits/NICE-EEG/model/' + model_idx + 'Proj_img_cls.pth')

                print('Epoch:', e,
                      '  Cos eeg: %.4f' % loss_eeg.detach().cpu().numpy(),
                      '  Cos img: %.4f' % loss_img.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                      )
                self.log_write.write('Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n'%(e, loss_eeg.detach().cpu().numpy(), loss_img.detach().cpu().numpy(), vloss.detach().cpu().numpy()))
        
        return 0, 0, 0

    
    def freeze_model(self, model, train=False):
        for name, param in model.named_parameters():
            param.requires_grad = train
    
    def test(self, subject):

        self.nSub = subject

        _, _, test_eeg, test_label = self.get_eeg_data()
        # _, _ = self.get_image_data() 
        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        # shuffle the training data
        # train_shuffle = np.random.permutation(len(train_eeg))
        # train_eeg = train_eeg[train_shuffle]
        # train_img_feature = train_img_feature[train_shuffle]

        # val_eeg = torch.from_numpy(train_eeg[:740])
        # val_image = torch.from_numpy(train_img_feature[:740])

        # train_eeg = torch.from_numpy(train_eeg[740:])
        # train_image = torch.from_numpy(train_img_feature[740:])


        # dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        # self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        # val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        # self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_eeg = torch.from_numpy(test_eeg)
        # test_img_feature = torch.from_numpy(test_img_feature)
        self.test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)


        self.Enc_eeg.load_state_dict(torch.load('/home/jbhol/EEG/gits/BrainCoder/model/pretrain/' + model_idx + 'Enc_eeg_cls.pth'), strict=False)
        # self.eeg_subj_emb.load_state_dict(torch.load('/home/jbhol/EEG/gits/BrainCoder/model/pretrain/' + model_idx + 'eeg_subj_emb.pth'), strict=False)
        # self.EEGAlignment.load_state_dict(torch.load('/home/jbhol/EEG/gits/BrainCoder/model/pretrain/' + model_idx + 'EEGAlignment.pth'), strict=False)

        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()

        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        # * test part
        all_center = self.test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        # self.Enc_eeg.load_state_dict(torch.load('/home/jbhol/EEG/gits/NICE-EEG/model/' + model_idx + 'Enc_eeg_cls.pth'), strict=False)
        self.Proj_eeg.load_state_dict(torch.load('/home/jbhol/EEG/gits/NICE-EEG/model/' + model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load('/home/jbhol/EEG/gits/NICE-EEG/model/' + model_idx + 'Proj_img_cls.pth'), strict=False)

        # self.Enc_eeg.eval()
        # self.Proj_eeg.eval()
        # self.Proj_img.eval()
        # self.eeg_subj_emb.eval()
        # self.EEGAlignment.eval()

        self.freeze_model(self.Proj_img)
        self.freeze_model(self.Proj_eeg)
        self.freeze_model(self.Enc_eeg)
        # self.freeze_model(self.eeg_subj_emb)


        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))   

                teeg_features = self.Enc_eeg(teeg)
                # tsubject_embeddings = self.eeg_subj_emb(teeg_features)

                # tSEdk =  tsubject_embeddings - teeg_features
                # E = teeg_features - tSEdk

                # tsub_combined_features = torch.add(teeg_features, tSEdk)

                # tsub_combined_features = torch.add(teeg_features, tsubject_embeddings)
                # taligned_eeg_1 = self.EEGAlignment(teeg_features, tsubject_embeddings)
                # taligned_eeg_1 = taligned_eeg_1 / taligned_eeg_1.norm(dim=1, keepdim=True)

                # tZ =  torch.mean(torch.stack([teeg_features,tsubject_embeddings]), dim=0)
                # tZ =  torch.add(teeg_features,tsubject_embeddings)
                tmu_eeg =  torch.mean(teeg_features,dim=0)
                # tstd_eeg =  torch.std(teeg_features,dim=0)
                # centering and norm
                tE =  teeg_features - tmu_eeg
                # tE = tSEdk/tstd_eeg

                tfea = self.Proj_eeg(tE)
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
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))
        
        return top1_acc, top3_acc, top5_acc


def main():
    args = parser.parse_args()

    num_sub = args.num_sub   
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []
    
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
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        # print('THE BEST ACCURACY IS ' + str(Acc))
        # print("Pretraining Done")

        Acc, Acc3, Acc5 = ie.fine_tune()
        # print('THE BEST ACCURACY IS ' + str(Acc))

        Acc, Acc3, Acc5 = ie.test(subject=i+1)
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
    pd_all.to_csv(result_path + 'result.csv')

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))