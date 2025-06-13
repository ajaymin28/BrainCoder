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
from utils.losses import cosine_dissimilarity_loss, kld_loss, SoftmaxThresholdLoss, CosineSimilarityLoss
from utils.losses import HyperParams, DINOLoss
from utils.common import CheckpointManager, RunManager
from archs.TSConvAE import NICEEncoder, SubjectVarianceEncoder, SubjectVarianceDecoder
from archs.FD import FeatureDecomposerV2, ContrastiveLoss,orthogonality_loss,reconstruction_loss
from archs.subjectInvariant import EEGDisentangler, SubjectDiscriminator


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
parser.add_argument('--train', default='0', type=int)
parser.add_argument('--cycles', default='1', type=int)
parser.add_argument('--pretrain_fd_epoch', default='25', type=int)
parser.add_argument('--pretrain_img_epoch', default='5', type=int)
parser.add_argument('--num_sub', default=1, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total'
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training.')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

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

        self.ssl_lr = 0.01
        self.ALPHA = 10
        self.BETA = 0.75
        self.GAMMA = 10.

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

        self.Enc_eeg = NICEEncoder(ec_dim=768, es_dim=768).cuda()
        # self.Dec_eeg = NICEDecoder(ec_dim=768, es_dim=768).cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Dec_eeg = nn.DataParallel(self.Dec_eeg, device_ids=[i for i in range(len(gpus))])

        # self.Proj_eeg = Proj_eeg().cuda()
        # self.Proj_img = Proj_img().cuda()
        # self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])


        self.sub_emb_encoder = SubjectVarianceEncoder(in_feature_dim=1440, embedding=768).cuda()
        self.sub_emb_encoder = nn.DataParallel(self.sub_emb_encoder, device_ids=[i for i in range(len(gpus))])

        # self.sub_emb_decoder = SubjectVarianceDecoder(in_feature_dim=1440, embedding=768).cuda()
        # self.sub_emb_decoder = nn.DataParallel(self.sub_emb_decoder, device_ids=[i for i in range(len(gpus))])

        # self.cont_eeg_learner = EEGDisentangler(feature_dim=768)
        # self.cont_eeg_learner = nn.DataParallel(self.cont_eeg_learner, device_ids=[i for i in range(len(gpus))])
        # self.cont_eeg_learner = nn.DataParallel(self.cont_eeg_learner, device_ids=[i for i in range(len(gpus))])

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
    

    def compute_loss(self, eeg1, eeg2, recon1, recon2, ec1, ec2, es1, es2, img_features):
        # Reconstruction loss
        loss_recon = 0.4 *  nn.MSELoss()(recon1, eeg1) + nn.MSELoss()(recon2, eeg2)
        
        # CLIP alignment loss
        loss_clip = 0.3 * nn.MSELoss()(ec1, img_features) + nn.MSELoss()(ec2, img_features)
        
        # Class feature consistency (same image should have similar Ec)
        loss_sim = 0.2 * nn.MSELoss()(ec1, ec2)
        
        # Subject variance separation (different subjects should have different Es)
        # loss_var = 0.1 * -nn.MSELoss()(es1, es2)  # Negative to maximize difference
        loss_var = 0.1 * -nn.CosineSimilarity()(es1, es2).mean()
        
        # Combine losses
        total_loss = (
            loss_recon +
            loss_clip +
            loss_sim 
            # loss_var
        )

        return total_loss, loss_recon,loss_clip,loss_sim,loss_var

    def contrastive_loss(self, Ej1, Ej2, Ek1, Ek2, margin=1.0):
        """
        Computes the contrastive loss for pairs of features.
        - Ej1, Ej2: feature vectors for subject J (same subject)
        - Ek1, Ek2: feature vectors for subject K (same subject)
        The function also computes the loss between a pair from J and a pair from K.
        """
        # Euclidean distances for same-subject pairs
        dist_j = F.pairwise_distance(Ej1, Ej2, p=2)
        dist_k = F.pairwise_distance(Ek1, Ek2, p=2)
        
        # Euclidean distances for different-subject pairs
        # Here, we use Ej1 vs Ek1 and Ej2 vs Ek2, but you can choose other pairings.
        dist_jk1 = F.pairwise_distance(Ej1, Ek1, p=2)
        dist_jk2 = F.pairwise_distance(Ej2, Ek2, p=2)
        
        # For the same-subject pairs, we want the distance to be small.
        loss_same = (dist_j + dist_k) / 2.0
        
        # For the different-subject pairs, we want the distance to be larger than the margin.
        loss_diff = torch.relu(margin - (dist_jk1 + dist_jk2) / 2.0)
        
        # Total loss is the sum of the same-subject loss and the penalized different-subject loss.
        total_loss = loss_same + loss_diff
        return total_loss.mean(), loss_same, loss_diff

    
    def cross_e_loss(self, f1,f2, temprature=0.2):
        p = torch.softmax(f1, dim=-1)  # Shape: (32, 768)
        q = torch.softmax(f2, dim=-1)  # Shape: (32, 768)
        # Compute cross-entropy loss
        cross_e_loss = -torch.mean(torch.sum(p/temprature * torch.log(q), dim=-1)) 
        return cross_e_loss 


    def norm_feat(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        return feat
    
    
    def feature_decompose(self, runId):

        self.Enc_eeg = NICEEncoder(ec_dim=768, es_dim=768).cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        cpm_Enc_eeg = CheckpointManager(prefix="Enc_eeg_SSL",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")

        self.subject_discriminator = SubjectDiscriminator(features_dim=1440).cuda()
        self.subject_discriminator = nn.DataParallel(self.subject_discriminator, device_ids=[i for i in range(len(gpus))])

        self.sub_emb_encoder = SubjectVarianceEncoder(in_feature_dim=1440, embedding=768).cuda()
        self.sub_emb_encoder = nn.DataParallel(self.sub_emb_encoder, device_ids=[i for i in range(len(gpus))])

        cpm_sub_emb_encoder = CheckpointManager(prefix="SubEmbEnc_SSL",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        # cpm_sub_emb_decoder = CheckpointManager(prefix="SubEmbDec",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")

        # subject_loss_fn = nn.BCEWithLogitsLoss()

        
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(),
                                                            self.sub_emb_encoder.parameters()), lr=self.ssl_lr, betas=(self.b1, self.b2))


        self.opt_sub_desc = torch.optim.Adam(self.subject_discriminator.parameters(), lr=self.ssl_lr)

        # number of subnjects
        NumberofSubject_Train = 2  # (1,2,3) 1 + 2

        epoch_counter = 0
        self.margin = 1.0
        best_epoch = 0
        best_loss_val = np.inf

        self.iters = 0

        for sub_i in range(2,2+NumberofSubject_Train):
            dataset = EEG_Dataset2(args=self.args,nsub=self.args.num_sub,
                        agument_data=True,
                        load_individual_files=True,
                        subset="train",
                        include_neg_sample=False,
                        preTraning=True,
                        cache_data=True,
                        constrastive_subject=sub_i,
                        saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")
                
            self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

            for e_min in range(self.pretrain_img_epoch):

                self.freeze_model(self.Enc_eeg, train=True)
                # self.freeze_model(self.Proj_img, train=True)
                self.freeze_model(self.sub_emb_encoder, train=True)
                # self.freeze_model(self.sub_emb_decoder, train=True)
                self.freeze_model(self.subject_discriminator, train=True)

                # start_steps = e * len(self.dataloader)
                # total_steps = self.pretrain_img_epoch * len(self.dataloader)

                for i, (data1,data2,data3) in enumerate(self.dataloader):

                    self.iters += 1
                    p = self.iters / (self.pretrain_img_epoch*NumberofSubject_Train * self.batch_size)
                    lambd = (2. / (1. + np.exp(-self.GAMMA * p))) - 1


                    lr = self.ssl_lr / (1. + self.ALPHA * p) ** self.BETA
                    self.optimizer.lr = lr
                    self.opt_sub_desc.lr = lr


                    (eeg, image_features, cls_label_id, subid) = data1
                    (eeg_2, image_features_2, cls_label_id_2, subid2) = data2  # contrastive subject
                    # (eeg_3, image_features_3, cls_label_id_3, subid3) = data3

                    image_features = Variable(image_features.cuda().type(self.Tensor))
                    image_features_2 = Variable(image_features_2.cuda().type(self.Tensor))

                    # Subject J  out of 4 sessions get mean of first two and second two
                    # Ej1 = torch.mean(eeg[:,0:2,:,:],dim=1,keepdim=True)
                    # Ej2 = torch.mean(eeg[:,2:,:,:],dim=1,keepdim=True)

                    # Ej1 = torch.mean(eeg,dim=1,keepdim=True)
                    random_idx = random.randint(0, 3)
                    Ej1 = Variable(eeg[:,random_idx,:,:].unsqueeze(1).cuda().type(self.Tensor))

                    random_idx = random.randint(0, 3)
                    # Ek1 = torch.mean(eeg_2,dim=1,keepdim=True)
                    Ek1 = Variable(eeg_2[:,random_idx,:,:].unsqueeze(1).cuda().type(self.Tensor))

                    self.optimizer.zero_grad()

                    # encode backbone
                    E_cj1_backbone  = self.Enc_eeg(Ej1)
                    E_ck1_backbone  = self.Enc_eeg(Ek1)

                    # encoding projection
                    E_cj1_cls_embeddings,E_cj1_sub_embeddings = self.sub_emb_encoder(E_cj1_backbone)
                    E_ck1_cls_embeddings,E_ck1_sub_embeddings = self.sub_emb_encoder(E_ck1_backbone)

                    # push embeddings towards image features
                    labels = torch.arange(eeg.shape[0])  # used for the loss
                    labels = Variable(labels.cuda().type(self.LongTensor))
                    loss_contrastive = self.get_contrastive_loss(E_cj1_cls_embeddings, image_features, labels=labels)
                    img_val_loss = self.get_contrastive_loss(E_ck1_cls_embeddings, image_features_2, labels=labels) 
                    loss_contrastive.backward()
                    self.optimizer.step()


                    self.opt_sub_desc.zero_grad()
                    
                    Discj1  = self.subject_discriminator(E_cj1_backbone.detach(), lambd)
                    Disck1  = self.subject_discriminator(E_ck1_backbone.detach(), lambd)

                    batch_size = eeg.size(0)
                    s1 = torch.ones(batch_size).cuda().type(self.LongTensor).requires_grad_(False)  # Subject labels
                    s2 = torch.zeros(batch_size).cuda().type(self.LongTensor).requires_grad_(False)

                    # Subject Classifier Prediction (Adversarial)
                    subject_labels = torch.cat([s1.float(), s2.float()]).unsqueeze(1)  # Subject labels (S1=0, S2=1)
                    subject_pred = torch.cat([Discj1, Disck1])
                    dom_loss = F.binary_cross_entropy(subject_pred, subject_labels)

                    dom_loss.backward()
                    self.opt_sub_desc.step()

                    # loss, temp = criterion_feature_dist(E_ck1_cls_embeddings,E_cj1_cls_embeddings, epoch_counter)
                    # loss = torch.abs(loss_fn(E_cj1_cls_embeddings, E_ck1_cls_embeddings).mean())
                    # loss = self.criterion_l2(E_cj1_cls_embeddings, E_ck1_cls_embeddings)

                    dom_loss_calc = lambd * dom_loss
                    loss = loss_contrastive - dom_loss_calc

                    


                if (e_min + 1) % 1 == 0:

                    if loss <=best_loss_val:
                        best_loss_val = loss
                        best_epoch = e_min

                        cpm_Enc_eeg.save_checkpoint(model=self.Enc_eeg,optimizer=self.optimizer,epoch="best")
                        # cpm_Proj_img.save_checkpoint(model=self.Proj_img,optimizer=self.optimizer,epoch="best")
                        cpm_sub_emb_encoder.save_checkpoint(model=self.sub_emb_encoder,optimizer=self.optimizer,epoch="best")
                        # cpm_sub_emb_decoder.save_checkpoint(model=self.sub_emb_decoder,optimizer=self.optimizer,epoch="best")


                    print('Sub id', self.nSub, 
                    ' CrossSub', sub_i,
                    f' Epoch:{e_min}',
                    ' best epoch: %d' % best_epoch,
                    ' Total loss: %.4f' % loss.detach().cpu().numpy(),
                    f' img_loss {loss_contrastive:.4f}', 
                    f' img_val_loss {img_val_loss:.4f}', 
                    f' dom_loss_calc: {dom_loss_calc:.4f} lamb: {lambd:.4f}',
                    )

                epoch_counter +=1

                    # self.log_write.write('Epoch %d: loss: %.4f, \n'%(e, loss.detach().cpu().numpy()))
            
            # load best model back
            cpm_Enc_eeg.load_checkpoint(model=self.Enc_eeg,optimizer=self.optimizer,epoch="best")
            cpm_sub_emb_encoder.load_checkpoint(model=self.sub_emb_encoder,optimizer=self.optimizer,epoch="best")

        return 0,0,0

    def finetune(self, runId):


        self.Enc_eeg = NICEEncoder(ec_dim=768, es_dim=768).cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        cpm_Enc_eeg = CheckpointManager(prefix="Enc_eeg_SSL",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")

        # self.subject_discriminator = SubjectDiscriminator(features_dim=768).cuda()
        # self.subject_discriminator = nn.DataParallel(self.subject_discriminator, device_ids=[i for i in range(len(gpus))])

        cpm_Enc_eeg.load_checkpoint(model=self.Enc_eeg,optimizer=None,epoch="best")

        self.sub_emb_encoder = SubjectVarianceEncoder(in_feature_dim=1440, embedding=768).cuda()
        self.sub_emb_encoder = nn.DataParallel(self.sub_emb_encoder, device_ids=[i for i in range(len(gpus))])

        cpm_sub_emb_encoder = CheckpointManager(prefix="SubEmbEnc_FT",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        # cpm_sub_emb_encoder.load_checkpoint(model=self.Enc_eeg,optimizer=None,epoch="best")

        # cpm_sub_emb_decoder = CheckpointManager(prefix="SubEmbDec",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        
        # self.Proj_eeg = Proj_eeg().cuda()
        # self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        # cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        # # cpm_Proj_eeg.load_checkpoint(model=self.Proj_eeg,optimizer=None,epoch="best")

        # self.Proj_img = Proj_img().cuda()
        # self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])
        # cpm_Proj_img = CheckpointManager(prefix="Proj_img",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        # # cpm_Proj_img.load_checkpoint(model=self.Proj_img,optimizer=None,epoch="best")

        # subject_loss_fn = nn.BCEWithLogitsLoss()

        
        self.optimizer = torch.optim.Adam(itertools.chain(self.sub_emb_encoder.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        
        for sub_i in range(2,3):
            best_loss_val = np.inf

            dataset = EEG_Dataset2(args=self.args,nsub=self.args.num_sub,
                    agument_data=True,
                    load_individual_files=True,
                    subset="train",
                    include_neg_sample=False,
                    preTraning=True,
                    cache_data=True,
                    constrastive_subject=sub_i,
                    saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata")

            self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
            self.margin = 1.0
            best_epoch = 0


            for e in range(self.pretrain_img_epoch):

                self.freeze_model(self.Enc_eeg, train=False)
                # self.freeze_model(self.Proj_img, train=True)
                self.freeze_model(self.sub_emb_encoder, train=True)
                # self.freeze_model(self.sub_emb_decoder, train=True)
                # self.freeze_model(self.subject_discriminator, train=True)


                start_steps = e * len(self.dataloader)
                total_steps = self.pretrain_img_epoch * len(self.dataloader)

                for i, (data1,data2,data3) in enumerate(self.dataloader):


                    (eeg, image_features, cls_label_id, subid) = data1
                    (eeg_2, image_features_2, cls_label_id_2, subid2) = data2  # contrastive subject
                    # (eeg_3, image_features_3, cls_label_id_3, subid3) = data3

                    image_features = Variable(image_features.cuda().type(self.Tensor))
                    image_features_2 = Variable(image_features_2.cuda().type(self.Tensor))

                    # Subject J  out of 4 sessions get mean of first two and second two
                    # Ej1 = torch.mean(eeg[:,0:2,:,:],dim=1,keepdim=True)
                    # Ej2 = torch.mean(eeg[:,2:,:,:],dim=1,keepdim=True)

                    Ej1 = torch.mean(eeg,dim=1,keepdim=True)
                    Ej1 = Variable(Ej1.cuda().type(self.Tensor))

                    Ek1 = torch.mean(eeg_2,dim=1,keepdim=True)
                    Ek1 = Variable(Ek1.cuda().type(self.Tensor))

                    # encode
                    E_cj1_backbone  = self.Enc_eeg(Ej1)
                    E_ck1_backbone  = self.Enc_eeg(Ek1)

                    # encoding head
                    E_cj1_cls_embeddings,E_cj1_sub_embeddings = self.sub_emb_encoder(E_cj1_backbone)
                    E_ck1_cls_embeddings,E_ck1_sub_embeddings = self.sub_emb_encoder(E_ck1_backbone)


                    labels = torch.arange(eeg.shape[0])  # used for the loss
                    labels = Variable(labels.cuda().type(self.LongTensor))

                    img_loss = self.get_contrastive_loss(E_cj1_cls_embeddings, image_features, labels)
                    img_val_loss = self.get_contrastive_loss(E_ck1_cls_embeddings, image_features_2, labels) 

                    loss = img_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                if (e + 1) % 1 == 0:

                    if loss <=best_loss_val:
                        best_loss_val = loss
                        best_epoch = e

                        # cpm_Enc_eeg.save_checkpoint(model=self.Enc_eeg,optimizer=self.optimizer,epoch="best")
                        # cpm_Proj_img.save_checkpoint(model=self.Proj_img,optimizer=self.optimizer,epoch="best")
                        cpm_sub_emb_encoder.save_checkpoint(model=self.sub_emb_encoder,optimizer=self.optimizer,epoch="best")
                        # cpm_sub_emb_decoder.save_checkpoint(model=self.sub_emb_decoder,optimizer=self.optimizer,epoch="best")


                    print('Sub id', self.nSub, 
                    ' CrossSub', sub_i,
                    ' Epoch:', e,
                    ' best epoch: %d' % best_epoch,
                    ' Total loss: %.4f' % loss.detach().cpu().numpy(), 
                    ' img_loss: %.4f' % img_loss.detach().cpu().numpy(), 
                    ' img_val_loss: %.4f' % img_val_loss.detach().cpu().numpy(), 
                    )


        return 0,0,0
    
    def freeze_model(self, model, train=False):
        for name, param in model.named_parameters():
            param.requires_grad = train
    
    def test(self, subjectId, runId):


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


        # cpm_Proj_eeg = CheckpointManager(prefix="Proj_eeg",
        #                                 base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        # cpm_Proj_img = CheckpointManager(prefix="Proj_img",
        #                                 base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        # cpm_cont_eeg_learner = CheckpointManager(prefix="EEGLearner",
        #                                          base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")

        self.Enc_eeg = NICEEncoder(ec_dim=768, es_dim=768).cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])

        cpm_Enc_eeg = CheckpointManager(prefix="Enc_eeg_SSL", base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        cpm_sub_emb_encoder = CheckpointManager(prefix="SubEmbEnc_FT",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")
        # cpm_sub_emb_decoder = CheckpointManager(prefix="SubEmbDec",base_dir=f"/home/jbhol/EEG/gits/BrainCoder/model/{runId}")

        cpm_Enc_eeg.load_checkpoint(model=self.Enc_eeg,optimizer=None,epoch="best")
        cpm_sub_emb_encoder.load_checkpoint(model=self.sub_emb_encoder,optimizer=None,epoch="best")
        # cpm_sub_emb_decoder.load_checkpoint(model=self.sub_emb_decoder,optimizer=None,epoch="best")
        

        # * test part
        all_center = self.test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0
        avg_loss = 0
        avg_loss_cnt = 0

        # self.freeze_model(self.Proj_img)
        # self.freeze_model(self.Proj_eeg)
        self.freeze_model(self.Enc_eeg)
        self.freeze_model(self.sub_emb_encoder)
        # self.freeze_model(self.sub_emb_decoder)

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))   

                E_backbone = self.Enc_eeg(teeg)
                E_cls_embeddings,E_sub_embeddings = self.sub_emb_encoder(E_backbone) 

                tfea =   E_cls_embeddings

                # tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # no use 100?
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()
            

            # print(f"Test loss: {avg_loss/avg_loss_cnt:.4f} for {avg_loss_cnt} samples")
            # top1_acc = 0
            # top3_acc = 0
            # top5_acc = 0

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

    # "20250210/171336/5e1f92bc" SSL contrastive loss
    runMan = RunManager(run_id="20250214/095819/617b2126") 
    runId = runMan.getRunID()
    print(runId)
    
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

        print(f"train: {args.train}")
        if args.train==1:
            Acc, Acc3, Acc5 = ie.feature_decompose(runId=runId) #Stage 2
        elif args.train==2:
            Acc, Acc3, Acc5 = ie.finetune(runId=runId) #Stage 2
        else:
            Acc, Acc3, Acc5 = ie.test(subjectId=i+1, runId=runId)
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
        
        print(runId)

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))