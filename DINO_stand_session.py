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
from tqdm import tqdm
from utils.gpu_utils import memory_stats

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
result_path = 'results' 
model_idx = 'likely-armadillo-64'
# checkpoints\likely-armadillo-64
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='dinov2', type=str)
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--num_sub', default=1, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=8000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')

import wandb


def weights_init_normal(m): 
    """
    Modified version to avoid the Error "object has no attribute 'weight'"
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from archs.DinIE import MultiModalEncoder, DINOHead
from archs.nice import Proj_img, Proj_eeg, Enc_eeg
from archs.EEGDinoEncoder import DynamicEEG2DEncoder


# Image2EEG
class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500 
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        # NICE-EEG\Data
        self.eeg_data_path = '/home/ja882177/EEG/gits/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz'
        self.img_data_path = '/home/ja882177/EEG/gits/NICE-EEG/dnn_feature/'
        self.test_center_path = '/home/ja882177/EEG/gits/NICE-EEG/dnn_feature/'
        # self.eeg_data_path = 'D:\\Datasets\\EEG DATASET\\things2\\NICE-EEG\\Data\\Things-EEG2\\Preprocessed_data_250Hz'
        # self.img_data_path = 'D:\\Datasets\\EEG DATASET\\things2\\NICE-EEG\\dnn_feature\\'
        # self.test_center_path = 'D:\\Datasets\\EEG DATASET\\things2\\NICE-EEG\\dnn_feature\\'
        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Enc_eeg = Enc_eeg().cuda()
        # self.Enc_eeg = ThingsEEGConv(channels=63,time=250,n_classes=1654,proj_dim=768).cuda()
        
        teacher_eeg_encoder = DynamicEEG2DEncoder(proj_dim=768).cuda()
        # DINO_Head_Dim = 128
        # teacher_dino_head = DINOHead(
        #     in_dim=768,                   # Feature dim from both encoders
        #     out_dim=DINO_Head_Dim,        # Embedding dim for DINO loss
        #     use_bn=False,
        #     norm_last_layer=True,
        #     nlayers=3,
        #     hidden_dim=2048,
        #     bottleneck_dim=256
        # )
        self.Enc_eeg = MultiModalEncoder(teacher_eeg_encoder, img_encoder=None, dino_head=None).cuda()
        # loaded_dict = torch.load(f"checkpoints\\{model_idx}\\dinov2_ckpt_epoch99_final.pth")
        # self.Enc_eeg.load_state_dict(loaded_dict["model_teacher"])

        self.Proj_eeg = Proj_eeg(embedding_dim=768, proj_dim=768).cuda()
        self.Proj_img = Proj_img(embedding_dim=768, proj_dim=768).cuda()
        # self.Proj_eeg = Proj_eeg(proj_dim=768).cuda()
        # self.Proj_img = Proj_img(proj_dim=768).cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')


    def get_eeg_data(self,train_img_feature,test_img_feature, train_sessions=[0,1,2,3], test_sessions=[i for i in range(80)],mean_data=False):
        train_data = []
        # train_label = []
        test_data = []
        updated_test_labels = np.arange(200)
        updated_train_img_feat = []
        updated_test_labels = []

        train_img_feat_len = train_img_feature.shape[0]


        if type(train_sessions)==list and len(train_sessions)>0:
            train_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
            train_data = train_data['preprocessed_eeg_data'][:,train_sessions,:,:] # (batch, sessions, channels, time)
            if mean_data:
                if len(train_data.shape)>3:
                    train_data = np.mean(train_data, axis=1)
                
                for feat_i in range(train_img_feat_len):
                    updated_train_img_feat.append(train_img_feature[feat_i])
                updated_train_img_feat = np.array(updated_train_img_feat,dtype=train_img_feature.dtype)

            else:
                # treat each session as one sample
                batch, sessions, ch, time = train_data.shape
                train_data = np.reshape(train_data,(batch*sessions, ch,time))

                for feat_i in range(train_img_feat_len):
                    for ses_len in range(len(train_sessions)):
                        updated_train_img_feat.append(train_img_feature[feat_i])
                updated_train_img_feat = np.array(updated_train_img_feat,dtype=train_img_feature.dtype)
                
            
            train_data = np.expand_dims(train_data, axis=1)

        if type(test_sessions)==list and len(test_sessions)>0:
            test_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
            test_data = test_data['preprocessed_eeg_data'][:,test_sessions,:,:] # (batch, sessions, channels, time)
            
            if mean_data:
                if len(test_data.shape)>3:
                    test_data = np.mean(test_data, axis=1)
                    
                for test_cls_i in range(200):
                    updated_test_labels.append(test_cls_i)
                updated_test_labels = np.array(updated_test_labels)
            else:
                # treat each session as one sample
                batch, sessions, ch, time = test_data.shape
                test_data = np.reshape(test_data,(batch*sessions, ch,time))
                batch, ch, time = test_data.shape
                for test_cls_i in range(200):
                    for session_i in test_sessions:
                        updated_test_labels.append(test_cls_i)
                updated_test_labels = np.array(updated_test_labels)

            test_data = np.expand_dims(test_data, axis=1)

        return train_data, updated_train_img_feat, test_data, updated_test_labels
    
    # def get_eeg_data(self, train_sessions=[0,1,2,3], test_sessions=[i for i in range(80)]):
    #     train_data = []
    #     train_label = []
    #     test_data = []
    #     test_label = np.arange(200)

    #     if type(train_sessions)==list and len(train_sessions)>0:
    #         train_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
    #         train_data = train_data['preprocessed_eeg_data'][:,train_sessions,:,:] # (batch, sessions, channels, time)
    #         if len(train_data.shape)>3:
    #             train_data = np.mean(train_data, axis=1)
            
    #         train_data = np.expand_dims(train_data, axis=1)

    #     if type(test_sessions)==list and len(test_sessions)>0:
    #         test_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
    #         test_data = test_data['preprocessed_eeg_data'][:,test_sessions,:,:] # (batch, sessions, channels, time)
    #         if len(test_data.shape)>3:
    #             test_data = np.mean(test_data, axis=1)
    #         test_data = np.expand_dims(test_data, axis=1)

    #     return train_data, train_label, test_data, test_label

    def get_image_data(self):
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_training.npy', allow_pickle=True)
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_test.npy', allow_pickle=True)

        train_img_feature = np.squeeze(train_img_feature)
        test_img_feature = np.squeeze(test_img_feature)

        return train_img_feature, test_img_feature
        
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self):
        
        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        train_img_feature_base, test_img_feature = self.get_image_data() 
        MEAN_TRAIN_DATA = True
        MEAN_TEST_DATA = True
        train_eeg, train_img_feature, _, _ = self.get_eeg_data(train_img_feature_base,test_img_feature, train_sessions=[0,1,2], test_sessions=[i for i in range(80)], mean_data=MEAN_TRAIN_DATA) # use seperate session for train and val
        _, _, test_eeg, test_label = self.get_eeg_data(train_img_feature_base,test_img_feature, train_sessions=None, test_sessions=[i for i in range(80)], mean_data=MEAN_TEST_DATA) # use seperate session for train and val
        val_eeg, val_img_feature, _, _ = self.get_eeg_data(train_img_feature_base,test_img_feature,train_sessions=[3], test_sessions=None, mean_data=MEAN_TRAIN_DATA) # different session for val
        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)
    
        # shuffle the training data

        # train_shuffle = np.random.permutation(len(train_eeg))
        # train_eeg = train_eeg[train_shuffle]
        # train_img_feature = train_img_feature[train_shuffle]

        # val_eeg = torch.from_numpy(train_eeg[:740])
        # val_image = torch.from_numpy(train_img_feature[:740])

        # train_eeg = torch.from_numpy(train_eeg[740:])
        # train_image = torch.from_numpy(train_img_feature[740:])

        train_eeg = torch.from_numpy(train_eeg)
        val_eeg = torch.from_numpy(val_eeg)
        train_image = torch.from_numpy(train_img_feature) # val_eeg is different session but image features remains same for different sessions
        val_image = torch.from_numpy(val_img_feature)
        # val_image = torch.from_numpy(train_img_feature)  # since sessions are different and image features are same for all session

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_eeg = torch.from_numpy(test_eeg)
        # test_img_feature = torch.from_numpy(test_img_feature)
        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(itertools.chain(self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf

        for e in tqdm(range(self.n_epochs)):
            in_epoch = time.time()

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img) in enumerate(self.dataloader):

                eeg = eeg.cuda().type(self.Tensor)
                # img = Variable(img.cuda().type(self.Tensor))
                img_features = img.cuda().type(self.Tensor)
                # label = Variable(label.cuda().type(self.LongTensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = labels.cuda().type(self.LongTensor)

                # obtain the features
                eeg_features = self.Enc_eeg([eeg],  ['eeg'], get_dino_head=False)
                eeg_features = torch.stack(eeg_features, dim=0).view(eeg.size(0), -1)
                # eeg_features = self.Enc_eeg(eeg)
                
                # print("eeg feat", eeg_features.size())

                eeg_features = self.Proj_eeg(eeg_features)
                # print("project eeg feat", eeg_features.size())
                img_features = self.Proj_img(img_features)

                # print("project image feat", img_features.size())

                # print("Target min/max:", labels.min().item(), labels.max().item())
                # print("Output shape:", outputs.shape)

                # eeg_features, eeg_cross_attn_weights = self.Enc_eeg.module.cross_attention_fuse(eeg_features=eeg_features,img_features=img_features) # cross atn between image and eeg

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

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            if (e + 1) % 1 == 0:
                self.Enc_eeg.eval()
                self.Proj_eeg.eval()
                self.Proj_img.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = veeg.cuda().type(self.Tensor)
                        vimg_features = vimg.cuda().type(self.Tensor)
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = vlabels.cuda().type(self.LongTensor)

                        veeg_features = self.Enc_eeg([veeg], ['eeg'] , get_dino_head=False)
                        veeg_features = torch.stack(veeg_features, dim=0).view(veeg.size(0), -1)
                        # veeg_features = self.Enc_eeg(veeg)
                        
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        # veeg_features, eeg_cross_attn_weights = self.Enc_eeg.module.cross_attention_fuse(eeg_features=veeg_features,img_features=vimg_features) # cross atn between image and eeg

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        # if vloss <= best_loss_val:
                        #     best_loss_val = vloss
                        #     best_epoch = e + 1
                        torch.save(self.Enc_eeg.module.state_dict(), './model/' + model_idx + 'Enc_eeg_cls.pth')
                        torch.save(self.Proj_eeg.module.state_dict(), './model/' + model_idx + 'Proj_eeg_cls.pth')
                        torch.save(self.Proj_img.module.state_dict(), './model/' + model_idx + 'Proj_img_cls.pth')

                
                loss_egg = loss_eeg.detach().cpu().numpy()
                loss_img = loss_img.detach().cpu().numpy()

                vloss_eeg = vloss_eeg.detach().cpu().numpy()
                vloss_img = vloss_img.detach().cpu().numpy()


                vloss = vloss.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy()

                print('Epoch:', e,
                      '  Cos eeg: %.4f' % loss_egg,
                      '  Cos img: %.4f' % loss_img,
                      '  loss val: %.4f' % vloss,
                      )
                self.log_write.write('Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n'%(e, loss_eeg, loss_img, vloss))

                wandb.log({
                    "Epoch": e if e>0 else 1,
                    "Cos eeg": loss_eeg,
                    "Cos img": loss_img,
                    "loss": loss,
                    "Cos veeg": vloss_eeg,
                    "Cos vimg": vloss_img,
                    "loss val": vloss,
                })

                memory_stats()


        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg.load_state_dict(torch.load('./model/' + model_idx + 'Enc_eeg_cls.pth'), strict=False)
        self.Proj_eeg.load_state_dict(torch.load('./model/' + model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load('./model/' + model_idx + 'Proj_img_cls.pth'), strict=False)

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = teeg.type(self.Tensor)
                tlabel = tlabel.type(self.LongTensor)
                all_center = all_center.type(self.Tensor)           

                tfea = self.Enc_eeg([teeg], ['eeg'] , get_dino_head=False)
                tfea = torch.stack(tfea, dim=0).view(teeg.size(0), -1)
                # tfea = self.Enc_eeg(teeg)
                tfea = self.Proj_eeg(tfea)


                # tfea = self.Proj_eeg(tfea)
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

        wandb.log({
            f"Subject_{self.nSub}/Top-1": top1_acc,
            f"Subject_{self.nSub}/Top-3": top3_acc,
            f"Subject_{self.nSub}/Top-5": top5_acc,
        })

        return top1_acc, top3_acc, top5_acc
        # writer.close()

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

def main():
    args = parser.parse_args()

    num_sub = args.num_sub   
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []

    for i in range(num_sub):

        cal_num += 1
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        dict_args = config_to_dict(args)
        wandb.init(
            project="DinoV2EEG_IMG_Align",          # your project name
            # mode="offline"
            config=dict_args,             # log all config parameters
            notes="[subject-1]  SESSION 1,2,3 NICE Model"
        )

        print('Subject %d' % (i+1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
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

    wandb.finish()

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))