#Original model presented in: C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, M. Shah, Deep Learning Human Mind for Automated Visual Classification, CVPR 2017 
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    
    def getNegativeLoss(self, anchor, negative):
        # Compute Euclidean distance
        distance = F.pairwise_distance(anchor, negative)
        distance = torch.mean(distance ** 2)
        # Compute contrastive loss for the negative pair
        loss = max(0, self.margin - distance) ** 2
        return loss

    def getPositiveLoss(self, anchor, positive):
        distance = F.pairwise_distance(anchor, positive)
        loss_positive = torch.mean(distance ** 2)
        return loss_positive
    
    def forward(self, anchor, positive, negative):

        neg_loss = self.getNegativeLoss(anchor, negative)
        pos_loss = self.getPositiveLoss(anchor, positive)

        loss = neg_loss + pos_loss
        return loss

class Model(nn.Module):
    def __init__(
        self, 
        input_size=63, 
        lstm_size=200, 
        lstm_layers=4, 
        output_size=768, 
        include_top=True, 
        n_classes=200,
        dropout_rate=0.3,
        bidirectional=False,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        self.include_top = include_top
        self.bidirectional = bidirectional

        # LSTM with dropout for regularization
        self.lstm = nn.LSTM(
            input_size, 
            lstm_size, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=self.bidirectional , 
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )

        # Fully connected layers
        if self.bidirectional:
            self.L0 = nn.Linear(lstm_size * 2, output_size)
        else:
            self.L0 = nn.Linear(lstm_size, output_size)
        
        # Classifier if include_top is True
        self.classifier = nn.Linear(output_size, n_classes)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(output_size)

        # Dropout layer for the output layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Forward through LSTM
        x, _ = self.lstm(x)  # Shape: (batch_size, seq_length, hidden_dim)
        
        # Extract the last hidden state
        x = x[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Apply GELU activation
        x = F.gelu(x)

        # Forward through the linear layer and batch normalization
        x = self.L0(x)
        x = self.batch_norm(x)

        # Apply dropout
        x = self.dropout(x)

        # Pass through classifier if include_top is True
        if self.include_top:
            x = self.classifier(x)
        
        return x


# class Model(nn.Module):

#     def __init__(self, input_size=100, lstm_size=63, lstm_layers=4, output_size=768, include_top=True, n_classes=40):
#         # Call parent
#         super().__init__()
#         # Define parameters
#         self.input_size = input_size
#         self.lstm_size = lstm_size
#         self.lstm_layers = lstm_layers
#         self.output_size = output_size
#         self.include_top = include_top

#         # Define internal modules
#         self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True, bidirectional=True)
#         self.L0 = nn.Linear(lstm_size*2, output_size)

#         self.classifier = nn.Linear(output_size,n_classes)
        
#     def forward(self, x):
#         # Forward LSTM and get final state
#         x, _ = self.lstm(x)   # Shape: (batch_size, seq_length, hidden_dim)
#         x = x[:, -1, :]  # Shape: (batch_size, hidden_dim)
#         x = F.gelu(x)
#         # Forward output
#         x = self.L0(x)
#         if self.include_top:
#             x = self.classifier((x))
#         return x
