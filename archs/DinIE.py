import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, query, key):
        # query, key: (B, seq_len, d_model)
        attn_out, attn_weights = self.attn(query, key, key)
        return attn_out, attn_weights

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        attn = self.avg_pool(x)           # (B, C, 1)
        attn = self.fc(attn)              # (B, C, 1)
        return x * attn                   # (B, C, T)

class TemporalAttention(nn.Module):
    def __init__(self, time, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(time, time // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(time // reduction, time, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        x_perm = x.permute(0, 2, 1)      # (B, T, C)
        attn = self.avg_pool(x_perm)     # (B, T, 1)
        attn = self.fc(attn)             # (B, T, 1)
        out = x_perm * attn              # (B, T, C)
        return out.permute(0, 2, 1)      # (B, C, T)

class DinIE(nn.Module):
    def __init__(
        self,
        channels=63,
        time=250,
        n_classes=1654,
        proj_dim=768,
        drop_proj=0.5,
        adv_training=False,
        num_subjects=2,
        use_layernorm=True
    ):
        super().__init__()
        self.adv_training = adv_training
        self.num_subjects = num_subjects
        self.use_layernorm = use_layernorm

        # Spatial feature extraction (across channels)
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
        )

        # Temporal feature extraction (across time)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(time, 128, kernel_size=3, stride=1, padding=1),  # treat time as "channels"
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
        )

        self.channel_attn = ChannelAttention(32)        # 32 output channels from temporal_conv
        self.temporal_attn = TemporalAttention(32)      # 32 output timepoints from temporal_conv
        self.cross_attn_spatio_temporal = CrossAttention(d_model=32, n_heads=8) # cross attn between channels and time
        self.cross_eeg_img = CrossAttention(d_model=proj_dim, n_heads=4) # cross attn between eeg and image

        # Dynamic feature head input dimension: infer using dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, channels, time)
            z = self.spatial_conv(dummy)           # (1, 32, T')
            zc = self.channel_attn(z)              # (1, 32, T')
            z = zc.permute(0, 2, 1)                # (1, T', 32)
            zt = self.temporal_conv(z)             # (1, 32, 32)
            zt = self.temporal_attn(zt)            # (1, 32, 32)
            feat_dim = zt.reshape(1, -1).shape[1]

        # Feature projection head
        self.feature_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            ResidualAdd(nn.Sequential(
                nn.SiLU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj)
            )),
        )
        if self.use_layernorm:
            self.ln = nn.LayerNorm(proj_dim)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, n_classes),
        )

        # Optional domain discriminator for adversarial training
        if adv_training:
            self.domain_head = nn.Sequential(
                nn.Linear(proj_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_subjects)
            )
        else:
            self.domain_head = None

        # Contrastive loss utilities
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, alpha=0.5):
        # x: (B, C, T)
        if len(x.shape) == 4:
            x = x[:, 0, :, :]  # handle extra dimension

        z = self.spatial_conv(x)         # (B, 32, T')
        zc = self.channel_attn(z)        # (B, 32, T')
        z = zc.permute(0, 2, 1)          # (B, T', 32)
        zt = self.temporal_conv(z)       # (B, 32, 32)
        zt = self.temporal_attn(zt)      # (B, 32, 32)

        features = self.feature_head(zt.reshape(zt.shape[0], -1))  # (B, proj_dim)
        if self.use_layernorm:
            features = self.ln(features)

        return features