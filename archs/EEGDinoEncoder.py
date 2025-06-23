import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- EEG Encoder with global pooling ---
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        reduction = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (B, C, T)
        attn = x.mean(dim=2)      # (B, C)
        attn = self.fc(attn)      # (B, C)
        attn = attn.unsqueeze(2)  # (B, C, 1)
        return x * attn           # (B, C, T)
    
class TemporalAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        reduction = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (B, C, T)
        x_perm = x.permute(0, 2, 1)    # (B, T, C)
        attn = self.fc(x_perm)         # (B, T, C)
        attn = attn.permute(0, 2, 1)   # (B, C, T)
        return x * attn                # (B, C, T)

class EEGEncoder(nn.Module):
    def __init__(self, proj_dim=768, drop_proj=0.5, use_layernorm=True):
        super().__init__()
        self.use_layernorm = use_layernorm
        # Flexible convs, attention
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Conv1d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )
        self.channel_attn = ChannelAttention(32)
        self.temporal_attn = TemporalAttention(32)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_head = nn.Sequential(
            nn.Linear(32, proj_dim),
            nn.BatchNorm1d(proj_dim),
            ResidualAdd(nn.Sequential(
                nn.SiLU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj)
            )),
        )
        if self.use_layernorm:
            self.ln = nn.LayerNorm(proj_dim)
    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = x.reshape(B, 1, C*T)
        x = self.spatial_conv(x)  # (B, 32, C*T)
        x = x.view(B, 32, -1)
        x = self.channel_attn(x)
        x = self.temporal_attn(x)
        x = self.global_pool(x).squeeze(-1)  # (B, 32)
        features = self.feature_head(x)      # (B, proj_dim)
        if self.use_layernorm:
            features = self.ln(features)
        return features

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=10):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    def forward(self, x):
        # x: (B, C, T)
        x = self.proj(x)  # (B, embed_dim, new_T)
        x = x.transpose(1, 2)  # (B, new_T, embed_dim)
        return x

class FlexiblePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=256):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
    def forward(self, x):
        # x: (B, T, D)
        T = x.shape[1]
        if T > self.pos_embed.shape[1]:
            raise ValueError("Increase max_len for positional encoding.")
        pe = F.interpolate(
            self.pos_embed[:, :T].transpose(1, 2), size=T, mode='linear', align_corners=False
        ).transpose(1, 2)
        return x + pe

class EEGTransformerEncoder(nn.Module):
    def __init__(
        self, in_channels=63, embed_dim=192, num_layers=4, num_heads=4,
        patch_size=10, max_len=256, mlp_dim=512
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.pos_enc = FlexiblePositionalEncoding(embed_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        # x: (B, C, T)
        x = self.patch_embed(x)  # (B, T', D)
        x = self.pos_enc(x)      # (B, T', D)
        x = self.transformer(x)  # (B, T', D)
        x = self.norm(x)
        # Mean pooling (global embedding)
        x = x.mean(dim=1)        # (B, D)
        return x

class DINOV2EEGEncoder(nn.Module):
    def __init__(
        self, in_channels=63, embed_dim=192, patch_size=10, num_layers=4, num_heads=4, max_len=250, mlp_dim=512, proj_dim=786
    ):
        super().__init__()
        self.encoder = EEGTransformerEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
            mlp_dim=mlp_dim,
        )
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x):
        features = self.encoder(x)
        proj = self.proj_head(features)
        return proj
    

class DynamicEEG2DEncoder(nn.Module):
    def __init__(self, proj_dim=768, drop_proj=0.5):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, C, T)
        #     nn.BatchNorm2d(64),
        #     nn.ELU(),
        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # (B, 32, C, T)
        #     nn.BatchNorm2d(32),
        #     nn.ELU(),
        #     nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), # (B, 16, C, T)
        #     nn.BatchNorm2d(16),
        #     nn.ELU(),
        #     nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1), # (B, 4, C, T)
        #     nn.BatchNorm2d(8),
        #     nn.ELU(),
        # )
        # # Global pooling: output is always (B, 32,32)
        # self.global_pool = nn.AdaptiveAvgPool2d((63,100))  # (B, 4, 63, 100)

        # FROM THE NICE Paper but customized for 63,100 input and needed output (32,32)
        # self.tsconv = nn.Sequential(
        #     nn.Conv2d(1, 40, (1, 25), (1, 1)),
        #     nn.AvgPool2d((1, 51), (1, 5)),
        #     nn.BatchNorm2d(40),
        #     nn.ELU(),
        #     nn.Conv2d(40, 40, (63, 1), (1, 1)),
        #     nn.BatchNorm2d(40),
        #     nn.ELU(),
        #     nn.Dropout(0.5),
        # )
        

        # Input shape: (2, 1, 63, 250)
        # self.tsconv = nn.Sequential(
        #     nn.Conv2d(1, 64, (1, 8), (1, 1)),   # torch.Size([2, 64, 63, 243])
        #     nn.BatchNorm2d(64),                 # torch.Size([2, 64, 63, 243])
        #     nn.ELU(),                           # torch.Size([2, 64, 63, 243])
        #     nn.AvgPool2d((1, 25), (1, 2)),      # torch.Size([2, 64, 63, 110])

        #     nn.Conv2d(64, 40, (32, 1), (1, 1)), # torch.Size([2, 32, 14, 107])  
        #     nn.BatchNorm2d(40),                 # torch.Size([2, 32, 14, 107])
        #     nn.ELU(),                           # torch.Size([2, 32, 14, 107])
        #     nn.Dropout(0.5),                    # torch.Size([2, 32, 14, 107])
        # )

        # self.global_pool = nn.AdaptiveAvgPool2d((4,32))  # Pool (B, filters, 10, 32)

        self.tsconv_1d = nn.Sequential(
            nn.Conv1d(63, 96, 3, 1, 0),        # (Conv1d): torch.Size([5, 40, 248])
            nn.BatchNorm1d(96),             # (BatchNorm1d): torch.Size([5, 40, 248])
            nn.ELU(),                       # (ELU): torch.Size([5, 40, 248])
            nn.AvgPool1d(3,2,1),            # (AvgPool1d): torch.Size([5, 40, 124])

            nn.Conv1d(96, 40, 1, 1, 0),     # (Conv1d): torch.Size([5, 32, 63])
            nn.BatchNorm1d(40),             # (BatchNorm1d): torch.Size([5, 32, 63])
            nn.ELU(), 
            
            nn.Conv1d(40, 32, 2, 1, 0),     # (Conv1d): torch.Size([5, 32, 63])
            nn.BatchNorm1d(32),             # (BatchNorm1d): torch.Size([5, 32, 63])
            nn.ELU(),                       # (ELU): torch.Size([5, 32, 63])
            nn.AvgPool1d(3,2,1),            # (AvgPool1d): torch.Size([5, 32, 32])
        )
        self.global_pool_1d = nn.AdaptiveAvgPool1d(32)   # fix time 32


        self.flatten = nn.Flatten(1)
        self.feature_head = nn.Sequential(
            nn.Linear(1024, proj_dim),
            nn.BatchNorm1d(proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
        )



        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, C, T)
        #     nn.BatchNorm2d(64),
        #     nn.ELU(),
        #     nn.AdaptiveAvgPool2d((64,128)),
        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # (B, 32, C, T)
        #     nn.BatchNorm2d(32),
        #     nn.ELU(),
        #     nn.AdaptiveAvgPool2d((32,64)),
        #     nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), # (B, 16, C, T)
        #     nn.BatchNorm2d(16),
        #     nn.ELU(),
        #     nn.AdaptiveAvgPool2d((32,32)),
        #     nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1), # (B, 4, C, T)
        #     nn.BatchNorm2d(4),
        #     nn.ELU(),
        #     nn.AdaptiveAvgPool2d((32,32))
        # )
        # print(self.encoder)

        
        # self.flatten = nn.Flatten(1)
        # self.feature_head = nn.Sequential(
        #     nn.Linear(8448, proj_dim),
        #     nn.BatchNorm1d(proj_dim),
        #     ResidualAdd(nn.Sequential(
        #         nn.GELU(),
        #         nn.Linear(proj_dim, proj_dim),
        #         nn.Dropout(drop_proj),
        #     )),
        # )

       
        # # Experimental
        # self.tsconv = nn.Sequential(
        #     nn.Conv2d(1, 64, (1, 3), (1, 1)), 
        #     nn.BatchNorm2d(64),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 2), (1, 3)),
        #     nn.Conv2d(64, 128, (32, 1), (1, 1)),
        #     nn.BatchNorm2d(128),
        #     nn.ELU(),
        #     nn.Dropout(0.5),
        #     nn.Conv2d(128, 256, (32, 1), (1, 1)),
        #     nn.BatchNorm2d(256),
        #     nn.ELU(),
        #     nn.Dropout(0.5),
        # )
        # self.feature_head = nn.Sequential(
        #     nn.Linear(256, proj_dim),
        #     nn.BatchNorm1d(proj_dim),
        #     ResidualAdd(nn.Sequential(
        #         nn.GELU(),
        #         nn.Linear(proj_dim, proj_dim),
        #         nn.Dropout(drop_proj),
        #     )),
        # )

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

    def nice_contrastive_loss(self, eeg_features, img_features):
        vlabels = torch.arange(eeg_features.shape[0])
        vlabels = vlabels.cuda().type(self.LongTensor)

        eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
        img_features = img_features / img_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_eeg = logit_scale * eeg_features @ img_features.t()
        logits_per_img = logits_per_eeg.t()

        loss_eeg = self.criterion_cls(logits_per_eeg, vlabels)
        loss_img = self.criterion_cls(logits_per_img, vlabels)
        loss = (loss_eeg + loss_img) / 2

        return loss

    def forward(self, x):
        # x: (B, 1, C, T) with variable C and T across batches (but fixed within batch)
        
        # if len(x.shape)==3:
        #     # batch, channel, time
        #     x = x.unsqueeze(1) # add extra channel dim batch, 1, channel, time

        # For 1D Conv
        if len(x.shape) == 4:
            x = x.squeeze(1)

        # # print("x", x.shape)
        # z = self.tsconv(x)
        # # print("tsconv", z.shape)
        # z = self.global_pool(z)
        # # print("global_pool", z.shape)

        z = self.tsconv_1d(x)
        z = self.global_pool_1d(z)

        z = self.flatten(z)
        # print("flatten", z.shape)
        z = self.feature_head(z)
        # print("feature_head", z.shape)
        
        # print("in size", x.shape)
        # z = self.encoder(x)
        # print("encoder", z.shape)
        # z = self.global_pool(z)
        # print("global_pool", z.shape)
        # z = self.tsconv(z)
        # print("tsconv", z.shape)
        # z = self.flatten(z)
        # print("flatten", z.shape)
        # z = self.feature_head(z)
        # print("feature_head", z.shape)
        return z