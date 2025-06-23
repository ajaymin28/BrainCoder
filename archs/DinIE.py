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
    





# --- Good Image Encoder (ViT, DINOv2 friendly) ---
class ImageEncoder(nn.Module):
    def __init__(self, proj_dim=768, vit_ckpt=None):
        super().__init__()
        # Use ViT-Base (DINOv2's default backbone). Requires timm
        self.vit = vit_base_patch16_224(pretrained=True if vit_ckpt is None else False)
        if vit_ckpt is not None:
            state = torch.load(vit_ckpt, map_location='cpu')
            self.vit.load_state_dict(state, strict=False)
        self.proj = nn.Sequential(
            nn.Linear(self.vit.embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
    def forward(self, x):
        # x: (B, 3, H, W), any H/W
        features = self.vit.forward_features(x)  # (B, embed_dim)
        features = self.proj(features)
        return features

def trunc_normal_(tensor, mean=0., std=1.):
    # Timm/trunc_normal compatible with recent PyTorch
    with torch.no_grad():
        return tensor.normal_(mean=mean, std=std)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=65536):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # weight_norm wrapper
        self.last_layer = torch.nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        # Fill only the weight_g parameter with 1 (per DINO)
        if hasattr(self.last_layer, "weight_g"):
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

# --- Shared Multi-modal Wrapper ---
class MultiModalEncoder(nn.Module):
    def __init__(self, eeg_encoder, img_encoder, dino_head):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.img_encoder = img_encoder
        self.dino_head = dino_head

    def forward(self, crops, crop_types=None, get_dino_head=True):
        outs = []

        if type(crops)==list:
            for i, crop in enumerate(crops):
                ctype = None
                if crop_types is not None:
                    ctype = crop_types[i]
                else:
                    if crop.dim() == 3:
                        ctype = 'eeg'
                    elif crop.dim() == 4 and crop.shape[1] == 3:
                        ctype = 'img'
                    else:
                        raise ValueError(f"Unknown crop shape: {crop.shape}")
                if ctype == 'eeg':
                    feat = self.eeg_encoder(crop)
                elif ctype == 'img':
                    feat = self.img_encoder(crop)
                else:
                    raise ValueError(f"Unknown crop type: {ctype}")
                
                if get_dino_head:
                    out = self.dino_head(feat)
                    outs.append(out)
                else: 
                    outs.append(feat)
        else:
            ctype = None
            if crop_types is not None:
                ctype = crop_types[i]
            else:
                if crop.dim() == 3:
                    ctype = 'eeg'
                elif crop.dim() == 4 and crop.shape[1] == 3:
                    ctype = 'img'
                else:
                    raise ValueError(f"Unknown crop shape: {crop.shape}")
            if ctype == 'eeg':
                feat = self.eeg_encoder(crop)
            elif ctype == 'img':
                feat = self.img_encoder(crop)
            else:
                raise ValueError(f"Unknown crop type: {ctype}")
            
            if get_dino_head:
                out = self.dino_head(feat)
                outs.append(out)
            else: 
                outs.append(feat)

        return outs