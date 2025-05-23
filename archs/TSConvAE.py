import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from archs.subjectInvariant import SubjectDiscriminator
# --------------------------
# 1. Encoder Architecture
# --------------------------
# --------------------------
# Original Encoder Code (Kept Exactly as Provided)
# --------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, num_heads=4, output_dim=1440):
        super().__init__()
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

    def forward(self, x):
        x = self.tsconv(x)
        x = self.projection(x)
        # x, _ = self.attention(x, x, x)
        # x = self.fc(x)
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
    
class SubjectVarianceDecoder(nn.Sequential):
    def __init__(self, in_feature_dim, embedding=768):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(embedding, 256),
            nn.ReLU(),
            nn.Linear(256, in_feature_dim)
        )

    def forward(self, x_e, x_v):
        # decoded_f = self.decoder(torch.cat([x_e,x_v], dim=-1))
        # decoded_f = self.decoder(x_e + x_v)
        decoded_f = self.decoder(x_e)
        return decoded_f

class SubjectVarianceEncoder(nn.Sequential):
    def __init__(self, in_feature_dim, embedding=768):
        super().__init__()

        # Shared stimulus encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding)
        )

        self.sub_encoder = nn.Sequential(
            nn.Linear(in_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding)
        )


    def forward(self, x):
        # s_embeddings = []
        c_embeddings = self.encoder(x)
        s_embeddings = self.sub_encoder(x)
        return c_embeddings,s_embeddings


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

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class NICEEncoder(nn.Module):  # Changed from Sequential to Module
    def __init__(self, emb_size=40, ec_dim=768, es_dim=128, num_heads=4,dropout=0.2):
        super().__init__()
        # Original components
        self.patch_embed = PatchEmbedding(emb_size)
        self.flatten = FlattenHead()
        
        # New projection heads
        # self.subject_attention = nn.MultiheadAttention(embed_dim=1440, num_heads=num_heads, batch_first=True)
        # self.es_proj = nn.Linear(1440, es_dim)  # Subject features
        # self.norm_subject = nn.LayerNorm(es_dim)
        # self.dropout1 = nn.Dropout(dropout)
        
        # self.visual_stim_attention = nn.MultiheadAttention(embed_dim=1440, num_heads=num_heads, batch_first=True)
        # self.ec_proj = nn.Linear(1440, ec_dim)  # Class features
        # self.layernorm = nn.LayerNorm(ec_dim)
        # self.dropout2 = nn.Dropout(dropout)

        # self.eeg_proj = Proj_eeg(embedding_dim=ec_dim, proj_dim=ec_dim)
        # self.gelu = nn.GELU()

        # self.subject_variance_1 = nn.Parameter(torch.rand(size=(768,)))
        # self.subject_variance_2 = nn.Parameter(torch.rand(size=(768,)))
        # self.subj_variance = SubjectVarianceDecoder(in_feature_dim=1440,embedding=768)


    def forward(self, x):
        # Original forward path
        x = self.patch_embed(x)
        x_backbone = self.flatten(x)

        # split embeddings and decode
        # x_cls_embeddings,x_sub_embeddings,x_decoded = self.subj_variance(x_backbone)
        # x_eeg_proj = self.eeg_proj(x_backbone)
        return x_backbone 
    
        # sub_attn_variance, sub_attn_weight = self.subject_attention(x,x,x)
        # sub_attn_variance = self.norm_subject(sub_attn_variance)
        # sub_attn_variance = self.dropout1(sub_attn_variance)

        
        # x_eeg_proj = self.layernorm(x_eeg_proj)

        # New projections
        # return x_backbone, x_eeg_proj, torch.mean(x2_comm_proj,x_comm_proj)
# --------------------------
# Enhanced Decoder Architecture
# --------------------------
class NICEDecoder(nn.Module):
    def __init__(self, ec_dim=768, es_dim=64):
        super().__init__()
        
        # Feature recombination
        self.recombine = nn.Linear(ec_dim, 1440)
        
        # Reverse of FlattenHead + PatchEmbedding
        self.reshape = nn.Sequential(
            nn.Linear(1440, 40*1*36),
            Rearrange('b (c h w) -> b c h w', c=40, h=1, w=36)
        )
        
        # Reverse temporal-spatial convolution with exact parameters
        self.deconv = nn.Sequential(
            # Reverse spatial convolution (63 channels)
            nn.ConvTranspose2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            
            # Reverse temporal pooling
            nn.ConvTranspose2d(40, 40, (1, 51), stride=(1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            
            # Fixed final convolution for exact 250 samples
            nn.ConvTranspose2d(40, 1, (1, 25), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, ec, es):
        # Combine features
        # x = torch.cat([ec, es], dim=-1)
        x = self.recombine(ec)
        x = self.reshape(x)
        return self.deconv(x)

# --------------------------
# 3. Full Model with Losses
# --------------------------[]
class EEGCLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NICEEncoder()
        self.decoder = NICEDecoder()
        # self.clip_model, _ = clip.load("ViT-B/32")
        
        # Freeze CLIP model
        # for param in self.clip_model.parameters():
        #     param.requires_grad = False

    def forward(self, eeg_subj1, eeg_subj2, images):
        # Encode both subjects
        ec1, es1 = self.encoder(eeg_subj1)
        ec2, es2 = self.encoder(eeg_subj2)
        
        # # Get CLIP embeddings
        # with torch.no_grad():
        #     clip_emb = self.clip_model.encode_image(images).float()
        
        # Decode both subjects
        recon1 = self.decoder(ec1, es1)
        recon2 = self.decoder(ec2, es2)
        
        return ec1, es1, ec2, es2, recon1, recon2, clip_emb

    def compute_loss(self, eeg1, eeg2, recon1, recon2, ec1, ec2, es1, es2, clip_emb):
        # Reconstruction loss
        loss_recon = nn.MSELoss()(recon1, eeg1) + nn.MSELoss()(recon2, eeg2)
        
        # CLIP alignment loss
        loss_clip = nn.MSELoss()(ec1, clip_emb) + nn.MSELoss()(ec2, clip_emb)
        
        # Class feature consistency (same image should have similar Ec)
        loss_sim = nn.MSELoss()(ec1, ec2)
        
        # Subject variance separation (different subjects should have different Es)
        loss_var = -nn.MSELoss()(es1, es2)  # Negative to maximize difference
        
        # Combine losses
        total_loss = (
            0.4 * loss_recon +
            0.3 * loss_clip +
            0.2 * loss_sim +
            0.1 * loss_var
        )
        
        return total_loss

# --------------------------
# 4. Usage Example
# --------------------------
if __name__ == "__main__":
    # Initialize model
    model = EEGCLIPModel()
    
    # Example inputs
    batch_size = 32
    eeg_subj1 = torch.randn(batch_size, 1, 63, 1000)  # Subject 1 EEG
    eeg_subj2 = torch.randn(batch_size, 1, 63, 1000)  # Subject 2 EEG
    images = torch.randn(batch_size, 3, 224, 224)     # Stimulus images
    
    # Forward pass
    ec1, es1, ec2, es2, recon1, recon2, clip_emb = model(eeg_subj1, eeg_subj2, images)
    
    # Compute loss
    loss = model.compute_loss(eeg_subj1, eeg_subj2, recon1, recon2, 
                             ec1, ec2, es1, es2, clip_emb)
    
    print(f"Total loss: {loss.item():.4f}")