import torch
import torch.nn as nn
from archs.common_layers import ResidualAdd

class EEGTransformer(nn.Module):
    def __init__(
        self,
        channels=63,
        time=250,
        patch_size=10,         # Length of each patch along time
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        drop_trans=0.1,
        proj_dim=768,
        drop_proj=0.5
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = time // patch_size

        # Patch embedding: Conv1d over time, kernel = patch_size, stride = patch_size
        self.patch_embed = nn.Conv1d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.elu = nn.ELU()
        self.bn = nn.BatchNorm1d(embed_dim)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=drop_trans,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self.feature_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
        )

    def forward(self, x):
        # x: (B, 63, 250) or (B, 1, 63, 250)
        if len(x.shape) == 4:
            x = x.squeeze(1)  # (B, 63, 250)

        # Patchify along time (Conv1d expects (B, C, L))
        x = self.patch_embed(x)         # (B, embed_dim, num_patches)
        x = self.bn(x)
        x = self.elu(x)
        x = x.transpose(1, 2)           # (B, num_patches, embed_dim)
        x = x + self.pos_embed[:, :x.shape[1], :]  # Add positional encoding
        x = self.transformer(x)         # (B, num_patches, embed_dim)
        x = self.norm(x)
        feat = x.mean(dim=1)            # (B, embed_dim)
        feat = self.feature_head(feat)  # (B, proj_dim)
        return feat