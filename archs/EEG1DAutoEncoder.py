import torch.nn as nn
from archs.common_layers import ResidualAdd
import torch.nn.functional as F

class Dynamic1DEEGAEncoder(nn.Module):
    def __init__(self, ch=63, time_int=250, proj_dim=768, drop_proj=0.5):
        super().__init__()

        self.tsconv_1d = nn.Sequential(
            nn.Conv1d(63, 40, 25, 1, 0),    # [B, 40, 226]
            nn.BatchNorm1d(40),
            nn.ELU(),
            nn.AvgPool1d(3,2,1),            # [B, 40, 113]
            nn.Conv1d(40, 32, 2, 1, 0),     # [B, 32, 112]
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(3,3,1),            # [B, 40, 113]
            nn.Dropout(0.5),
        )

        self.global_pool_1d = nn.AdaptiveAvgPool1d(32)  # For fixed-size latent
        self.flatten = nn.Flatten(1)

        self.unflatten = nn.Unflatten(1, (32, 32))

        self.projection = nn.Sequential(
            nn.Linear(1024, proj_dim),
            nn.BatchNorm1d(proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
        )

        self.decoder = nn.Sequential(
            # Start: (B, 32, 32)
            nn.Upsample(size=112, mode='linear'),                 # (B, 32, 112)
            nn.ConvTranspose1d(32, 40, kernel_size=3, stride=1),   # (B, 40, 114)
            nn.BatchNorm1d(40),
            nn.ELU(),
            nn.Upsample(size=226, mode='linear'),                 # (B, 40, 226)
            nn.ConvTranspose1d(40, 63, kernel_size=25, stride=1),  # (B, 63, 250)
            nn.BatchNorm1d(63),
            nn.ELU(),
            nn.Conv1d(63, 63, kernel_size=1),                      # (B, 63, 250) (optional smoothing)
        )


    def forward(self, x):

        dim4_input = False
        if len(x.shape) == 4:                 # (B, 1, 63, 250)
            x = x.squeeze(1)                  # (B, 63, 250)
            dim4_input = True

        z = self.tsconv_1d(x)                 # (B, 32, ~112)
        z = self.global_pool_1d(z)            # (B, 32, 32)
        latent_z = self.flatten(z)            # (B, 32*32)

        proj_z = self.projection(latent_z)    # (B, 768)
        
        z_d = self.unflatten(latent_z)        # (B, 32, 32)
        recon = self.decoder(z_d)             # (B, 63, 250)
        if dim4_input:
            recon = recon.unsqueeze(1)        # (B, 1, 63, 250)

        return latent_z,recon,proj_z


class ResidualBlock1D(nn.Module):
    def __init__( self,in_ch,out_ch,k,s,p,pool_k,pool_s,pool_p):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, s, p),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.AvgPool1d(pool_k, pool_s, pool_p)
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = self.main(x)
        skip = self.skip(x)
        if out.shape[-1] != skip.shape[-1]:
            skip = F.interpolate(skip, size=out.shape[-1], mode='nearest')
        return out + skip

class TSConv1DResNet(nn.Module):
    def __init__(self, proj_dim, drop_proj=0.5):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock1D(in_ch=63, out_ch=40, k=15, s=2, p=0, pool_k=3, pool_s=1, pool_p=0),
            ResidualBlock1D(in_ch=40, out_ch=32, k=7, s=2, p=0, pool_k=3, pool_s=1, pool_p=0),
            nn.AdaptiveAvgPool1d(32),
            nn.Dropout(0.5),
            nn.Flatten(1),
        )

        self.projection = nn.Sequential(
            nn.Linear(1024, proj_dim),
            nn.BatchNorm1d(proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
        )

    def forward(self, x):
        if len(x.shape) == 4:                 # (B, 1, 63, 250)
            x = x.squeeze(1)                  # (B, 63, 250)
        x = self.blocks(x)
        x = self.projection(x)
        return x