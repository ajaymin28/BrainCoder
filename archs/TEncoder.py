import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    from flash_attn.modules.mha import MHA as FlashMHA
    FLASH_AVAILABLE = True
    print("Flash Attention is available.")
except ImportError:
    FLASH_AVAILABLE = False

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class EEGTransformerVAE(nn.Module):
    def __init__(self, channels=63, time_points=250, sessions=4, latent_dim=384, image_feature_dim=384, num_subjects=5, grl_alpha=1.0, encoder_only=False, use_flash=False, use_feature_confusion=False):
        super().__init__()
        self.channels = channels
        self.time_points = time_points
        self.sessions = sessions
        self.latent_dim = latent_dim
        self.grl_alpha = grl_alpha
        self.encoder_only = encoder_only
        self.use_flash = use_flash and FLASH_AVAILABLE
        self.use_feature_confusion = use_feature_confusion

        self.input_projection = nn.Linear(channels, latent_dim)
        self.pos_encoder = PositionalEncoding(latent_dim, max_len=time_points)

        if self.use_flash:
            self.transformer_encoder = FlashMHA(embed_dim=latent_dim, num_heads=8)
            print("Using FlashMHA")
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, batch_first=True,norm_first=True, nhead=8, dim_feedforward=512, dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.fc_mu = nn.Linear(latent_dim * time_points, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * time_points, latent_dim)

        self.subject_classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_subjects)
        )

        self.decoder_input = nn.Linear(latent_dim, latent_dim * time_points)
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=8, dim_feedforward=512, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.output_projection = nn.Linear(latent_dim, channels)

        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, image_feature_dim)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        batch, sessions, channels, time = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(batch * sessions, time, channels)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.reshape(batch * sessions, -1)
        mu = self.fc_mu(x)
        # logvar = F.softplus(self.fc_logvar(x)) + 1e-4
        logvar = self.fc_logvar(x)
        return mu.view(batch, sessions, -1), logvar.view(batch, sessions, -1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return torch.clamp(z, min=-5, max=5)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(z.size(0), self.time_points, self.latent_dim)
        tgt = torch.zeros_like(x)
        x = self.transformer_decoder(tgt, x)
        x = self.output_projection(x)
        return x.permute(0, 2, 1)

    def forward(self, x, step=None, total_steps=10000, kl_anneal=True):
        batch, sessions, _, _ = x.shape

        # Encoding
        mu, logvar = self.encode(x)  # Shape: [B, S, latent_dim]
        z = self.reparameterize(mu, logvar)  # [B, S, latent_dim]
        z_flat = z.view(batch * sessions, self.latent_dim)
        z_avg = z.mean(dim=1)
        mu_avg = mu.mean(dim=1)
        logvar_avg = logvar.mean(dim=1)

        # Decode if training full model
        recon = None
        if not self.encoder_only:
            recon = self.decode(z_flat).view(batch, sessions, self.channels, self.time_points)

        # Projection Head to image space
        projected_z = self.projection_head(z_avg)

        # Subject classifier logic
        if self.use_feature_confusion:
            subject_logits = self.subject_classifier(z_avg)  # No reversal
        else:
            # Grad Reverse scheduling
            alpha = self.grl_alpha
            if step is not None and total_steps is not None:
                alpha = self.grl_alpha * float(step) / total_steps  # linearly anneal
            reversed_z = grad_reverse(z_avg, alpha)
            subject_logits = self.subject_classifier(reversed_z)

        # # Optional: KL annealing scalar
        # kl_weight = 1.0
        # if kl_anneal and step is not None and total_steps is not None:
        #     kl_weight = min(1.0, step / (total_steps * 0.25))  # First 25% warmup

        
        return recon, mu_avg, logvar_avg, z_avg, subject_logits, projected_z, self.logit_scale, self.use_feature_confusion

        
   
    # def forward(self, x):
    #     batch, sessions, _, _ = x.shape
    #     mu, logvar = self.encode(x)
    #     z = self.reparameterize(mu, logvar)
    #     z_flat = z.view(batch * sessions, self.latent_dim)
    #     z_avg = z.mean(dim=1)
    #     mu_avg = mu.mean(dim=1)
    #     logvar_avg = logvar.mean(dim=1)

    #     if self.encoder_only:
    #         recon = None
    #     else:
    #         recon = self.decode(z_flat).view(batch, sessions, self.channels, self.time_points)

    #     projected_z = self.projection_head(z_avg)

    #     if self.use_feature_confusion:
    #         subject_logits = self.subject_classifier(z_avg)  # no grad reversal
    #     else:
    #         reversed_z = grad_reverse(z_avg, self.grl_alpha)
    #         subject_logits = self.subject_classifier(reversed_z)

    #     return recon, mu_avg, logvar_avg, z_avg, subject_logits, projected_z, self.logit_scale, self.use_feature_confusion


# Sample test for inference
if __name__ == "__main__":
    batch_size = 4
    sessions = 4
    channels = 63
    time_points = 250
    latent_dim = 768
    image_feature_dim = 768
    num_subjects = 5

    x = torch.randn(batch_size, sessions, channels, time_points)
    image_features = torch.randn(batch_size, image_feature_dim)
    subject_labels = torch.randint(0, num_subjects, (batch_size,))

    import sys
    sys.path.append('/home/jbhol/EEG/gits/BrainCoder')
    from utils.losses import vae_loss

    model = EEGTransformerVAE(
        channels=channels,
        time_points=time_points,
        sessions=sessions,
        latent_dim=latent_dim,
        image_feature_dim=image_feature_dim,
        num_subjects=num_subjects,
        encoder_only=True,
        use_flash=True,
        use_feature_confusion=True
    )

    recon, mu, logvar, z, subj_logits, proj, logit_scale, use_feature_confusion = model(x)

    loss, r_loss, kl_loss, c_loss, s_loss = vae_loss(
        recon, x, mu, logvar, proj, image_features, subj_logits, subject_labels, logit_scale, use_feature_confusion
    )

    print(f"Total loss: {loss:.4f}, Recon: {r_loss:.4f}, KL: {kl_loss:.4f}, Contrastive: {c_loss:.4f}, Subject: {s_loss:.4f}")
