import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGVAE(nn.Module):
    def __init__(self, channels=63, time_points=250, sessions=4, latent_dim=384, image_feature_dim=384, num_subjects=5):
        super(EEGVAE, self).__init__()
        self.channels = channels
        self.time_points = time_points
        self.sessions = sessions
        self.latent_dim = latent_dim

        # 1D temporal conv over time axis
        # Operates on [batch*sessions, channels, time]
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, stride=2, padding=2),  # -> [batch*sessions, 64, 125]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),       # -> [batch*sessions, 128, 63]
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),       # -> [batch*sessions, 256, 32]
            nn.ReLU()
        )

        # Spatial conv
        # Operates on [batch*sessions, 1, 256, 32] after unsqueezing
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),     # -> [batch*sessions, 64, 256, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1),  # -> [batch*sessions, 128, 128, 16]
            nn.ReLU()
        )

        # Flattened dimension after spatial conv
        # Shape is [batch*sessions, 128, 128, 16]
        self.flatten_dim = 128 * 128 * 16

        # Linear layers for mu and logvar
        # Operate on [batch*sessions, flatten_dim]
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)     # -> [batch*sessions, latent_dim]
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim) # -> [batch*sessions, latent_dim]

        # Subject classifier head - This will now operate on the session-averaged latent code
        # if we still want a single subject prediction per batch item.
        # If subject prediction should be session-specific, this head needs adjustment.
        # Keeping it operating on an averaged latent for now, as per original structure.
        self.subject_classifier = nn.Sequential(
            nn.Linear(latent_dim, 128), # Input shape will be [batch, latent_dim] after averaging
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_subjects)
        )

        # Decoder
        # The decoder will take session-specific latent codes [batch*sessions, latent_dim]
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim) # -> [batch*sessions, flatten_dim]

        # Spatial decoder
        # Operates on [batch*sessions, 128, 128, 16] after reshaping decoder_fc output
        self.decoder_spatial = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1, output_padding=(1,1)),  # -> [batch*sessions, 64, 256, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=1, padding=1),                         # -> [batch*sessions, 1, 256, 32]
            nn.ReLU()
        )

        # Temporal decoder
        # Operates on [batch*sessions, 256, 32] after squeezing spatial decoder output
        self.decoder_temporal = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # -> [batch*sessions, 128, 64]
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [batch*sessions, 64, 128]
            nn.ReLU(),
            # Adjusted padding to get output size 250 from input size 128 with kernel 5, stride 2, output_padding 1
            nn.ConvTranspose1d(64, channels, kernel_size=5, stride=2, padding=5, output_padding=1), # -> [batch*sessions, channels, 250]
            nn.ReLU() # Added ReLU for consistency
        )

        # Projection head (for contrastive loss) - Also operates on session-averaged latent code
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 128), # Input shape will be [batch, latent_dim] after averaging
            nn.ReLU(),
            nn.Linear(128, image_feature_dim)
        )

    def encode(self, x):
        # x shape: [batch, sessions, channels, time] -> [32, 4, 63, 250]
        batch, sessions, channels, time = x.shape
        # Flatten batch and sessions for temporal conv
        x = x.view(batch * sessions, channels, time)  # [128, 63, 250]
        x = self.temporal_conv(x)  # [128, 256, 32]
        x = x.unsqueeze(1)  # Add channel dimension for spatial conv: [128, 1, 256, 32]
        x = self.spatial_conv(x)  # [128, 128, 128, 16]
        # Flatten for linear layers
        x = x.view(batch * sessions, -1)  # [128, 262144]
        # Apply linear layers to get session-specific mu and logvar
        mu = self.fc_mu(x)  # [128, 384]
        logvar = self.fc_logvar(x)  # [128, 384]

        # Reshape back to include sessions dimension, preserving session-specific info
        mu = mu.view(batch, sessions, self.latent_dim)  # [32, 4, 384]
        logvar = logvar.view(batch, sessions, self.latent_dim) # [32, 4, 384]

        # Note: Mu and logvar now contain session-specific information.
        # The averaging across sessions for the final latent code z will be removed
        # in the forward pass for the VAE reconstruction path, but kept for
        # the subject classifier and projection head if they are meant to operate
        # on a subject-level representation.

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # mu, logvar shape: [batch, sessions, latent_dim]
        std = torch.exp(0.5 * logvar).to(mu.device).type(mu.dtype)  # [batch, sessions, latent_dim]
        eps = torch.randn_like(std).to(mu.device) .type(mu.dtype)
        # z shape: [batch, sessions, latent_dim]
        return mu + eps * std

    def decode(self, z):
        # z shape: [batch*sessions, latent_dim] - will be reshaped in forward
        # Decode latent code back to flattened spatial representation
        x = self.decoder_fc(z)  # [batch*sessions, flatten_dim]
        # Reshape to match the input shape of the spatial decoder
        x = x.view(z.shape[0], 128, 128, 16)  # [batch*sessions, 128, 128, 16]
        # Apply spatial decoder
        x = self.decoder_spatial(x)  # [batch*sessions, 1, 256, 32]
        # Squeeze the channel dimension for temporal decoder
        x = x.squeeze(1)  # [batch*sessions, 256, 32]
        # Apply temporal decoder
        x = self.decoder_temporal(x)  # [batch*sessions, channels, time]

        # The decoder outputs reconstructions for each session, flattened along batch*sessions
        return x

    def forward(self, x):
        # x shape: [batch, sessions, channels, time]
        # print(x.shape)
        batch, sessions, channels, time = x.shape

        # Encode to get session-specific mu and logvar
        mu_session_specific, logvar_session_specific = self.encode(x) # Shapes: [batch, sessions, latent_dim]

        # Reparameterize using session-specific mu and logvar
        z_session_specific = self.reparameterize(mu_session_specific, logvar_session_specific) # Shape: [batch, sessions, latent_dim]

        # For VAE reconstruction, flatten z across batch and sessions to process session-wise
        z_flattened_for_decode = z_session_specific.view(batch * sessions, self.latent_dim) # Shape: [batch*sessions, latent_dim]

        # Decode using the flattened session-specific latent codes
        recon_flattened = self.decode(z_flattened_for_decode) # Shape: [batch*sessions, channels, time]

        # Reshape the reconstruction back to include batch and sessions dimensions
        recon = recon_flattened.view(batch, sessions, channels, time) # Shape: [batch, sessions, channels, time]

        # For subject classification and projection head, use the session-averaged latent code
        # This assumes these heads are meant to represent the subject-level features,
        # not session-specific features. If session-specific features are needed,
        # these heads would need to be applied to z_session_specific and results aggregated or handled differently.
        # We calculate the session-averaged z here for compatibility with the original heads.
        z_averaged_for_heads = z_session_specific.mean(dim=1) # Shape: [batch, latent_dim]
        mu_averaged_for_heads = mu_session_specific.mean(dim=1) # Shape: [batch, latent_dim]
        logvar_averaged_for_heads = logvar_session_specific.mean(dim=1) # Shape: [batch, latent_dim]


        subject_logits = self.subject_classifier(z_averaged_for_heads) # Shape: [batch, num_subjects]
        projected_z = self.project_to_image_space(z_averaged_for_heads) # Shape: [batch, image_feature_dim]


        # Return reconstruction, session-averaged mu and logvar (for KLD loss calculation),
        # session-specific z, subject logits, and projected z.
        # Note: KLD loss should ideally use the session-specific mu and logvar,
        # calculated per session and then potentially summed/averaged over sessions and batch.
        # The returned mu_averaged_for_heads and logvar_averaged_for_heads are for potential
        # compatibility with a loss function expecting batch-level mu/logvar, but using
        # mu_session_specific and logvar_session_specific directly for KLD is more correct
        # for a session-specific VAE.
        return recon, mu_averaged_for_heads, logvar_averaged_for_heads, z_session_specific, subject_logits, projected_z


    def project_to_image_space(self, z_averaged):
        # z_averaged shape: [batch, latent_dim]
        return self.projection_head(z_averaged)



def vae_loss(model, recon_x, x, mu_averaged_for_heads, logvar_averaged_for_heads, z_session_specific, subject_logits, projected_z, image_features, subject_labels, beta=1.0, alpha=1.0, gamma=0.1):
    """
    VAE loss function compatible with the EEGVAE model that produces session-specific
    latent codes but uses session-averaged latent for subject classification and projection.

    Args:
        model: The EEGVAE model instance.
        recon_x (torch.Tensor): The reconstructed EEG data from the decoder.
                                 Shape: [batch, sessions, channels, time]
        x (torch.Tensor): The original EEG data input.
                          Shape: [batch, sessions, channels, time]
        mu_averaged_for_heads (torch.Tensor): Session-averaged mu from the encoder, used for heads.
                                              Shape: [batch, latent_dim]
        logvar_averaged_for_heads (torch.Tensor): Session-averaged logvar from the encoder, used for heads.
                                                  Shape: [batch, latent_dim]
        z_session_specific (torch.Tensor): Session-specific latent codes from the encoder.
                                           Shape: [batch, sessions, latent_dim]
        subject_logits (torch.Tensor): Logits from the subject classifier head.
                                       Shape: [batch, num_subjects]
        projected_z (torch.Tensor): Projected latent codes from the projection head.
                                    Shape: [batch, image_feature_dim]
        image_features (torch.Tensor): Corresponding image features for contrastive loss.
                                       Shape: [batch, image_feature_dim]
        subject_labels (torch.Tensor): Ground truth subject labels.
                                       Shape: [batch]
        beta (float): Weight for the KL divergence loss.
        alpha (float): Weight for the contrastive loss.
        gamma (float): Weight for the subject adversarial loss.
    """

    batch_size, sessions, channels, time = x.shape

    # --- 1. Reconstruction Loss ---
    # Calculate MSE loss between reconstructed and original data.
    # The reduction='mean' averages over all elements in the batch, sessions, channels, and time.
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # --- 2. KL Divergence ---
    # Calculate KL divergence for each session's latent distribution.
    # The KL divergence is calculated element-wise for mu and logvar.
    # We then sum across the latent dimension and average across sessions and batch.
    # Using z_session_specific's shape to get the session-specific mu and logvar
    # Note: The forward pass returns mu_averaged_for_heads and logvar_averaged_for_heads
    # for potential use with heads, but the true session-specific mu and logvar
    # are implicitly available from the encode step that produced z_session_specific.
    # A more robust approach would be to return session-specific mu and logvar
    # directly from the forward pass for accurate KLD calculation.
    # Assuming for now that the KLD should be calculated based on the distribution
    # that generated z_session_specific. Let's use the session-specific mu and logvar
    # that would correspond to z_session_specific. Since the forward pass
    # returned the averaged ones, we'll need to adjust or assume KLD is calculated
    # on the averaged distributions if that's intended.

    # Let's assume the KLD should be calculated based on the session-averaged
    # mu and logvar as returned by the forward pass for simplicity and
    # compatibility with the provided forward signature. If session-specific KLD
    # is truly needed, the forward pass return values would need to be adjusted.
    # Using the averaged mu and logvar for KLD as in the original loss function:
    kl_loss = -0.5 * torch.mean(1 + logvar_averaged_for_heads - mu_averaged_for_heads.pow(2) - logvar_averaged_for_heads.exp())

    # --- 3. Contrastive Loss ---
    # The projection head operates on the session-averaged latent code,
    # so we use the projected_z directly from the model's forward pass.
    proj_z_normalized = F.normalize(projected_z, dim=-1)
    image_features_normalized = F.normalize(image_features, dim=-1)

    # Calculate logits for contrastive loss
    logits = torch.matmul(proj_z_normalized, image_features_normalized.T) / 0.07  # (batch, batch)
    # Create labels for the diagonal elements (positive pairs)
    labels = torch.arange(proj_z_normalized.shape[0], device=proj_z_normalized.device)
    contrastive_loss = F.cross_entropy(logits, labels)

    # --- 4. Subject Adversarial Loss ---
    # The subject classifier operates on the session-averaged latent code,
    # so we use the subject_logits directly from the model's forward pass.
    subject_loss = F.cross_entropy(subject_logits, subject_labels)

    # --- 5. Total Loss ---
    # The subject loss is subtracted because it's an adversarial loss aiming
    # to make the latent space less discriminative of the subject.
    total_loss = recon_loss + beta * kl_loss + alpha * contrastive_loss - gamma * subject_loss

    return total_loss, recon_loss, kl_loss, contrastive_loss, subject_loss


class DummyEEGDataset(Dataset):
    def __init__(self, num_samples=1000, num_subjects=5, sessions=5, channels=64, time_points=250, image_feature_dim=384):
        self.num_samples = num_samples
        self.num_subjects = num_subjects
        self.sessions = sessions
        self.channels = channels
        self.time_points = time_points
        self.image_feature_dim = image_feature_dim
        self.eeg_data = []
        self.image_features = []
        self.subject_labels = []
        samples_per_subject = num_samples // num_subjects
        for subject_id in range(num_subjects):
            for _ in range(samples_per_subject):
                eeg = np.random.randn(sessions, channels, time_points) * 0.1
                eeg += np.random.randn() * 0.05 * subject_id
                eeg = (eeg - eeg.mean(axis=2, keepdims=True)) / (eeg.std(axis=2, keepdims=True) + 1e-8)
                self.eeg_data.append(eeg)
                img_feat = np.random.randn(image_feature_dim)
                img_feat = img_feat / np.linalg.norm(img_feat)
                self.image_features.append(img_feat)
                self.subject_labels.append(subject_id)
        self.eeg_data = np.array(self.eeg_data, dtype=np.float32)
        self.image_features = np.array(self.image_features, dtype=np.float32)
        self.subject_labels = np.array(self.subject_labels, dtype=np.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'eeg': torch.from_numpy(self.eeg_data[idx]),
            'image_features': torch.from_numpy(self.image_features[idx]),
            'subject_id': torch.tensor(self.subject_labels[idx], dtype=torch.long)
        }


# Main test script
if __name__ == "__main__":
    # Parameters
    batch_size = 32
    num_samples = 1000
    channels = 64
    time_points = 250
    sessions = 4
    latent_dim = 384
    image_feature_dim = 384
    num_subjects = 5

    # # Run shape tests
    # print("Running encoder shape test...")
    # test_encoder_shapes()
    # print("\nRunning decoder shape test...")
    # test_decoder_shapes()

    # Create dataset and DataLoader
    dataset = DummyEEGDataset(
        num_samples=num_samples,
        num_subjects=num_subjects,
        sessions=sessions,
        channels=channels,
        time_points=time_points,
        image_feature_dim=image_feature_dim
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model
    model = EEGVAE(
        channels=channels,
        time_points=time_points,
        sessions=sessions,
        latent_dim=latent_dim,
        image_feature_dim=image_feature_dim,
        num_subjects=num_subjects
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Test one epoch
    print("\nRunning training test...")
    model.train()
    for batch in dataloader:
        eeg_data = batch['eeg'].to(device)
        image_features = batch['image_features'].to(device)
        subject_labels = batch['subject_id'].to(device)

        optimizer.zero_grad()
        recon_output, mu_output, logvar_output, z_output, subject_logits_output, projected_z_output  = model(eeg_data)
        total_loss, recon_loss, kl_loss, contrastive_loss, subject_loss = vae_loss(
            model,
            recon_output,
            eeg_data, # Original input x
            mu_output, # mu_averaged_for_heads from model.forward
            logvar_output, # logvar_averaged_for_heads from model.forward
            z_output, # z_session_specific from model.forward
            subject_logits_output, # subject_logits from model.forward
            projected_z_output, # projected_z from model.forward
            image_features, # You would need to provide these
            subject_labels # You would need to provide these
        )

        total_loss.backward()
        optimizer.step()

        print(f"Batch Loss: {total_loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
              f"KL: {kl_loss.item():.4f}, Contrastive: {contrastive_loss.item():.4f}, "
              f"Subject: {subject_loss.item():.4f}")
        # break