import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class Lambda(nn.Module):
    """
    Input: A Function
    Returns : A Module that can be used
        inside nn.Sequential
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)


class SubjectDiscriminator(nn.Module):
    def __init__(self, features_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # self.disriminator_alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        # self.alpha = alpha

    def forward(self, x, alpha=1):
        x = GradientReversal.apply(x, alpha)
        return self.net(x)

class SubjectInvariantModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Shared stimulus encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # Subject-specific component
        self.subject_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Subject discriminator with gradient reversal
        self.discriminator = SubjectDiscriminator(features_dim=feature_dim)

    def forward(self, x):
        # Stimulus-invariant component
        Ed = self.encoder(x)
        
        # Subject-specific component
        B = self.subject_net(x)
        
        return Ed, B

class EEGDisentangler(nn.Module):
    def __init__(self, feature_dim, lr=3e-4):
        super().__init__()
        self.model = SubjectInvariantModel(feature_dim)
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.model.encoder.parameters()},
        #     {'params': self.model.subject_net.parameters()}
        # ], lr=lr)
        
        # self.disc_optimizer = torch.optim.Adam(
        #     self.model.discriminator.parameters(), lr=lr
        # )

        self.loss_fn = nn.MSELoss()

    def compute_loss(self, x1, x2, s1, s2):
        # Disentangled components
        Ed1, B1 = self.model(x1)
        Ed2, B2 = self.model(x2)
        
        # Reconstruction loss
        recon_loss = self.loss_fn(Ed1+B1, x1) + self.loss_fn(Ed2+B2, x2)
        
        # Stimulus consistency
        stimulus_loss = F.cosine_similarity(Ed1, Ed2, dim=-1).mean()
        
        # Adversarial subject confusion
        disc_Ed1 = self.model.discriminator(Ed1, alpha=1.0)
        disc_Ed2 = self.model.discriminator(Ed2, alpha=1.0)
        adv_loss = F.binary_cross_entropy_with_logits(
            torch.cat([disc_Ed1, disc_Ed2]), 
            torch.cat([s1.float(), s2.float()]).unsqueeze(1)
        )
        
        return recon_loss - stimulus_loss + adv_loss

    def train_step(self, batch):
        x1, s1, x2, s2 = batch
        
        # Train discriminator
        Ed1, B1 = self.model(x1)
        Ed2, B2 = self.model(x2)
        
        # self.disc_optimizer.zero_grad()
        disc_loss = F.binary_cross_entropy_with_logits(
            self.model.discriminator(torch.cat([Ed1, Ed2]), alpha=1.0),
            torch.cat([s1.float(), s2.float()]).unsqueeze(1)
        )
        # disc_loss.backward()
        # self.disc_optimizer.step()
        
        # Train main model
        # self.optimizer.zero_grad()
        main_loss = self.compute_loss(x1, x2, s1, s2)
        # main_loss.backward()
        # self.optimizer.step()
        
        return main_loss, disc_loss

# Usage example
if __name__ == "__main__":
    # Synthetic data parameters
    feature_dim = 128
    batch_size = 32
    
    # Initialize model
    disentangler = EEGDisentangler(feature_dim)
    
    # Training loop
    for epoch in range(100):
        # Generate synthetic batch (replace with real data loader)
        x1 = torch.randn(batch_size, feature_dim)  # Subject 1
        s1 = torch.zeros(batch_size)  # Subject labels
        x2 = torch.randn(batch_size, feature_dim)  # Subject 2
        s2 = torch.ones(batch_size)
        
        # Training step
        main_loss, disc_loss = disentangler.train_step((x1, s1, x2, s2))
        
        print(f"Epoch {epoch+1}: "
              f"Main Loss: {main_loss:.4f} | "
              f"Disc Loss: {disc_loss:.4f}")

    # Inference example
    with torch.no_grad():
        test_input = torch.randn(1, feature_dim)
        Ed, B = disentangler.model(test_input)
        print("\nDisentangled components:")
        print(f"Stimulus features shape: {Ed.shape}")
        print(f"Subject variance shape: {B.shape}")



# class SubjectDiscriminator(nn.Module):
#     def __init__(self, input_features=32):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_features, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )
    
#     def forward(self, x):
#         return self.net(x)

# class SubjectInvariantModel(nn.Module):
#     def __init__(self, feature_dim, latent_dim=768):
#         super().__init__()
#         # Shared stimulus encoder
#         self.stimulus_encoder = nn.Sequential(
#             nn.Linear(feature_dim, latent_dim),
#             nn.ReLU(),
#             nn.Linear(latent_dim, latent_dim)
#         )
        
#         # Subject variance estimator
#         self.subject_adapter = nn.Sequential(
#             nn.Linear(feature_dim, latent_dim),
#             nn.ReLU(),
#             nn.Linear(latent_dim, feature_dim)
#         )
        
#     def forward(self, x):
#         Ed = self.stimulus_encoder(x)
#         B = self.subject_adapter(x)
#         return Ed, B

# class ContrastiveEEGLearner(nn.Module):
#     def __init__(self, feature_dim):
#         super().__init__()
#         self.model = SubjectInvariantModel(feature_dim, latent_dim=feature_dim)
#         self.discriminator = SubjectDiscriminator(input_features=32)
        
#         # Projection head for contrastive loss
#         self.projection = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim),
#             nn.ReLU(),
#             nn.Linear(feature_dim, 32)
#         )

#     def forward(self, x): 
#         # Get decompositions
#         Ed, B = self.model(x)
#         z = self.projection(Ed)
#         return Ed, B, z

#     def compute_loss(self, batch):
#         # Batch contains pairs of same-class samples from different subjects
#         x1, x2 = batch  # (batch_size, feature_dim) pairs
        
#         # Get decompositions
#         Ed1, B1 = self.model(x1)
#         Ed2, B2 = self.model(x2)
        
#         # Reconstruction loss
#         recon_loss = F.mse_loss(Ed1 + B1, x1) + F.mse_loss(Ed2 + B2, x2)
        
#         # Stimulus consistency loss (same-class Ed's should match)
#         stimulus_loss = F.mse_loss(Ed1, Ed2)
        
#         # Subject contrastive loss
#         z1 = self.projection(Ed1)
#         z2 = self.projection(Ed2)
#         contrastive_loss = F.cosine_embedding_loss(
#             z1, z2, 
#             torch.ones(z1.size(0)).to(x1.device)
#         )

#         subject_preds = self.discriminator(z1)
#         adv_loss = F.binary_cross_entropy_with_logits(
#             subject_preds, 
#             torch.zeros_like(subject_preds)  # Try to fool discriminator
#         )
        
#         # Subject variance regularization
#         subject_reg = 0.01 * (B1.norm(p=2) + B2.norm(p=2))
        
#         return recon_loss + stimulus_loss + contrastive_loss + subject_reg