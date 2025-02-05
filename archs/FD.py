import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn

# Define the model
class FeatureDecomposerV2(nn.Module):
    def __init__(self, feature_dim, common_dim, use_activation=True, activation_fn=nn.ReLU, use_normalization=False):
        """
        Enhanced Feature Decomposer Model with reconstruction.
        Args:
            feature_dim (int): Dimension of input features.
            common_dim (int): Dimension of common and variance output features.
            use_activation (bool): Whether to apply an activation function.
            activation_fn (nn.Module): Activation function to use.
            use_normalization (bool): Whether to use normalization layers.
        """
        super(FeatureDecomposerV2, self).__init__()
        
        self.use_activation = use_activation
        self.activation_fn = activation_fn
        self.use_normalization = use_normalization

        # Encoder for common features
        self.encoder_common = nn.Sequential(
            nn.Linear(feature_dim, common_dim * 2),
            activation_fn() if use_activation else nn.Identity(),
            nn.Linear(common_dim * 2, common_dim),
            nn.LayerNorm(common_dim) if use_normalization else nn.Identity(),
        )

        # Encoder for variance features
        self.encoder_variance = nn.Sequential(
            nn.Linear(feature_dim, common_dim * 2),
            activation_fn() if use_activation else nn.Identity(),
            nn.Linear(common_dim * 2, common_dim),
            nn.LayerNorm(common_dim) if use_normalization else nn.Identity(),
        )

        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim * 3),
            activation_fn() if use_activation else nn.Identity(),
            nn.Linear(common_dim * 3, feature_dim),
        )

    def forward(self, x, subject_variance=None, E_c_subject=None):
        """
        Forward pass
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            Common features, variance features, and reconstructed input.
        """
        # Compute common features
        if E_c_subject is not None:
            E_c = E_c_subject
        else:
            E_c = self.encoder_common(x)
            
        # Compute variance features
        if subject_variance is not None:
            V = subject_variance
        else:
            V = self.encoder_variance(x)

        combined = torch.cat((E_c,V),dim=-1)

        # Reconstruct input from E_c + V
        reconstructed = self.decoder(combined)
        return E_c, V, reconstructed

# Define the model
class FeatureDecomposer(nn.Module):
    def __init__(self, feature_dim, common_dim):
        super(FeatureDecomposer, self).__init__()
        self.encoder_common = nn.Linear(feature_dim, common_dim)  # Shared features
        self.encoder_variance = nn.Linear(feature_dim, common_dim)  # Variance features

    def forward(self, x):
        E_c = self.encoder_common(x)  # Common features
        V = self.encoder_variance(x)  # Variance
        return E_c, V

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, E_c_j, E_c_k):
        # Minimize distance between common features
        return torch.mean((E_c_j - E_c_k).pow(2))

# Orthogonality Loss
def orthogonality_loss(E_c, V):
    # Encourage E_c and V to be orthogonal
    dot_product = torch.sum(E_c * V, dim=1)  # Batch-wise dot product
    return torch.mean(dot_product.pow(2))

# Reconstruction Loss
def reconstruction_loss(E_c, V, original):
    reconstructed = E_c + V
    return torch.mean((reconstructed - original).pow(2))

# Training loop
def train_model(feature_dim, common_dim, data_loader, epochs=50, lr=1e-3):
    model = FeatureDecomposer(feature_dim, common_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    contrastive_criterion = ContrastiveLoss()
    lambda_ortho = 0.1  # Weight for orthogonality loss
    lambda_recon = 0.1  # Weight for reconstruction loss

    for epoch in range(epochs):
        total_loss = 0
        for E_j, E_k in data_loader:
            # Forward pass
            E_c_j, V_j = model(E_j)
            E_c_k, V_k = model(E_k)

            # Losses
            contrastive_loss = contrastive_criterion(E_c_j, E_c_k)
            ortho_loss_j = orthogonality_loss(E_c_j, V_j)
            ortho_loss_k = orthogonality_loss(E_c_k, V_k)
            recon_loss_j = reconstruction_loss(E_c_j, V_j, E_j)
            recon_loss_k = reconstruction_loss(E_c_k, V_k, E_k)

            # Total loss
            loss = contrastive_loss + lambda_ortho * (ortho_loss_j + ortho_loss_k) + lambda_recon * (recon_loss_j + recon_loss_k)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

    return model

# Example usage with dummy data
if __name__ == "__main__":
    # Dummy dataset
    feature_dim = 16
    common_dim = 8
    batch_size = 32

    # Generate random data for E_j and E_k
    data = [(torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)) for _ in range(100)]
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

    # Train the model
    model = train_model(feature_dim, common_dim, data_loader)
