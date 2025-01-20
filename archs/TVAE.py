import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset

# Define the Transformer Variational Autoencoder (TVAE)
class TVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads=7, num_layers=2):
        super(TVAE, self).__init__()
        
        # Encoder part
        self.encoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        
        # Latent space
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
        # Decoder part
        self.fc_latent_to_input = nn.Linear(latent_dim, input_dim)  # Project latent to input dimension for decoder
        self.decoder_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        
        # Final output layer
        self.fc_out = nn.Linear(input_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, encoder_only=False):
        # Encoder
        x = x.transpose(0, 1)  # Transform to shape (seq_len, batch_size, input_dim) for transformer
        encoder_output = self.encoder_transformer(x)
        
        # Extract the last encoder output for latent space representation
        enc_output_last = encoder_output[-1, :, :]

        if encoder_only:
            return enc_output_last
        
        # Latent space
        mu = self.fc_mu(enc_output_last)
        logvar = self.fc_logvar(enc_output_last)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Project latent vector z to the input_dim for the decoder
        z_proj = self.fc_latent_to_input(z).unsqueeze(0).repeat(x.size(0), 1, 1)  # Repeat across time dimension
        
        # Decoder - use the projected latent vector as input
        decoder_output = self.decoder_transformer(z_proj, encoder_output)
        
        # Output layer
        output = self.fc_out(decoder_output)
        
        return output.transpose(0, 1), mu, logvar  # Transpose back to (batch_size, seq_len, input_dim)
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (Mean Squared Error)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss


if __name__=="__main__":
    # Create the model
    input_channels = 63
    seq_length = 256
    model = TVAE(input_channels=input_channels, seq_length=seq_length)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop for VAE
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mean, logvar = model(batch)
            
            # Compute the VAE loss
            loss = model.loss_function(recon_batch, batch, mean, logvar)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")