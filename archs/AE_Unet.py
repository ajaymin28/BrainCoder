import torch
import torch.nn as nn
import torch.nn.functional as F


# EEG Augmenter for Visual Stimuli (No specific occipital indices required upfront)
import torch

class EEGDataAugmentation:
    def __init__(self, probability=0.5, noise_level=0.01, time_shift_max=10, 
                 amplitude_scale_range=(0.9, 1.1), dropout_prob=0.1):
        """
        Initialize EEG data augmentation parameters.
        
        Args:
            probability (float): Probability of applying each augmentation
            noise_level (float): Standard deviation of Gaussian noise
            time_shift_max (int): Maximum number of samples for time shifting
            amplitude_scale_range (tuple): Range for scaling amplitude
            dropout_prob (float): Probability of channel dropout
        """
        self.probability = probability
        self.noise_level = noise_level
        self.time_shift_max = time_shift_max
        self.amplitude_scale_range = amplitude_scale_range
        self.dropout_prob = dropout_prob

    def random_apply(self, x, augmentation_func):
        """Apply augmentation randomly based on probability"""
        if torch.rand(1) < self.probability:
            return augmentation_func(x)
        return x

    def add_gaussian_noise(self, x):
        """Add Gaussian noise to simulate neural noise in visual processing"""
        noise = torch.normal(mean=0.0, std=self.noise_level, size=x.shape)
        return x + noise

    def time_shift(self, x):
        """Shift time samples to simulate variations in visual response latency"""
        shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
        if shift > 0:
            return torch.roll(x, shifts=shift, dims=2)
        elif shift < 0:
            # Pad with zeros when shifting left
            padding = torch.zeros_like(x[:, :, :abs(shift)])
            shifted = torch.roll(x, shifts=shift, dims=2)
            shifted[:, :, -abs(shift):] = 0
            return shifted
        return x

    def amplitude_scaling(self, x):
        """Scale amplitude to simulate varying signal strength"""
        scale = torch.empty(x.shape[0]).uniform_(*self.amplitude_scale_range)
        scale = scale.view(-1, 1, 1)  # Reshape for broadcasting
        return x * scale

    def channel_dropout(self, x):
        """Randomly drop channels to simulate electrode noise"""
        mask = torch.ones_like(x)
        dropout_mask = torch.bernoulli(torch.full((x.shape[0], x.shape[1], 1), 
                                                1 - self.dropout_prob))
        mask = mask * dropout_mask
        return x * mask

    def frequency_jitter(self, x):
        """Add slight frequency perturbation for visual stimuli variation"""
        fft = torch.fft.rfft(x, dim=2)
        noise = torch.normal(0, self.noise_level/2, fft.shape)
        fft_perturbed = fft + noise
        return torch.fft.irfft(fft_perturbed, n=x.shape[2], dim=2)

    def augment(self, x):
        """
        Apply a combination of augmentations to EEG data.
        
        Args:
            x (torch.Tensor): Input EEG data of shape (batch, 63, 250)
            
        Returns:
            torch.Tensor: Augmented EEG data
        """
        # Ensure input is float tensor
        x = x.float()
        
        # Define augmentations
        augmentations = [
            self.add_gaussian_noise,
            self.time_shift,
            self.amplitude_scaling,
            self.channel_dropout,
            self.frequency_jitter
        ]
        
        # Randomly shuffle augmentation order using torch.randperm
        indices = torch.randperm(len(augmentations))
        shuffled_augmentations = [augmentations[i] for i in indices]
        
        # Apply each augmentation with probability
        augmented_x = x.clone()
        for aug in shuffled_augmentations:
            augmented_x = self.random_apply(augmented_x, aug)
            
        return augmented_x

        

# Spatial Attention Module to model electrode relationships
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch_size, channels, time)
        batch_size, C, T = x.size()
        
        # Project to query, key, value
        query = self.query_conv(x).view(batch_size, -1, T)  # (B, C//8, T)
        key = self.key_conv(x).view(batch_size, -1, T)      # (B, C//8, T)
        value = self.value_conv(x).view(batch_size, -1, T)  # (B, C, T)
        
        # Attention: (B, C//8, T) @ (B, T, C//8) -> (B, C//8, C//8)
        energy = torch.bmm(query.transpose(1, 2), key)      # (B, T, T)
        attention = self.softmax(energy)                    # (B, T, T)
        
        # Apply attention: (B, T, T) @ (B, T, C) -> (B, T, C)
        out = torch.bmm(value, attention.transpose(1, 2))   # (B, C, T)
        out = out.view(batch_size, C, T)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out

# Enhanced Residual Block with Dilated Convolution for temporal context
class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, dropout_rate=0.1):
        super(TemporalResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.adjust_channels = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        if self.adjust_channels is not None:
            residual = self.adjust_channels(residual)
            
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out = out + residual
        out = F.relu(out)
        return out

class SpatioTemporalEEGAutoencoder(nn.Module):
    def __init__(self, in_channels=63, latent_dim=32, init_features=32, dropout_rate=0.1):
        super(SpatioTemporalEEGAutoencoder, self).__init__()
        
        features = init_features
        
        # Spatial Attention at input
        self.spatial_att_in = SpatialAttention(in_channels)
        
        # Encoder with spatial and temporal modeling
        self.enc1 = TemporalResidualBlock(in_channels, features, dilation=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.spatial_att1 = SpatialAttention(features)
        
        self.enc2 = TemporalResidualBlock(features, features * 2, dilation=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.spatial_att2 = SpatialAttention(features * 2)
        
        self.enc3 = TemporalResidualBlock(features * 2, features * 4, dilation=4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.spatial_att3 = SpatialAttention(features * 4)
        
        self.enc4 = TemporalResidualBlock(features * 4, features * 8, dilation=8)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Latent space
        self.flatten = nn.Flatten()
        self.fc_to_latent = nn.Linear(features * 8 * 15, latent_dim)
        self.fc_bn = nn.BatchNorm1d(latent_dim)
        self.fc_dropout = nn.Dropout(dropout_rate)
        
        # Decoder
        self.fc_from_latent = nn.Linear(latent_dim, features * 8 * 15)
        self.unflatten = nn.Unflatten(1, (features * 8, 15))
        
        self.up4 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec4 = TemporalResidualBlock(features * 4, features * 4, dilation=8)
        self.spatial_att_dec4 = SpatialAttention(features * 4)
        
        self.up3 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec3 = TemporalResidualBlock(features * 2, features * 2, dilation=4)
        self.spatial_att_dec3 = SpatialAttention(features * 2)
        
        self.up2 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.dec2 = TemporalResidualBlock(features, features, dilation=2)
        self.spatial_att_dec2 = SpatialAttention(features)
        
        self.up1 = nn.ConvTranspose1d(features, features, kernel_size=2, stride=2)
        self.dec1 = TemporalResidualBlock(features, features, dilation=1)
        
        self.conv_final = nn.Conv1d(features, in_channels, kernel_size=1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def encode(self, x):
        x = self.spatial_att_in(x)
        
        enc1 = self.enc1(x)
        enc1 = self.spatial_att1(enc1)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        enc2 = self.spatial_att2(enc2)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        enc3 = self.spatial_att3(enc3)
        pool3 = self.pool3(enc3)
        
        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        
        flat = self.flatten(pool4)
        latent = self.fc_dropout(F.relu(self.fc_bn(self.fc_to_latent(flat))))
        return latent
    
    def decode(self, latent):
        x = self.fc_from_latent(latent)
        x = self.unflatten(x)
        
        up4 = self.up4(x)
        dec4 = self.dec4(up4)
        dec4 = self.spatial_att_dec4(dec4)
        
        up3 = self.up3(dec4)
        dec3 = self.dec3(up3)
        dec3 = self.spatial_att_dec3(dec3)
        
        up2 = self.up2(dec3)
        dec2 = self.dec2(up2)
        dec2 = self.spatial_att_dec2(dec2)
        
        up1 = self.up1(dec2)
        dec1 = self.dec1(up1)
        
        diff = 250 - dec1.size(2)
        if diff > 0:
            dec1 = F.pad(dec1, (diff//2, diff - diff//2))
            
        output = self.conv_final(dec1)
        return output
    
    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)
        return output, latent

# Training code (same as before, included for completeness)
def train_model():
    batch_size = 32
    learning_rate = 0.001
    epochs = 50
    latent_dim = 32
    dropout_rate = 0.1
    
    model = SpatioTemporalEEGAutoencoder(latent_dim=latent_dim, dropout_rate=dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    X = torch.randn(batch_size, 63, 250).to(device)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        outputs, latent = model(X)
        loss = criterion(outputs, X)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        scheduler.step(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print(f'Latent shape: {latent.shape}')
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    model = SpatioTemporalEEGAutoencoder(latent_dim=32)
    x = torch.randn(1, 63, 250)
    output, latent = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Output shape: {output.shape}")
    # train_model()