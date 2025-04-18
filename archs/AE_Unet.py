import torch
import torch.nn as nn
import torch.nn.functional as F

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, T = x.size()
        query = self.query_conv(x).view(batch_size, -1, T)
        key = self.key_conv(x).view(batch_size, -1, T)
        value = self.value_conv(x).view(batch_size, -1, T)
        
        energy = torch.bmm(query.transpose(1, 2), key)
        attention = self.softmax(energy)
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(batch_size, C, T)
        
        return self.gamma * out + x

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, dropout_rate=0.1):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class SpatioTemporalEEGAutoencoder(nn.Module):
    def __init__(self, in_channels=63, latent_dim=32, init_features=32, dropout_rate=0.1):
        super(SpatioTemporalEEGAutoencoder, self).__init__()
        
        features = init_features
        
        # Encoder
        self.enc1 = TemporalConvBlock(in_channels, features, dilation=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.att1 = SpatialAttention(features)
        
        self.enc2 = TemporalConvBlock(features, features * 2, dilation=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.att2 = SpatialAttention(features * 2)
        
        self.enc3 = TemporalConvBlock(features * 2, features * 4, dilation=4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.att3 = SpatialAttention(features * 4)
        
        self.enc4 = TemporalConvBlock(features * 4, features * 8, dilation=8)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.att4 = SpatialAttention(features * 8)
        
        # Bottleneck with latent space
        self.bottleneck = TemporalConvBlock(features * 8, features * 16, dilation=8)
        self.latent_in_features = features * 16 * 15  # 512 * 15 = 7680
        self.fc_to_latent = nn.Linear(self.latent_in_features, latent_dim)
        self.fc_bn = nn.BatchNorm1d(latent_dim)
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc_from_latent = nn.Linear(latent_dim, self.latent_in_features)
        self.unflatten = nn.Unflatten(1, (features * 16, 15))
        
        # Decoder
        self.up4 = nn.ConvTranspose1d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec4 = TemporalConvBlock(features * 16, features * 8, dilation=8)  # 256+256=512 -> 256
        self.att_dec4 = SpatialAttention(features * 8)
        
        self.up3 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = TemporalConvBlock(features * 8, features * 4, dilation=4)   # 128+128=256 -> 128
        self.att_dec3 = SpatialAttention(features * 4)
        
        self.up2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = TemporalConvBlock(features * 4, features * 2, dilation=2)   # 64+64=128 -> 64
        self.att_dec2 = SpatialAttention(features * 2)
        
        self.up1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = TemporalConvBlock(features * 2, features, dilation=1)       # 32+32=64 -> 32
        self.att_dec1 = SpatialAttention(features)
        
        self.final_conv = nn.Conv1d(features, in_channels, kernel_size=1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def encode(self, x):
        # Encoder
        enc1 = self.enc1(x)        # (batch, 32, 250)
        enc1_att = self.att1(enc1)
        pool1 = self.pool1(enc1_att)  # (batch, 32, 125)
        
        enc2 = self.enc2(pool1)    # (batch, 64, 125)
        enc2_att = self.att2(enc2)
        pool2 = self.pool2(enc2_att)  # (batch, 64, 62)
        
        enc3 = self.enc3(pool2)    # (batch, 128, 62)
        enc3_att = self.att3(enc3)
        pool3 = self.pool3(enc3_att)  # (batch, 128, 31)
        
        enc4 = self.enc4(pool3)    # (batch, 256, 31)
        enc4_att = self.att4(enc4)
        pool4 = self.pool4(enc4_att)  # (batch, 256, 15)
        
        # Bottleneck to latent
        bottleneck = self.bottleneck(pool4)  # (batch, 512, 15)
        flat = bottleneck.view(bottleneck.size(0), -1)  # (batch, 7680)
        latent = self.fc_dropout(F.relu(self.fc_bn(self.fc_to_latent(flat))))  # (batch, latent_dim)
        
        return latent, (enc1_att, enc2_att, enc3_att, enc4_att)
    
    def decode(self, latent, enc_features):
        enc1_att, enc2_att, enc3_att, enc4_att = enc_features
        
        # Latent to bottleneck
        x = self.fc_from_latent(latent)  # (batch, 7680)
        x = self.unflatten(x)           # (batch, 512, 15)
        
        # Decoder with skip connections
        up4 = self.up4(x)              # (batch, 256, 30)
        up4 = torch.cat((up4, enc4_att[:, :, :30]), dim=1)  # (batch, 512, 30)
        dec4 = self.dec4(up4)          # (batch, 256, 30)
        dec4_att = self.att_dec4(dec4)
        
        up3 = self.up3(dec4_att)       # (batch, 128, 60)
        up3 = torch.cat((up3, enc3_att[:, :, :60]), dim=1)  # (batch, 256, 60)
        dec3 = self.dec3(up3)          # (batch, 128, 60)
        dec3_att = self.att_dec3(dec3)
        
        up2 = self.up2(dec3_att)       # (batch, 64, 120)
        up2 = torch.cat((up2, enc2_att[:, :, :120]), dim=1)  # (batch, 128, 120)
        dec2 = self.dec2(up2)          # (batch, 64, 120)
        dec2_att = self.att_dec2(dec2)
        
        up1 = self.up1(dec2_att)       # (batch, 32, 240)
        up1 = torch.cat((up1, enc1_att[:, :, :240]), dim=1)  # (batch, 64, 240)
        dec1 = self.dec1(up1)          # (batch, 32, 240)
        dec1_att = self.att_dec1(dec1)
        
        # Pad to match input size
        diff = 250 - dec1_att.size(2)
        if diff > 0:
            dec1_att = F.pad(dec1_att, (diff//2, diff - diff//2))  # (batch, 32, 250)
        
        output = self.final_conv(dec1_att)  # (batch, 63, 250)
        return output
    
    def forward(self, x, encoder_only=False):
        latent, enc_features = self.encode(x)
        if encoder_only:
            return [], latent
        output = self.decode(latent, enc_features)
        return output, latent

# Training code (same as before, included for completeness)
def train_model():
    batch_size = 32
    learning_rate = 0.001
    epochs = 50
    latent_dim = 32
    dropout_rate = 0.3
    
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