import torch
import torch.nn as nn
import torch.nn.functional as F


# Gradient Reversal Layer (GRL)
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)

# # Import the model
# class CNNTrans(nn.Module):
#     def __init__(self, input_channels=63, time_samples=250, embed_dim=512):
#         super(CNNTrans, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
#         self.bn = nn.BatchNorm2d(128)
        
#         # Transformer processes spatial (channel) relationships
#         num_heads = 8  # 128 is divisible by 8
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, dim_feedforward=512, dropout=0.1),
#             num_layers=4
#         )
        
#         # Attention pooling instead of mean pooling
#         self.attn_pool = nn.MultiheadAttention(embed_dim=128, num_heads=4)
#         self.fc = nn.Linear(128, embed_dim)

#         self.embed_dim = embed_dim

#     def forward(self, x):
#         # Input shape: (batch, 63, 250)
#         x = x.unsqueeze(1)  # (batch, 1, 63, 250)
#         x = F.relu(self.conv1(x))  # (batch, 32, 63, 125)
#         x = F.relu(self.conv2(x))  # (batch, 64, 63, 63)
#         x = F.relu(self.conv3(x))  # (batch, 128, 63, 32)
#         x = self.bn(x)  # (batch, 128, 63, 32)
        
#         # Reshape to (batch, 32, 128, 63) and treat channels as sequence length
#         x = x.permute(3, 0, 2, 1)  # (32, batch, 63, 128)
#         x = x.reshape(32, -1, 128)  # (32, batch*63, 128)
        
#         # Transformer processes (seq_len=63, batch*32, embedding_dim=128)
#         x = self.transformer(x)  # Output: (32, batch*63, 128)
        
#         # Attention pooling to extract relevant features
#         x = x.mean(dim=0).unsqueeze(0)  # (1, batch*63, 128)
#         x, _ = self.attn_pool(x, x, x)  # (1, batch*63, 128)
#         x = x.squeeze(0)  # (batch*63, 128)
        
#         # Project to embedding space
#         x = self.fc(x)  # (batch*63, 512)
#         x = x.view(-1, 63, self.embed_dim).mean(dim=1)  # Aggregate spatial features (batch, 512)
        
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

# class CNNTrans(nn.Module):
#     """A CNN-Transformer hybrid for spatio-temporal EEG data embedding using 1D convolutions.
    
#     Args:
#         input_channels (int): Number of input channels (default: 63).
#         time_samples (int): Number of time samples (default: 250).
#         embed_dim (int): Output embedding dimension (default: 512).
    
#     Input:
#         x (Tensor): Shape (batch, input_channels, time_samples)
#     Output:
#         Tensor: Shape (batch, embed_dim)
#     """
#     def __init__(self, input_channels=63, time_samples=250, embed_dim=512):
#         super(CNNTrans, self).__init__()
#         # 1D convolutions: kernel_size=5, stride=2 to reduce time dimension
#         self.conv1 = nn.Conv1d(input_channels, 63, kernel_size=3, stride=2, padding=2)
#         self.conv2 = nn.Conv1d(63, 60, kernel_size=3, stride=2, padding=2)
#         self.conv3 = nn.Conv1d(50, 40, kernel_size=3, stride=2, padding=2)
#         self.bn = nn.BatchNorm1d(40)
#         self.dropout = nn.Dropout(0.2)
#         self.input_channels = input_channels
#         self.time_samples = time_samples
        
#         # Transformer: d_model = 512 (from conv3 output channels)
#         num_heads = 8
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=40, nhead=num_heads, dim_feedforward=embed_dim, dropout=0.1),
#             num_layers=4
#         )
        
#         # Fully connected layer to produce final embedding
#         self.fc = nn.Linear(1280, embed_dim)
#         self.embed_dim = embed_dim

#     def forward(self, x):
#         assert x.shape[1] == self.input_channels and x.shape[2] == self.time_samples, f"Expected (batch, 63, 250), got {x.shape}"
        
#         # 1D convolutions
#         x = F.relu(self.conv1(x))  # (batch, 63, 125)
#         x = F.relu(self.conv2(x))  # (batch, 30, 63)
#         x = F.relu(self.conv3(x))  # (batch, 40, 32)
#         x = self.bn(x)
#         x = self.dropout(x)

#         x = x.view(x.size(0), -1)  # (batch, 1280)
        
        
#         # # Reshape for transformer
#         # # seq_len = x.shape[2]  # 32
#         # x = x.permute(2, 0, 1)  # (32, batch, 512)
        
#         # # Transformer
#         # x = self.transformer(x)  # (32, batch, 512)
#         # x = x.transpose(0, 1)  # (batch, 32, 512)
        
#         # # Mean pooling across sequence length
#         # x = x.mean(dim=1)  # (batch, 512)

        
#         # Final embedding
#         x = self.fc(x)  # (batch, 512)
#         return x

# class CNNTrans(nn.Module):
#     """A CNN-Transformer hybrid for spatio-temporal EEG data embedding using 1D convolutions.
    
#     Args:
#         input_channels (int): Number of input channels (default: 63).
#         time_samples (int): Number of time samples (default: 250).
#         embed_dim (int): Output embedding dimension (default: 512).
    
#     Input:
#         x (Tensor): Shape (batch, input_channels, time_samples)
#     Output:
#         Tensor: Shape (batch, embed_dim)
#     """
#     def __init__(self, input_channels=63, time_samples=250, embed_dim=512):
#         super(CNNTrans, self).__init__()
#         # 1D convolutions: kernel_size=3, stride=2 to reduce time dimension
#         self.conv1 = nn.Conv1d(input_channels, 63, kernel_size=3, stride=2, padding=2)
#         self.conv2 = nn.Conv1d(63, 50, kernel_size=3, stride=2, padding=2)
#         self.conv3 = nn.Conv1d(50, 40, kernel_size=3, stride=2, padding=2)
#         self.bn = nn.BatchNorm1d(40)
#         self.dropout = nn.Dropout(0.2)
#         self.input_channels = input_channels
#         self.time_samples = time_samples
        
#         # Fully connected layer to produce final embedding
#         self.fc = nn.Linear(1320, embed_dim)
#         self.embed_dim = embed_dim

#     def forward(self, x):
#         assert x.shape[1] == self.input_channels and x.shape[2] == self.time_samples, f"Expected (batch, 63, 250), got {x.shape}"
        
#         # 1D convolutions
#         x = F.relu(self.conv1(x))  # Shape: (B, 63, 126)
#         x = F.relu(self.conv2(x))  # Shape: (B, 60, 64)
#         x = F.relu(self.conv3(x))  # Shape: (B, 40, 33)
#         x = self.bn(x)
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)

#         # Final embedding
#         x = self.fc(x)
#         return x


class SubjectDiscriminator(nn.Module):
    def __init__(self,embed_dim=768, num_subjects=1, alpha=1.0):
        super(SubjectDiscriminator, self).__init__()

        self.alpha = alpha
        self.num_subjects = num_subjects


        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),  # Another hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),  # Another hidden layer
            nn.ReLU(),
            nn.Linear(512, embed_dim),  # Another hidden layer
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(embed_dim, num_subjects)  # Predict subject

        # # Subject Invariance Discriminator
        # self.discriminator = nn.Sequential(
        #     nn.Linear(embed_dim, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, num_subjects),
        #     # nn.Sigmoid(),
        # )
    
    def forward(self, x, alpha=None):
        if alpha is None: 
            alpha = self.alpha

        reversed_x = grad_reverse(x, alpha)  # Apply GRL
        subject_features = self.feature_extractor(reversed_x)  # Extract subject features
        subject_pred = self.classifier(subject_features)  # Predict subject
        return subject_pred, subject_features  # Return both
        
        # reversed_embedding = grad_reverse(x, alpha)
        # subject_pred = self.discriminator(reversed_embedding)
        # return subject_pred, reversed_embedding


class CNNTrans(nn.Module):
    """
    V0: 31/3/2025
    wandb:    top-1 0.09
    wandb:    top-3 0.12
    wandb:    top-5 0.135

    Args:
        input_channels (int): Number of input channels (default: 63).
        time_samples (int): Number of time samples (default: 250).
        embed_dim (int): Output embedding dimension (default: 768).
    
    Input:
        x (Tensor): Shape (batch, input_channels, time_samples)
    Output:
        Tensor: Shape (batch, embed_dim)
    """
    def __init__(self, input_channels=63, time_samples=250, embed_dim=768):
        super(CNNTrans, self).__init__()
        
        # Convolutional layers with pooling
        self.conv1 = nn.Conv1d(input_channels, 63, kernel_size=3, stride=2, padding=2)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(63, 50, kernel_size=3, stride=2, padding=2)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(50, 40, kernel_size=3, stride=2, padding=2)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        
        # Skip connection layers (1x1 conv to match dimensions)
        self.skip1 = nn.Conv1d(input_channels, 63, kernel_size=1, stride=2, padding=0)
        self.skip2 = nn.Conv1d(63, 50, kernel_size=1, stride=2, padding=0)
        self.skip3 = nn.Conv1d(50, 40, kernel_size=1, stride=2, padding=0)
        
        # Batch norm and dropout
        self.bn = nn.BatchNorm1d(40)
        self.dropout = nn.Dropout(0.2)
        
        # Store input dimensions
        self.input_channels = input_channels
        self.time_samples = time_samples
        
        # Fully connected layer (adjusted for correct input size)
        self.fc = nn.Linear(40 * 33, embed_dim)  # 40 channels * 33 time steps
        self.embed_dim = embed_dim


    def forward(self, x, lambda_adv=0.1):
        assert x.shape[1] == self.input_channels and x.shape[2] == self.time_samples, f"Expected (batch, 63, 250), got {x.shape}"
        
        # Block 1: conv1 + pool1 + residual
        identity = self.skip1(x)  # (B, 63, 125)
        x = F.relu(self.conv1(x))  # (B, 63, 126)
        x = self.pool1(x)  # (B, 63, 126)
        # Adjust identity to match x's time dimension (125 -> 126)
        identity = F.pad(identity, (0, 1), "constant", 0)  # (B, 63, 126)
        x = x + identity  # (B, 63, 126)
        
        # Block 2: conv2 + pool2 + residual
        identity = self.skip2(x)  # (B, 60, 63)
        x = F.relu(self.conv2(x))  # (B, 60, 64)
        x = self.pool2(x)  # (B, 60, 64)
        # Adjust identity to match x's time dimension (63 -> 64)
        identity = F.pad(identity, (0, 1), "constant", 0)  # (B, 60, 64)
        x = x + identity  # (B, 60, 64)
        
        # Block 3: conv3 + pool3 + residual
        identity = self.skip3(x)  # (B, 40, 32)
        x = F.relu(self.conv3(x))  # (B, 40, 33)
        x = self.pool3(x)  # (B, 40, 33)
        # Adjust identity to match x's time dimension (32 -> 33)
        identity = F.pad(identity, (0, 1), "constant", 0)  # (B, 40, 33)
        x = x + identity  # (B, 40, 33)
        
        # Batch norm and dropout
        x = self.bn(x)  # (B, 40, 33)
        x = self.dropout(x)  # (B, 40, 33)
        
        # Flatten and final embedding
        x = x.view(x.size(0), -1)  # (B, 1320)
        x = self.fc(x)  # (B, 512)

        return x



# class CNNTrans(nn.Module):
#     """
#     V1: 31/3/2025

#     # top-1 0.085
#     # top-3 0.18
#     # top-5 0.205

#     """


#     def __init__(self, input_channels=63, time_samples=250, embed_dim=768):
#         super(CNNTrans, self).__init__()
        
#         # Depthwise separable conv layers
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, groups=input_channels),
#             nn.Conv1d(input_channels, 63, kernel_size=1)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(63, 63, kernel_size=3, stride=1, padding=1, groups=63),
#             nn.Conv1d(63, 50, kernel_size=1)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(50, 50, kernel_size=3, stride=1, padding=1, groups=50),
#             nn.Conv1d(50, 40, kernel_size=1)
#         )
        
#         # Max pooling layers
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
#         # Skip connections (1x1 conv to match dimensions)
#         self.skip1 = nn.Conv1d(input_channels, 63, kernel_size=1, stride=1)
#         self.skip2 = nn.Conv1d(63, 50, kernel_size=1, stride=1)
#         self.skip3 = nn.Conv1d(50, 40, kernel_size=1, stride=1)
        
#         # Layer normalization
#         self.ln = nn.LayerNorm([40, 31])
        
#         # Dropout
#         self.dropout = nn.Dropout(0.2)
        
#         # Fully connected layer
#         self.fc = nn.Linear(40 * 31, embed_dim)

#     def forward(self, x):
#         assert x.shape[1] == 63 and x.shape[2] == 250, f"Expected (batch, 63, 250), got {x.shape}"
        
#         # Block 1: Convolution + MaxPool + Skip
#         identity = self.skip1(x)  # (B, 63, 250)
#         x = F.relu(self.conv1(x))  # (B, 63, 250)
#         x = self.pool(x)  # (B, 63, 125)
#         identity = self.pool(identity)  # (B, 63, 125)
#         x = x + identity  # (B, 63, 125)
        
#         # Block 2: Convolution + MaxPool + Skip
#         identity = self.skip2(x)  # (B, 50, 125)
#         x = F.relu(self.conv2(x))  # (B, 50, 125)
#         x = self.pool(x)  # (B, 50, 63)
#         identity = self.pool(identity)  # (B, 50, 63)
#         x = x + identity  # (B, 50, 63)
        
#         # Block 3: Convolution + MaxPool + Skip
#         identity = self.skip3(x)  # (B, 40, 63)
#         x = F.relu(self.conv3(x))  # (B, 40, 63)
#         x = self.pool(x)  # (B, 40, 33)
#         identity = self.pool(identity)  # (B, 40, 33)
#         x = x + identity  # (B, 40, 33)
        
#         # Layer norm and dropout
#         x = self.ln(x)  # (B, 40, 33)
#         x = self.dropout(x)  # (B, 40, 33)
        
#         # Flatten and fully connected layer
#         x = x.view(x.size(0), -1)  # (B, 1320)
#         x = self.fc(x)  # (B, 768)
        
#         return x

if __name__=="__main__":

    # Create synthetic EEG data (batch_size=4, channels=63, time_samples=250)
    batch_size = 4
    eeg_data = torch.randn(batch_size, 63, 250)
    clip_embeddings = torch.randn(batch_size, 512)  # CLIP feature embeddings

    # Initialize and test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTrans().to(device)
    eeg_data = eeg_data.to(device)
    output = model(eeg_data)

    print(f"Input EEG shape: {eeg_data.shape}")  # Should be (4, 63, 250)
    print(f"Output embedding shape: {output.shape}")  # Should be (4, 512)