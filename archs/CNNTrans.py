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

class SubjectDiscriminator(nn.Module):
    def __init__(self,encoder,embed_dim=768, num_subjects=1, alpha=1.0):
        super(SubjectDiscriminator, self).__init__()

        self.alpha = alpha
        self.num_subjects = num_subjects
        self.encoder = encoder

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
    
    def forward(self, x, alpha=None):
        if alpha is None: 
            alpha = self.alpha

        x = self.encoder(x)
        reversed_x = grad_reverse(x, alpha)  # Apply GRL
        subject_features = self.feature_extractor(reversed_x)  # Extract subject features
        subject_pred = self.classifier(subject_features)  # Predict subject
        return subject_pred, subject_features  # Return both

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