import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientCNNTrans(nn.Module):
    def __init__(self, input_channels=63, time_samples=250, embed_dim=768):
        super(EfficientCNNTrans, self).__init__()

        self.init_conv = nn.Conv2d(1, 64, kernel_size=(3,3), padding=(1,1))

        # Handle (channels, time) input shape carefully
        self.block1 = self._make_block(64, 64, stride=(2,2))
        self.block2 = self._make_block(64, 48, stride=(2,2))
        self.block3 = self._make_block(48, 32, stride=(2,2))

        self.attention_pool = AttentionPooling2D(32)
        self.fc = nn.Linear(32, embed_dim)
        self.dropout = nn.Dropout(0.3)

    def _make_block(self, in_channels, out_channels, stride=(2,2)):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),  # depthwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # pointwise
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock2D(out_channels),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.attention_pool(x)  # (B, C)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock2D, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class AttentionPooling2D(nn.Module):
    def __init__(self, channels):
        super(AttentionPooling2D, self).__init__()
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        attn = self.attention(x)  # (B, 1, H, W)
        attn = attn.view(b, 1, -1)
        attn = F.softmax(attn, dim=-1)
        x = x.view(b, c, -1)
        pooled = torch.bmm(x, attn.transpose(1, 2)).squeeze(-1)  # (B, C)
        return pooled


if __name__=="__main__":

    # Test Script
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a model instance
    model = EfficientCNNTrans(input_channels=63, time_samples=250, embed_dim=768).to(device)
    model.eval()

    # Create dummy EEG data
    dummy_eeg = torch.randn(8, 63, 250).to(device)  # (batch, channels, time)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_eeg)

    print("Input shape:", dummy_eeg.shape)
    print("Output shape:", output.shape)