import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Define the Transformer-based Autoencoder
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_channels, seq_length, embedding_dim=768, num_heads=8, num_layers=4):
        super(TransformerAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Encoder: Embedding Layer (project input EEG data to higher dimension)
        self.embedding = nn.Linear(input_channels, embedding_dim)
        
        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Decoder: A Linear layer to output back to original input size
        self.decoder = nn.Linear(embedding_dim, input_channels)
        
    def forward(self, x, encoder_only=False):
        # x shape: [batch_size, seq_length, input_channels]
        # batch_size, seq_length, _ = x.size()
        
        # Flatten the EEG data into a form suitable for transformer (seq_length, batch_size, input_channels)
        x = x.permute(1, 0, 2)  # [seq_length, batch_size, input_channels]
        
        # Pass the EEG data through embedding layer (Seq length, batch_size, embedding_dim)
        x = self.embedding(x)
        
        # Pass through Transformer Encoder
        x = self.encoder(x)

        # print(f"encoder out shape : {x.size()}")
        if encoder_only:
            encoder_last_t = x[-1, :, :]
            return encoder_last_t
        
        # Decoder (transform back to original input_channels size)
        x = x.permute(1, 0, 2)  # [batch_size, seq_length, embedding_dim]
        x = self.decoder(x)
        
        return x

if __name__=="__main__":
    
    # Create the model
    input_channels = 63   # Number of EEG channels
    seq_length = 256      # Length of the EEG sequence

    model = TransformerAutoencoder(input_channels=input_channels, seq_length=seq_length)

    # Example input: [batch_size, seq_length, input_channels]
    example_input = torch.randn(32, seq_length, input_channels)  # Example with batch_size=32

    # Forward pass
    output = model(example_input)

    print("Input shape:", example_input.shape)  # [batch_size, seq_length, input_channels]
    print("Output shape:", output.shape)        # [batch_size, seq_length, input_channels]