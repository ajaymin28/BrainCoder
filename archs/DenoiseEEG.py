class DAE(nn.Module):
    def __init__(self, input_channels, seq_length, embedding_dim=256, num_heads=8, num_layers=4):
        super(DAE, self).__init__()
        
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
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_channels]
        batch_size, seq_length, _ = x.size()
        
        # Flatten the EEG data into a form suitable for transformer (seq_length, batch_size, input_channels)
        x = x.permute(1, 0, 2)  # [seq_length, batch_size, input_channels]
        
        # Pass the EEG data through embedding layer (Seq length, batch_size, embedding_dim)
        x = self.embedding(x)
        
        # Pass through Transformer Encoder
        x = self.encoder(x)
        
        # Decoder (transform back to original input_channels size)
        x = x.permute(1, 0, 2)  # [batch_size, seq_length, embedding_dim]
        x = self.decoder(x)
        
        return x

    def add_noise(self, x, noise_factor=0.3):
        """Add noise to the input EEG data."""
        noise = torch.randn_like(x) * noise_factor
        return x + noise



# Denoising Autoencoder Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Add noise to the batch
        noisy_batch = model.add_noise(batch)
        
        # Forward pass
        recon_batch = model(noisy_batch)
        
        # Compute the MSE loss
        loss = F.mse_loss(recon_batch, batch, reduction='sum')
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")