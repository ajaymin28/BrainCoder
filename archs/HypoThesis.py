import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Synthetic CLIP features
num_classes, clip_dim = 5, 512
clip_features = torch.randn(num_classes, clip_dim)

# EEG Dataset
class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels, subject_ids=None):
        self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.subject_ids = torch.tensor(subject_ids, dtype=torch.long) if subject_ids is not None else None
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        if self.subject_ids is not None:
            return self.eeg_data[idx], self.labels[idx], self.subject_ids[idx]
        return self.eeg_data[idx], self.labels[idx]

# Modified EEG Decomposer Model
class EEGDecomposer(nn.Module):
    def __init__(self, eeg_channels=63, eeg_timesteps=250, clip_dim=512, subject_dim=128, num_subjects=3):
        super(EEGDecomposer, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(eeg_channels, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Flatten()
        )
        
        self.flattened_size = 256 * 63
        self.linear = nn.Linear(self.flattened_size, clip_dim + subject_dim)
        self.subject_embedding = nn.Embedding(num_subjects, subject_dim)
        
        self.clip_dim = clip_dim
    
    def forward(self, x, subject_ids=None):
        E = self.encoder(x)
        E = self.linear(E)
        Ec = E[:, :self.clip_dim]
        Esub = E[:, self.clip_dim:]
        
        if subject_ids is not None:
            subject_emb = self.subject_embedding(subject_ids)
            return Ec, Esub, subject_emb
        return Ec, Esub

# Loss Function
def compute_loss(Ec, Esub, subject_emb, true_image_features, subject_ids):
    mse_loss = nn.MSELoss()
    cos_sim = nn.CosineSimilarity(dim=1)
    
    image_loss = mse_loss(Ec, true_image_features)
    subject_loss = 1 - cos_sim(Esub, subject_emb).mean()
    
    batch_size = subject_ids.size(0)
    contrastive_loss = 0.0
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            is_same = (subject_ids[i] == subject_ids[j]).float()
            dist = torch.norm(Esub[i] - Esub[j])
            contrastive_loss += is_same * dist ** 2 + (1 - is_same) * max(1.0 - dist, 0) ** 2
    contrastive_loss = contrastive_loss / (batch_size * (batch_size - 1) / 2 + 1e-8)
    
    return image_loss + subject_loss + 0.1 * contrastive_loss, (image_loss, subject_loss, contrastive_loss)

# Training Function
def train_model(model, dataloader, clip_features, num_epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    cos_sim = nn.CosineSimilarity(dim=1)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        image_losses, subj_losses, cont_losses = [], [], []
        
        for batch in dataloader:
            eeg_data, labels, subject_ids = batch  # Assumes subject_ids are provided
            optimizer.zero_grad()
            Ec, Esub, subject_emb = model(eeg_data, subject_ids)
            true_image_features = clip_features[labels]
            
            loss, (img_loss, sub_loss, con_loss) = compute_loss(Ec, Esub, subject_emb, true_image_features, subject_ids)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            image_losses.append(img_loss.item())
            subj_losses.append(sub_loss.item())
            cont_losses.append(con_loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {avg_loss:.4f}")
        print(f"  Image Loss: {np.mean(image_losses):.4f}, Subject Loss: {np.mean(subj_losses):.4f}, Contrastive Loss: {np.mean(cont_losses):.4f}")

# Testing Function (no subject_ids)
def test_model(model, dataloader, clip_features):
    model.eval()
    cos_sim = nn.CosineSimilarity(dim=1)
    
    with torch.no_grad():
        for batch in dataloader:
            eeg_data, labels = batch  # No subject_ids
            Ec, Esub = model(eeg_data)  # subject_ids=None by default
            true_image_features = clip_features[labels]
            
            img_similarity = cos_sim(Ec, true_image_features).mean().item()
            print(f"Test: Ec vs CLIP Similarity: {img_similarity:.4f}")
            
            # Optionally inspect Esub (no ground-truth subject_emb to compare)
            print(f"Esub Sample (first 5 dims): {Esub[0, :5]}")
            break  # One batch for demo

# Run with Sample Data
if __name__ == "__main__":
    # Training data (with subject_ids)
    num_samples, batch_size = 100, 16
    train_eeg_data = np.random.randn(num_samples, 63, 250)
    train_labels = np.random.randint(0, num_classes, num_samples)
    train_subject_ids = np.random.randint(0, 3, num_samples)
    
    train_dataset = EEGDataset(train_eeg_data, train_labels, train_subject_ids)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Test data (no subject_ids)
    test_eeg_data = np.random.randn(20, 63, 250)
    test_labels = np.random.randint(0, num_classes, 20)
    test_dataset = EEGDataset(test_eeg_data, test_labels)  # No subject_ids
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = EEGDecomposer(clip_dim=clip_dim, subject_dim=128, num_subjects=3)
    
    # Train
    print("Training Model...")
    train_model(model, train_dataloader, clip_features, num_epochs=5)
    
    # Test
    print("\nTesting Model (no subject_ids)...")
    test_model(model, test_dataloader, clip_features)