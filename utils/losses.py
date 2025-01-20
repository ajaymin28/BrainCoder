import torch
import torch.nn.functional as F


def cosine_dissimilarity_loss(u, v):
    """
    Loss function to encourage dissimilarity between two feature vectors.
    
    Args:
        u (torch.Tensor): First feature vector (batch_size, feature_dim).
        v (torch.Tensor): Second feature vector (batch_size, feature_dim).
    
    Returns:
        torch.Tensor: Loss value.
    """
    # Normalize the vectors to ensure unit magnitude
    u = F.normalize(u, p=2, dim=-1)
    v = F.normalize(v, p=2, dim=-1)
    
    # Compute cosine similarity
    cosine_similarity = torch.sum(u * v, dim=-1)
    
    # Loss to encourage dissimilarity
    loss = torch.mean(1 + cosine_similarity)  # Adding 1 ensures non-negativity
    
    return loss


def kld_loss(p, q, epsilon=1e-7):
    # Convert to tensors
    # p = torch.tensor(p, dtype=torch.float32)
    # q = torch.tensor(q, dtype=torch.float32)
    
    # Normalize the tensors to represent probability distributions
    p = p / p.sum()
    q = q / q.sum()
    
    # Add epsilon for numerical stability
    q = q + epsilon
    
    # Compute KLD (log of Q is expected by F.kl_div)
    kld = F.kl_div(q.log(), p, reduction='batchmean')
    return kld.item()