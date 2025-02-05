import torch
import torch.nn.functional as F
import torch.nn as nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, student_outputs, teacher_outputs):
        loss = 1 - self.cosine_similarity(student_outputs, teacher_outputs).mean()
        return loss

class SoftmaxThresholdLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, temperature=1.0):
        """
        Custom loss function with a softmax-based threshold.

        Args:
            alpha (float): Weight for the common features term in the loss.
            beta (float): Weight for the difference features term in the loss.
            temperature (float): Temperature parameter for softmax scaling.
        """
        super(SoftmaxThresholdLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, v1, v2):
        """
        Calculate the loss based on softmax-weighted common and different features.

        Args:
            v1 (torch.Tensor): First feature vector (batch x features).
            v2 (torch.Tensor): Second feature vector (batch x features).

        Returns:
            torch.Tensor: Scalar loss value.
        """

        # SEjemb = v1 - v2
        # SEkemb = v2 - v1


        # Compute absolute differences between the two vectors
        differences = torch.abs(v1 - v2)

        # Apply softmax over differences (per feature, across batch dimension)
        softmax_weights = torch.softmax(-differences / self.temperature, dim=-1)

        # Weighted commonality and difference terms
        common_features = softmax_weights * v1
        common_loss = torch.sum((common_features - v1) ** 2, dim=1).mean()

        different_features = softmax_weights * differences
        difference_loss = torch.sum(different_features, dim=1).mean()

        # Weighted combination of the two losses
        total_loss = self.alpha * common_loss + self.beta * difference_loss

        return total_loss, common_loss, difference_loss

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