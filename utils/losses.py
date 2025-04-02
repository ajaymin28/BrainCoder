import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import os

class Parameters:
    ce_loss_weight = 0.95
    mse_loss_weight = 0.20
    soft_target_loss_weight = 0.05
    alpha = 0.5
    teacher_temp = 0.05
    student_temp=0.1

class HyperParams:
    learning_rate=0.001
    T=0.5
    soft_target_loss_weight=0.25
    ce_loss_weight=0.75
    warmup_teacher_temp = 0.07
    teacher_temp = 0.07
    warmup_teacher_temp_epochs = 5
    alpha = 0.5
    beta = 0.5

def orthogonality_loss(task_features, subject_features):
    """
    Enforces that task-related features and subject-related features are uncorrelated.
    """
    # Compute dot product between task and subject features
    loss_ortho = torch.norm(task_features.T @ subject_features, p='fro')  # Frobenius norm
    return loss_ortho

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))


        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'


        dist.init_process_group(
            # backend="nccl",
            backend="gloo",
            init_method="env://",
            world_size=1,
            rank=0,
        )


    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        HyperParams.T = self.teacher_temp_schedule[epoch]

        student_out = student_output / self.student_temp
        # student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output) / temp, dim=-1)
        # teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        total_loss += loss.mean()

        # total_loss = 0
        # n_loss_terms = 0
        # for iq, q in enumerate(teacher_out):
        #     for v in range(len(student_out)):
        #         if v == iq:
        #             # we skip cases where student and teacher operate on the same view
        #             continue
        #         loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        #         total_loss += loss.mean()
        #         n_loss_terms += 1
        # total_loss /= n_loss_terms

        # self.update_center(teacher_output)
        return total_loss, temp

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)   


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

# def get_contrastive_loss(feat1, feat2, labels, logit_scale):
#     """
#     Compute contrastive loss for DDP training.
    
#     Args:
#         feat1: First feature tensor (batch_size, embedding_dim)
#         feat2: Second feature tensor (batch_size, embedding_dim)
#         labels: Ground truth labels (batch_size)
#         logit_scale: Scaling parameter for logits
#     """
#     # Normalize features (per-GPU operation)
#     feat1 = F.normalize(feat1, dim=1)
#     feat2 = F.normalize(feat2, dim=1)
    
#     # Exponentiate logit scale (per-GPU)
#     logit_scale = logit_scale.exp()

#     # Compute logits efficiently (per-GPU)
#     logits_f1 = logit_scale * torch.einsum('ik,jk->ij', feat1, feat2)
    
#     # Compute contrastive loss for this GPU's batch
#     loss_f1 = F.cross_entropy(logits_f1, labels)
#     loss_f2 = F.cross_entropy(logits_f1.t(), labels)
#     local_loss = (loss_f1 + loss_f2) / 2
    
#     # Reduce loss across all GPUs
#     loss_tensor = torch.tensor(local_loss.item(), device=feat1.device)
#     dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
#     global_loss = loss_tensor / dist.get_world_size()

#     # Return local loss for backward pass, but use global loss for logging
#     return local_loss  # local_loss is used for gradients, global_loss could be logged if needed

def get_contrastive_loss(feat1, feat2, labels, logit_scale):
    feat1 = F.normalize(feat1, dim=1)
    feat2 = F.normalize(feat2, dim=1)
    logit_scale = logit_scale.exp()
    
    # Compute logits efficiently
    logits_f1 = logit_scale * torch.einsum('ik,jk->ij', feat1, feat2)
    
    # Compute contrastive loss
    loss = (F.cross_entropy(logits_f1, labels) + F.cross_entropy(logits_f1.t(), labels)) / 2
    return loss

# def get_contrastive_loss(feat1, feat2, labels, logit_scale):
#     feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
#     feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
#     logit_scale = logit_scale.exp()
#     logits_f1 = logit_scale * feat1 @ feat2.t()
#     logits_f2 = logits_f1.t()
#     l_contrastive_align_f1 = F.cross_entropy(logits_f1, labels)
#     l_contrastive_align_f2 = F.cross_entropy(logits_f2, labels)
#     loss = (l_contrastive_align_f1 + l_contrastive_align_f2) / 2
#     return loss