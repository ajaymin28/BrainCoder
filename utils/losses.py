import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import os

def cosine_scheduler(base_value, final_value, epochs, niter_per_epoch, 
                    warmup_epochs=0, start_warmup_value=0):
    """
    Cosine scheduler with optional linear warmup.
    
    Returns a list of length epochs * niter_per_epoch.
    """
    warmup_schedule = []
    if warmup_epochs > 0:
        warmup_iters = warmup_epochs * niter_per_epoch
        # Linear warmup from start_warmup_value to base_value
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    else:
        warmup_iters = 0

    # Cosine schedule after warmup
    total_iters = epochs * niter_per_epoch
    iters = np.arange(total_iters - warmup_iters)
    cosine_schedule = final_value + 0.5 * (base_value - final_value) * \
        (1 + np.cos(np.pi * iters / (total_iters - warmup_iters)))

    schedule = np.concatenate((warmup_schedule, cosine_schedule))
    assert len(schedule) == total_iters
    return schedule

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


def get_contrastive_loss(feat1, feat2, labels, logit_scale):
    feat1 = F.normalize(feat1, dim=1)
    feat2 = F.normalize(feat2, dim=1)
    logit_scale = logit_scale.exp()
    
    # Compute logits efficiently
    logits_f1 = logit_scale * torch.einsum('ik,jk->ij', feat1, feat2)
    
    # Compute contrastive loss
    loss = (F.cross_entropy(logits_f1, labels) + F.cross_entropy(logits_f1.t(), labels)) / 2
    return loss



def vae_loss(recon_x, x, mu, logvar, projected_z, image_features, subject_logits, subject_labels, logit_scale, use_feature_confusion, beta=1.0, alpha=1.0, gamma=1.0):
    if recon_x is not None:
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        recon_loss = torch.tensor(0.0, device=x.device)
        kl_loss = torch.tensor(0.0, device=x.device)

    proj_z_normalized = F.normalize(projected_z, dim=-1)
    image_features_normalized = F.normalize(image_features, dim=-1)
    logits = torch.matmul(proj_z_normalized, image_features_normalized.T) * logit_scale.exp().clamp(1, 100)
    labels = torch.arange(proj_z_normalized.shape[0], device=proj_z_normalized.device)
    contrastive_loss = F.cross_entropy(logits, labels)

    if use_feature_confusion:
        probs = F.softmax(subject_logits, dim=-1)
        uniform = torch.full_like(probs, 1.0 / probs.size(1))
        subject_loss = F.kl_div(probs.log(), uniform, reduction='batchmean')
        subject_loss_clamped = torch.clamp(subject_loss, max=1.0)
        total_loss = recon_loss + beta * kl_loss + alpha * contrastive_loss + gamma * subject_loss_clamped
    else:
        subject_loss = F.cross_entropy(subject_logits, subject_labels)
        subject_loss_clamped = torch.clamp(subject_loss, max=1.0)
        total_loss = recon_loss + beta * kl_loss + alpha * contrastive_loss - gamma * subject_loss_clamped

    return total_loss, recon_loss, kl_loss, contrastive_loss, subject_loss_clamped