import torch
import torch.nn as nn

class FTSurrogateEEG(nn.Module):
    """
    Batch FT surrogate for EEG data. Input: (batch, channels, time)
    """
    def __init__(self):
        super().__init__()

    def forward(self, eeg):
        batch, n_channels, t = eeg.shape
        device = eeg.device
        dtype = eeg.dtype
        fft_data = torch.fft.fft(eeg, dim=-1)
        amplitude = fft_data.abs()
        half = t // 2
        random_phases = torch.zeros_like(fft_data.real, device=device)
        if t % 2 == 0:
            random_phases[..., 0] = 0
            random_phases[..., half] = 0
            rand_vals = torch.rand((batch, n_channels, half - 1), device=device) * 2 * torch.pi
            random_phases[..., 1:half] = rand_vals
            random_phases[..., half+1:] = -rand_vals.flip(-1)
        else:
            random_phases[..., 0] = 0
            rand_vals = torch.rand((batch, n_channels, half), device=device) * 2 * torch.pi
            random_phases[..., 1:half+1] = rand_vals
            random_phases[..., half+1:] = -rand_vals.flip(-1)
        new_spectrum = amplitude * torch.exp(1j * random_phases)
        surrogate = torch.fft.ifft(new_spectrum, dim=-1).real
        return surrogate.type(dtype)

class EEGAug(nn.Module):
    """
    EEG global augmentation as nn.Module.
    Usage: aug = EEGGlobalAug(...); x_aug = aug(x)
    """
    def __init__(self, noise_std=0.05, mask_prob=1.0, min_len=200, max_len=230, crop_prob=1.0, ft_surrogate_prob=1.0, mask_patches=3):
        super().__init__()
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.min_len = min_len
        self.max_len = max_len
        self.crop_prob = crop_prob
        self.ft_surrogate_prob = ft_surrogate_prob
        self.ft_surrogate = FTSurrogateEEG()
        self.mask_patches = mask_patches

    def forward(self, x):
        batch, n_channels, t = x.shape
        x_aug = x
        # # --- 1. Random Temporal Crop ---
        # crop_len = torch.randint(self.min_len, self.max_len + 1, (1,)).item()
        # if self.crop_prob >= 1.0 or torch.rand(1).item() < self.crop_prob:
        #     if t > crop_len:
        #         start = torch.randint(0, t - crop_len + 1, (1,)).item()
        #         x_aug = x_aug[..., start:start + crop_len]
        #     else:
        #         crop_len = t  # fallback, no crop

        # --- 2. Additive Gaussian Noise ---
        if torch.rand(1).item() < 0.7:
            x_aug = x_aug + torch.randn_like(x_aug) * self.noise_std

        # # --- 3. FT Surrogate (batch) ---
        # if torch.rand(1).item() < self.ft_surrogate_prob:
        #     if x_aug.dim() == 2:
        #         x_aug = x_aug.unsqueeze(0)
        #         x_aug = self.ft_surrogate(x_aug)[0]
        #     else:
        #         x_aug = self.ft_surrogate(x_aug)

        # # --- 4. Random Temporal Masking ---
        if torch.rand(1).item() < self.mask_prob:
            for mpi in range(self.mask_patches):
                min_mask_len, max_mask_len = 10, 30
                t_current = x_aug.shape[-1]
                mask_len = min(torch.randint(min_mask_len, max_mask_len + 1, (1,)).item(), t_current)
                if t_current > mask_len:
                    start = torch.randint(0, t_current - mask_len + 1, (1,)).item()
                    x_aug[..., start:start + mask_len] = 0

        return x_aug


# def ft_surrogate_eeg_torch(eeg):
#     """
#     eeg: torch.Tensor, shape (batch, channels, t)
#     Returns: torch.Tensor, same shape (real part only)
#     """
#     # Set random seed per call if needed, but usually you want torch's global randomness
#     batch, n_channels, t = eeg.shape
#     device = eeg.device
#     dtype = eeg.dtype

#     # FFT
#     fft_data = torch.fft.fft(eeg, dim=-1)  # shape (B, C, T)
#     amplitude = fft_data.abs()
#     # Create random phases
#     half = t // 2
#     random_phases = torch.zeros_like(fft_data.real)
#     if t % 2 == 0:
#         random_phases[..., 0] = 0
#         random_phases[..., half] = 0
#         rand_vals = torch.rand((batch, n_channels, half - 1), device=device) * 2 * torch.pi
#         random_phases[..., 1:half] = rand_vals
#         random_phases[..., half+1:] = -rand_vals.flip(-1)
#     else:
#         random_phases[..., 0] = 0
#         rand_vals = torch.rand((batch, n_channels, half), device=device) * 2 * torch.pi
#         random_phases[..., 1:half+1] = rand_vals
#         random_phases[..., half+1:] = -rand_vals.flip(-1)

#     # Compose new FFT
#     new_spectrum = amplitude * torch.exp(1j * random_phases)
#     surrogate = torch.fft.ifft(new_spectrum, dim=-1).real
#     return surrogate.type(dtype)

# def eeg_global_aug_torch(
#     x: torch.Tensor,
#     noise_std=0.05, mask_prob=0.2, min_len=200, max_len=230, crop_prob=1.0, ft_surrogate_prob=1.0
# ):
#     """
#     x: torch.Tensor, (batch, n_channels, t)
#     Returns: torch.Tensor, possibly time-cropped, (batch, n_channels, t_out)
#     """
#     batch, n_channels, t = x.shape
#     x_aug = x.clone()

#     # --- 1. Random Temporal Crop ---
#     crop_len = torch.randint(min_len, max_len + 1, (1,)).item()
#     if crop_prob >= 1.0 or torch.rand(1).item() < crop_prob:
#         if t > crop_len:
#             start = torch.randint(0, t - crop_len + 1, (1,)).item()
#             x_aug = x_aug[..., start:start + crop_len]
#         else:
#             crop_len = t  # fallback, no crop

#     # --- 2. Additive Gaussian Noise ---
#     if torch.rand(1).item() < 0.7:
#         x_aug = x_aug + torch.randn_like(x_aug) * noise_std

#     # --- 3. FT Surrogate (batch) ---
#     if torch.rand(1).item() < ft_surrogate_prob:
#         # If batch size 1, keep shape consistent
#         if x_aug.dim() == 2:
#             x_aug = x_aug.unsqueeze(0)
#             x_aug = ft_surrogate_eeg_torch(x_aug)[0]
#         else:
#             x_aug = ft_surrogate_eeg_torch(x_aug)

#     # --- 4. Random Temporal Masking ---
#     if torch.rand(1).item() < mask_prob:
#         min_mask_len, max_mask_len = 10, 30
#         t_current = x_aug.shape[-1]
#         mask_len = min(torch.randint(min_mask_len, max_mask_len + 1, (1,)).item(), t_current)
#         if t_current > mask_len:
#             start = torch.randint(0, t_current - mask_len + 1, (1,)).item()
#             x_aug[..., start:start + mask_len] = 0

#     return x_aug


# def eeg_local_aug_torch(
#     x: torch.Tensor,
#     noise_std=0.09, mask_prob=0.2, min_len=165, max_len=200, crop_prob=1.0, ft_surrogate_prob=1.0
# ):
#     """
#     x: torch.Tensor, (batch, n_channels, t)
#     Returns: torch.Tensor, possibly time-cropped, (batch, n_channels, t_out)
#     """
#     batch, n_channels, t = x.shape
#     x_aug = x.clone()

#     # --- 1. Random Temporal Crop ---
#     crop_len = torch.randint(min_len, max_len + 1, (1,)).item()
#     if crop_prob >= 1.0 or torch.rand(1).item() < crop_prob:
#         if t > crop_len:
#             start = torch.randint(0, t - crop_len + 1, (1,)).item()
#             x_aug = x_aug[..., start:start + crop_len]
#         else:
#             crop_len = t  # fallback, no crop

#     # --- 2. Additive Gaussian Noise ---
#     if torch.rand(1).item() < 0.7:
#         x_aug = x_aug + torch.randn_like(x_aug) * noise_std

#     # --- 3. FT Surrogate (batch) ---
#     if torch.rand(1).item() < ft_surrogate_prob:
#         # If batch size 1, keep shape consistent
#         if x_aug.dim() == 2:
#             x_aug = x_aug.unsqueeze(0)
#             x_aug = ft_surrogate_eeg_torch(x_aug)[0]
#         else:
#             x_aug = ft_surrogate_eeg_torch(x_aug)

#     # --- 4. Random Temporal Masking ---
#     if torch.rand(1).item() < mask_prob:
#         min_mask_len, max_mask_len = 10, 30
#         t_current = x_aug.shape[-1]
#         mask_len = min(torch.randint(min_mask_len, max_mask_len + 1, (1,)).item(), t_current)
#         if t_current > mask_len:
#             start = torch.randint(0, t_current - mask_len + 1, (1,)).item()
#             x_aug[..., start:start + mask_len] = 0

#     return x_aug


if __name__=="__main__":

    # Simulate batch input
    x = torch.randn(16000, 64, 250).cuda()  # (batch, channels, time), on GPU

    # Initialize augmentation module (move to GPU if needed)
    global_aug = EEGAug(noise_std=0.05, ft_surrogate_prob=1.0, min_len=200, max_len=230).cuda()
    local_aug = EEGAug(noise_std=0.09, ft_surrogate_prob=1.0, min_len=165, max_len=190).cuda()
    global_aug.eval()
    local_aug.eval()


    # Forward pass (augmentation)
    gx_aug = global_aug(x)  # shape: (batch, channels, cropped_time)
    lx_aug = local_aug(x)  # shape: (batch, channels, cropped_time)
    print(gx_aug.shape, lx_aug.shape)
    print(gx_aug.mean(axis=0).mean(), gx_aug.std(axis=0).mean())
    print(lx_aug.mean(axis=0).mean(), lx_aug.std(axis=0).mean())