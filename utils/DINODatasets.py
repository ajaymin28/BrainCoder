import numpy as np
import os
import torch
from torch.utils.data import Dataset
import gc
import random
from collections import OrderedDict
from utils.common import TrainConfig
from collections import defaultdict

def eeg_global_aug(x, noise_std=0.05, jitter_prob=0.5, scaling_prob=0.5, mask_prob=0.2):
    """
    EEG sample (channels, time) or (batch, channels, time)
    """
    orig_shape = x.shape
    is_batch = (x.dim() == 3)
    if not is_batch:
        x = x.unsqueeze(0)  # Add batch dim: (1, C, T)

    # 1. Additive Gaussian Noise
    if torch.rand(1) < 0.7:
        x = x + torch.randn_like(x) * noise_std

    # 2. Channel Dropout
    if torch.rand(1) < 0.5:
        # mask shape: (channels,)
        mask = (torch.rand(x.shape[1], device=x.device) > mask_prob).float()
        x = x * mask.unsqueeze(0).unsqueeze(-1)

    # 3. Time Jitter (random roll)
    if torch.rand(1) < jitter_prob:
        shift = torch.randint(-15, 16, (1,)).item()
        x = torch.roll(x, shifts=shift, dims=-1)

    # 4. Channel-wise Scaling (multiplicative noise)
    if torch.rand(1) < scaling_prob:
        scale = 1.0 + (torch.randn(x.shape[1], device=x.device) * 0.1)
        x = x * scale.unsqueeze(0).unsqueeze(-1)

    # 5. Random Temporal Masking
    if torch.rand(1) < 0.5:
        t = x.shape[-1]
        mask_len = torch.randint(10, 40, (1,)).item()
        if t - mask_len > 0:
            start = torch.randint(0, t - mask_len, (1,)).item()
            x[..., start:start+mask_len] = 0

    if not is_batch:
        x = x.squeeze(0)  # Remove batch dim
    return x

def eeg_local_aug(x, min_len=64, max_len=100, noise_std=0.05, crop_prob=1.0):
    """
    EEG sample (channels, time) or (batch, channels, time)
    """
    orig_shape = x.shape
    is_batch = (x.dim() == 3)
    if not is_batch:
        x = x.unsqueeze(0)

    t = x.shape[-1]
    c = x.shape[1]

    # 1. Random Temporal Crop
    crop_len = torch.randint(min_len, max_len + 1, (1,)).item()
    if crop_prob >= 1.0 or torch.rand(1) < crop_prob:
        if t - crop_len + 1 > 0:
            start = torch.randint(0, t - crop_len + 1, (1,)).item()
            x = x[:, :, start:start+crop_len]

    # 2. Add Gaussian Noise
    if torch.rand(1) < 0.7:
        x = x + torch.randn_like(x) * noise_std

    # 3. Small Channel Scaling
    if torch.rand(1) < 0.5:
        scale = 1.0 + (torch.randn(c, device=x.device) * 0.05)
        x = x * scale.unsqueeze(0).unsqueeze(-1)

    if not is_batch:
        x = x.squeeze(0)
    return x




class DINOV2EEGDataset(Dataset):
    def __init__(
        self, 
        args: TrainConfig,
        subject_ids=[1],             # List of subject numbers, e.g. [1,2,3]
        session_ids=[0],             # List of session indices (0-3)
        subset="train",
        n_global_crops=2,
        n_local_crops=6,
        transform_global=None,   # DINO global aug function
        transform_local=None,    # DINO local aug function
        keep_dim_after_mean=False,
        mean_eeg_data=False,
        seed=42,
        max_cache_size=2  # <- How many subject files to keep in RAM per worker
    ):
        super().__init__()
        self.args = args
        self.subject_ids = subject_ids
        self.session_ids = session_ids
        self.subset = subset
        self.keep_dim_after_mean = keep_dim_after_mean
        self.mean_eeg_data = mean_eeg_data
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.transform_global = transform_global
        self.transform_local = transform_local
        self.rng = random.Random(seed)
        self.max_cache_size = max_cache_size
        self._file_cache = OrderedDict()  # subject_id -> eeg_data np.array
        self.n_same_class_aug = n_local_crops + n_global_crops

        # --- Load image metadata & features (shared for all subjects) ---
        img_meta_dir = os.path.join(args.global_config.DATA_BASE_DIR, "Data", "Things-EEG2", 'Image_set')
        img_metadata = np.load(os.path.join(img_meta_dir, 'image_metadata.npy'), allow_pickle=True).item()

        data_key = "train" if subset in {"train", "val"} else "test"
        self.img_file_names = img_metadata[f'{data_key}_img_files']
        self.img_concepts = img_metadata[f'{data_key}_img_concepts']
        self.class_to_id = {c: i for i, c in enumerate(sorted(set(self.img_concepts)))}

        if self.subset == "train" or self.subset == "val":
            img_feature_path = os.path.join(args.global_config.IMG_DATA_PATH, self.args.dnn + '_feature_maps_training.npy')
        else:
            img_feature_path = os.path.join(args.global_config.IMG_DATA_PATH, self.args.dnn + '_feature_maps_test.npy')
        self.img_features = np.load(img_feature_path, allow_pickle=True).squeeze()

        # --- Build master index: (subject, class_idx, session_idx) for random access ---
        self.master_index = []
        for subject_idx, subject_id in enumerate(subject_ids):
            eegfilekey = "training" if data_key == "train" else data_key
            eeg_path = os.path.join(self.args.global_config.EEG_DATA_PATH, f"sub-{subject_id:02d}", f"preprocessed_eeg_{eegfilekey}.npy")
            eeg_dict = np.load(eeg_path, allow_pickle=True)
            eeg_data = eeg_dict['preprocessed_eeg_data'] # (1654, 4, 63, 250)
            num_classes, num_sessions = eeg_data.shape[:2]
            for class_idx in range(num_classes):
                for sess in session_ids:
                    self.master_index.append({
                        "subject_idx": subject_idx,
                        "subject_id": subject_id,
                        "class_idx": class_idx,
                        "session_idx": sess,
                        "sample_idx_in_file": class_idx  # as class_idx==sample_idx in file
                    })
            del eeg_dict, eeg_data
            gc.collect()

        self.class_sample_lookup = defaultdict(list)
        for entry in self.master_index:
            self.class_sample_lookup[entry["class_idx"]].append(
                (entry["subject_id"], entry["session_idx"])
            )

    def __len__(self):
        return len(self.master_index)

    def _get_subject_data(self, subject_id):
        # Try cache first
        if subject_id in self._file_cache:
            # Move to end to show it was recently used
            self._file_cache.move_to_end(subject_id)
            return self._file_cache[subject_id]
        # Else, load file and add to cache
        eegfilekey = "training" if self.subset in {"train", "val"} else self.subset
        eeg_path = os.path.join(self.args.global_config.EEG_DATA_PATH, f"sub-{subject_id:02d}", f"preprocessed_eeg_{eegfilekey}.npy")
        eeg_dict = np.load(eeg_path, allow_pickle=True)
        eeg_data = eeg_dict['preprocessed_eeg_data']
        del eeg_dict
        gc.collect()
        # Cache management: pop oldest if needed
        if len(self._file_cache) >= self.max_cache_size:
            self._file_cache.popitem(last=False)
        self._file_cache[subject_id] = eeg_data
        return eeg_data

    def __getitem__(self, idx):
        sample_info = self.master_index[idx]
        subject_id = sample_info["subject_id"]
        class_idx = sample_info["class_idx"]
        session_idx = sample_info["session_idx"]

        # Efficient subject file caching:
        eeg_data = self._get_subject_data(subject_id)
        eeg_sample = eeg_data[class_idx, session_idx, :, :]

        # Get image features and class id
        image_features = self.img_features[class_idx]
        class_id = self.class_to_id[self.img_concepts[class_idx]]

        # ========== DINOv2 Augmentations ==========
        crops = []

        # add original sample from the current index
        crops.append(torch.tensor(eeg_sample, dtype=torch.float32))


        # # ---- Now get Same-class Augmentations (across subject/session) ----
        # # same_class_augs = []
        # # List of (subject_id, session_idx) for the class
        # candidates = self.class_sample_lookup[class_idx].copy()
        # # Remove current (subject, session) to avoid duplicates
        # candidates = [(sid, sess) for (sid, sess) in candidates if not (sid == subject_id and sess == session_idx)]
        # random.shuffle(candidates)

        # num_augs = min(self.n_same_class_aug, len(candidates))
        # # print(num_augs, len(candidates))
        # for i in range(num_augs):
        #     aug_subj, aug_sess = candidates[i]
        #     aug_eeg = self._get_subject_data(aug_subj)[class_idx, aug_sess, :, :]
        #     # You can use transform_global or a dedicated transform for these:
        #     aug_crop = torch.tensor(aug_eeg, dtype=torch.float32) 
        #     crops.append(aug_crop)

        return {
            "crops": crops,
            "class_id": class_id,
            "image_features": image_features,
            "subject_id": subject_id,
            "class_idx": class_idx,
            "session_idx": session_idx,
        }
    

class DINOV2EEGDatasetKaggle(Dataset):
    def __init__(
        self, 
        args: TrainConfig,
        subject_ids=[1],             # List of subject numbers, e.g. [1,2,3]
        session_ids=[0],             # List of session indices (0-3)
        subset="train",
        n_global_crops=2,
        n_local_crops=6,
        transform_global=None,   # DINO global aug function
        transform_local=None,    # DINO local aug function
        keep_dim_after_mean=False,
        mean_eeg_data=False,
        seed=42,
        max_cache_size=2  # <- How many subject files to keep in RAM per worker
    ):
        super().__init__()
        self.args = args
        self.subject_ids = subject_ids
        self.session_ids = session_ids
        self.subset = subset
        self.keep_dim_after_mean = keep_dim_after_mean
        self.mean_eeg_data = mean_eeg_data
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.transform_global = transform_global
        self.transform_local = transform_local
        self.rng = random.Random(seed)
        self.max_cache_size = max_cache_size
        self._file_cache = OrderedDict()  # subject_id -> eeg_data np.array
        self.n_same_class_aug = n_local_crops + n_global_crops

        # --- Load image metadata & features (shared for all subjects) ---
        # img_meta_dir = os.path.join(args.global_config.DATA_BASE_DIR, "Data", "Things-EEG2", 'Image_set')
        img_metadata = np.load(os.path.join(args.global_config.IMG_DATA_PATH, 'image_metadata.npy'), allow_pickle=True).item()

        data_key = "train" if subset in {"train", "val"} else "test"
        self.img_file_names = img_metadata[f'{data_key}_img_files']
        self.img_concepts = img_metadata[f'{data_key}_img_concepts']
        self.class_to_id = {c: i for i, c in enumerate(sorted(set(self.img_concepts)))}

        if self.subset == "train" or self.subset == "val":
            img_feature_path = os.path.join(args.global_config.TEST_CENTER_PATH, self.args.dnn + '_feature_maps_training.npy')
        else:
            img_feature_path = os.path.join(args.global_config.TEST_CENTER_PATH, self.args.dnn + '_feature_maps_test.npy')
        self.img_features = np.load(img_feature_path, allow_pickle=True).squeeze()

        # --- Build master index: (subject, class_idx, session_idx) for random access ---
        self.master_index = []
        for subject_idx, subject_id in enumerate(subject_ids):
            eegfilekey = "training" if data_key == "train" else data_key
            eeg_path = os.path.join(self.args.global_config.EEG_DATA_PATH, f"sub-{subject_id:02d}", f"preprocessed_eeg_{eegfilekey}.npy")
            eeg_dict = np.load(eeg_path, allow_pickle=True)
            eeg_data = eeg_dict['preprocessed_eeg_data'] # (1654, 4, 63, 250)
            num_classes, num_sessions = eeg_data.shape[:2]
            for class_idx in range(num_classes):
                for sess in session_ids:
                    self.master_index.append({
                        "subject_idx": subject_idx,
                        "subject_id": subject_id,
                        "class_idx": class_idx,
                        "session_idx": sess,
                        "sample_idx_in_file": class_idx  # as class_idx==sample_idx in file
                    })
            del eeg_dict, eeg_data
            gc.collect()

        self.class_sample_lookup = defaultdict(list)
        for entry in self.master_index:
            self.class_sample_lookup[entry["class_idx"]].append(
                (entry["subject_id"], entry["session_idx"])
            )

    def __len__(self):
        return len(self.master_index)

    def _get_subject_data(self, subject_id):
        # Try cache first
        if subject_id in self._file_cache:
            # Move to end to show it was recently used
            self._file_cache.move_to_end(subject_id)
            return self._file_cache[subject_id]
        # Else, load file and add to cache
        eegfilekey = "training" if self.subset in {"train", "val"} else self.subset
        eeg_path = os.path.join(self.args.global_config.EEG_DATA_PATH, f"sub-{subject_id:02d}", f"preprocessed_eeg_{eegfilekey}.npy")
        eeg_dict = np.load(eeg_path, allow_pickle=True)
        eeg_data = eeg_dict['preprocessed_eeg_data']
        del eeg_dict
        gc.collect()
        # Cache management: pop oldest if needed
        if len(self._file_cache) >= self.max_cache_size:
            self._file_cache.popitem(last=False)
        self._file_cache[subject_id] = eeg_data
        return eeg_data

    def __getitem__(self, idx):
        sample_info = self.master_index[idx]
        subject_id = sample_info["subject_id"]
        class_idx = sample_info["class_idx"]
        session_idx = sample_info["session_idx"]

        # Efficient subject file caching:
        eeg_data = self._get_subject_data(subject_id)
        eeg_sample = eeg_data[class_idx, session_idx, :, :]

        # Get image features and class id
        image_features = self.img_features[class_idx]
        class_id = self.class_to_id[self.img_concepts[class_idx]]

        # ========== DINOv2 Augmentations ==========
        crops = []

        # add original sample from the current index
        crops.append(torch.tensor(eeg_sample, dtype=torch.float32))


        # # ---- Now get Same-class Augmentations (across subject/session) ----
        # # same_class_augs = []
        # # List of (subject_id, session_idx) for the class
        # candidates = self.class_sample_lookup[class_idx].copy()
        # # Remove current (subject, session) to avoid duplicates
        # candidates = [(sid, sess) for (sid, sess) in candidates if not (sid == subject_id and sess == session_idx)]
        # random.shuffle(candidates)

        # num_augs = min(self.n_same_class_aug, len(candidates))
        # # print(num_augs, len(candidates))
        # for i in range(num_augs):
        #     aug_subj, aug_sess = candidates[i]
        #     aug_eeg = self._get_subject_data(aug_subj)[class_idx, aug_sess, :, :]
        #     # You can use transform_global or a dedicated transform for these:
        #     aug_crop = torch.tensor(aug_eeg, dtype=torch.float32) 
        #     crops.append(aug_crop)

        return {
            "crops": crops,
            "class_id": class_id,
            "image_features": image_features,
            "subject_id": subject_id,
            "class_idx": class_idx,
            "session_idx": session_idx,
        }
    

if __name__=="__main__":

    args = TrainConfig()

    dataset = DINOV2EEGDataset(
        args=args,
        subject_ids=[1,2],  # or [1] or up to [1,2,...,10]
        session_ids=[0,1,2,3],    # select sessions you want
        subset="train",
        n_global_crops=2,
        n_local_crops=6,
        transform_global=eeg_global_aug,
        transform_local=eeg_local_aug,
        max_cache_size=2
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)


    for databatch in dataloader:
        crops = databatch["crops"]