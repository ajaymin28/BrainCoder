import numpy as np
import os
import torch
from torch.utils.data import Dataset
import gc
import random
from collections import OrderedDict
from utils.common import TrainConfig

def dummy_dino_global_aug(x):  # replace with your EEG augmentation
    # Example: Add random Gaussian noise as "augmentation"
    noise = torch.randn_like(x) * 0.05
    return x + noise

def dummy_dino_local_aug(x, time_samples_to_preserve=120):
    # Example: Crop 120 random timepoints as local
    idx = torch.randint(0, x.shape[-1]-time_samples_to_preserve, (1,)).item()
    return x[:, idx:idx+time_samples_to_preserve]



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
        for _ in range(self.n_global_crops):
            crop = self.transform_global(torch.tensor(eeg_sample, dtype=torch.float32)) if self.transform_global else torch.tensor(eeg_sample, dtype=torch.float32)
            crops.append(crop)
        for _ in range(self.n_local_crops):
            crop = self.transform_local(torch.tensor(eeg_sample, dtype=torch.float32)) if self.transform_local else torch.tensor(eeg_sample, dtype=torch.float32)
            crops.append(crop)

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
        transform_global=dummy_dino_global_aug,
        transform_local=dummy_dino_local_aug,
        max_cache_size=2
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)


    for databatch in dataloader:
        crops = databatch["crops"]

