import numpy as np
import os
import torch
from torch.utils.data import Dataset
import gc
import random
from collections import OrderedDict
from utils.common import TrainConfig
from collections import defaultdict
import copy


def ft_surrogate_eeg(eeg: np.ndarray) -> np.ndarray:
    n_channels, t_current = eeg.shape
    fft_data = np.fft.fft(eeg, axis=-1)
    amplitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    random_phases = np.zeros_like(phase)
    half = t_current // 2
    np.random.seed(np.random.randint(0,500)) # make the random actually random, else seed will fix it
    if t_current % 2 == 0:
        random_phases[:, 0] = 0
        random_phases[:, half] = 0
        rand_vals = np.random.uniform(0, 2 * np.pi, (n_channels, half - 1))
        random_phases[:, 1:half] = rand_vals
        random_phases[:, half+1:] = -rand_vals[:, ::-1]
    else:
        random_phases[:, 0] = 0
        rand_vals = np.random.uniform(0, 2 * np.pi, (n_channels, half))
        random_phases[:, 1:half+1] = rand_vals
        random_phases[:, half+1:] = -rand_vals[:, ::-1]
    new_spectrum = amplitude * np.exp(1j * random_phases)
    surrogate = np.fft.ifft(new_spectrum, axis=-1).real
    return surrogate.astype(eeg.dtype)

def eeg_global_aug(
    x: np.ndarray, noise_std=0.05, jitter_prob=0.5, scaling_prob=0.5,
    mask_prob=0.2, min_len=200, max_len=230, crop_prob=1.0,
    ft_surrogate_prob=0.7
):
    """
    EEG sample augmentation with cropping, noise, masking, and FT surrogate.
    Input:
        x: np.ndarray, shape (n_channels, t)
    Returns:
        Augmented np.ndarray
    """

    np.random.seed(np.random.randint(0,500)) # make the random actually random, else seed will fix it

    n_channels, t = x.shape
    x_aug = np.copy(x)

    # 1. Random Temporal Crop
    crop_len = np.random.randint(min_len, max_len + 1)
    if crop_prob >= 1.0 or np.random.rand() < crop_prob:
        if t > crop_len:
            start = np.random.randint(0, t - crop_len + 1)
            x_aug = x_aug[:, start:start + crop_len]
        else:
            crop_len = t  # fallback, no crop

    # 2. Additive Gaussian Noise
    if np.random.rand() < 0.7:
        x_aug = x_aug + np.random.randn(*x_aug.shape) * noise_std

    # 3. FT Surrogate
    if np.random.rand() < ft_surrogate_prob:
        x_aug = ft_surrogate_eeg(x_aug)

    # 4. Random Temporal Masking
    if np.random.rand() < mask_prob:
        min_mask_len, max_mask_len = 10,30
        t_current = x_aug.shape[-1]
        mask_len = min(np.random.randint(min_mask_len, max_mask_len + 1), t_current)
        if t_current > mask_len:
            start = np.random.randint(0, t_current - mask_len + 1)
            x_aug[..., start:start + mask_len] = 0

    return x_aug


def eeg_local_aug(
    x: np.ndarray, noise_std=0.09, jitter_prob=0.5, scaling_prob=0.5,
    mask_prob=0.2, min_len=165, max_len=200, crop_prob=1.0,
    ft_surrogate_prob=0.7, random_seed=None
):
    """
    EEG sample augmentation with cropping, noise, masking, and FT surrogate.
    Local crops should have smaller time window
    Input:
        x: np.ndarray, shape (n_channels, t)
    Returns:
        Augmented np.ndarray
    """

    np.random.seed(np.random.randint(0,500)) # make the random actually random, else seed will fix it

    n_channels, t = x.shape
    x_aug = np.copy(x)

    # 1. Random Temporal Crop
    crop_len = np.random.randint(min_len, max_len + 1)
    if crop_prob >= 1.0 or np.random.rand() < crop_prob:
        if t > crop_len:
            start = np.random.randint(0, t - crop_len + 1)
            x_aug = x_aug[:, start:start + crop_len]
        else:
            crop_len = t  # fallback, no crop

    # 2. Additive Gaussian Noise
    if np.random.rand() < 0.7:
        x_aug = x_aug + np.random.randn(*x_aug.shape) * noise_std

    # 3. FT Surrogate
    if np.random.rand() < ft_surrogate_prob:
        x_aug = ft_surrogate_eeg(x_aug)

    # 4. Random Temporal Masking
    if np.random.rand() < mask_prob:
        min_mask_len, max_mask_len = 10,30
        t_current = x_aug.shape[-1]
        mask_len = min(np.random.randint(min_mask_len, max_mask_len + 1), t_current)
        if t_current > mask_len:
            start = np.random.randint(0, t_current - mask_len + 1)
            x_aug[..., start:start + mask_len] = 0

    return x_aug


class DINOV2EEGDataset(Dataset):
    def __init__(
        self, 
        args: TrainConfig,
        subject_ids=[1],             # List of subject numbers, e.g. [1,2,3]
        session_ids=[0],             # List of session indices (0-3)
        subset="train",
        n_global_crops=1,
        n_local_crops=7,
        transform_global=None,   # DINO global aug function
        transform_local=None,    # DINO local aug function
        keep_dim_after_mean=False,
        mean_eeg_data=False,
        seed=42,
        max_cache_size=2,  # <- How many subject files to keep in RAM per worker
        sample_cross_session_subjects=False
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
        self.sample_cross_session_subjects = sample_cross_session_subjects

        # --- Load image metadata & features (shared for all subjects) ---
        img_meta_dir = os.path.join(args.DATA_BASE_DIR, "Data", "Things-EEG2", 'Image_set')
        img_metadata = np.load(os.path.join(img_meta_dir, 'image_metadata.npy'), allow_pickle=True).item()

        data_key = "train" if subset in {"train", "val"} else "test"
        self.img_file_names = img_metadata[f'{data_key}_img_files']
        self.img_concepts = img_metadata[f'{data_key}_img_concepts']
        self.class_to_id = {c: i for i, c in enumerate(sorted(set(self.img_concepts)))}

        if self.subset == "train" or self.subset == "val":
            img_feature_path = os.path.join(args.IMG_DATA_PATH, self.args.dnn + '_feature_maps_training.npy')
        else:
            img_feature_path = os.path.join(args.IMG_DATA_PATH, self.args.dnn + '_feature_maps_test.npy')
        self.img_features = np.load(img_feature_path, allow_pickle=True).squeeze()

        # --- Build master index: (subject, class_idx, session_idx) for random access ---
        self.master_index = []
        self.global_class_wise_indexes = {}
        for subject_idx, subject_id in enumerate(subject_ids):
            eegfilekey = "training" if data_key == "train" else data_key
            eeg_path = os.path.join(self.args.EEG_DATA_PATH, f"sub-{subject_id:02d}", f"preprocessed_eeg_{eegfilekey}.npy")
            eeg_dict = np.load(eeg_path, allow_pickle=True)
            eeg_data = eeg_dict['preprocessed_eeg_data'] # (1654, 4, 63, 250)
            num_classes, num_sessions = eeg_data.shape[:2]

            for img_feat_idx in range(num_classes):  # total 16540 images
                image_class_idx = img_feat_idx // 10  # 0-based class index


                if image_class_idx not in self.global_class_wise_indexes.keys():
                    self.global_class_wise_indexes[image_class_idx] = []

                for sess in session_ids:
                    self.master_index.append({
                        "subject_idx": subject_idx,
                        "subject_id": subject_id,
                        "class_idx": image_class_idx,   # classes from 0 to 1653
                        "session_idx": sess,
                        "sample_idx_in_file": img_feat_idx,   # actual feature index in file
                        "image_file_name": self.img_file_names[img_feat_idx],
                        "image_concept_name": self.img_concepts[img_feat_idx]
                    })

                    current_master_index = len(self.master_index)-1

                    self.global_class_wise_indexes[image_class_idx].append(current_master_index)


            # for class_idx in range(num_classes):
            #     for sess in session_ids:
            #         self.master_index.append({
            #             "subject_idx": subject_idx,
            #             "subject_id": subject_id,
            #             "class_idx": class_idx,
            #             "session_idx": sess,
            #             "sample_idx_in_file": class_idx  # as class_idx==sample_idx in file
            #         })
            #         current_master_index = len(self.master_index)
            #         if class_idx not in self.global_class_wise_indexes.keys():
            #             self.global_class_wise_indexes[class_idx] = []
            #         self.global_class_wise_indexes[class_idx].append(current_master_index-1)

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
        eeg_path = os.path.join(self.args.EEG_DATA_PATH, f"sub-{subject_id:02d}", f"preprocessed_eeg_{eegfilekey}.npy")
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

        # ========== SSL Augmentations ==========
        crops = []
        # crops_img_feats = []


        sample_info = self.master_index[idx]
        subject_id = sample_info["subject_id"]
        sample_idx_in_file = sample_info["sample_idx_in_file"]  # 0-1654
        session_idx = sample_info["session_idx"]
        class_idx = sample_info["class_idx"]   # classes 0,1,2

        # Efficient subject file caching:
        eeg_data = self._get_subject_data(subject_id)
        eeg_sample = eeg_data[sample_idx_in_file, session_idx, :, :]

        # Get image features and class id
        image_features = self.img_features[sample_idx_in_file]
        # crops_img_feats.append(image_features)
        class_id = self.class_to_id[self.img_concepts[sample_idx_in_file]]

        crops.append(eeg_sample)

        if self.sample_cross_session_subjects:

            # ---- Now get Same-class Augmentations (across subject/session) ----
            same_class_global_sample_indexes = copy.deepcopy(self.global_class_wise_indexes[class_idx]) # contains global indexes of different subjects and sessions
            random.shuffle(same_class_global_sample_indexes)
            # num_augs = min(self.n_local_crops+self.n_global_crops-1, len(same_class_global_sample_indexes))
            # print(num_augs, len(candidates))
            # if idx in same_class_global_sample_indexes[:num_augs]:
                # num_augs +=1

            ## For Auto Encoder
            for i in range(2):
                
                current_g_index = same_class_global_sample_indexes[i]
                if current_g_index==idx:
                    continue # in case of same sample
                
                master_sample_info_aug = self.master_index[same_class_global_sample_indexes[i]]
                aug_subject_id = master_sample_info_aug["subject_id"]
                aug_class_idx = master_sample_info_aug["class_idx"]
                aug_session_idx = master_sample_info_aug["session_idx"]
                aug_eeg = self._get_subject_data(aug_subject_id)[aug_class_idx, aug_session_idx, :, :]
                crops.append(aug_eeg)
                break

            # # add original sample from the current index
            # crops.append(eeg_sample)

            # for i in range(self.n_local_crops + self.n_global_crops):
                
            #     current_g_index = same_class_global_sample_indexes[i]
            #     if current_g_index==idx:
            #         continue

            #     # master_sample_info_aug = self.master_index[same_class_global_sample_indexes[i]]
            #     # aug_subject_id = master_sample_info_aug["subject_id"]
            #     # aug_class_idx = master_sample_info_aug["class_idx"]
            #     # aug_session_idx = master_sample_info_aug["session_idx"]
            #     # aug_eeg = self._get_subject_data(aug_subject_id)[aug_class_idx, aug_session_idx, :, :]
            
            #     # You can use transform_global or a dedicated transform for these:
            #     # aug_crop = torch.tensor(aug_eeg, dtype=torch.float32) 

            #     if len(crops)<self.n_global_crops:
            #         crops.append(eeg_global_aug(eeg_sample))
            #     else:
            #         # TODO get from different session/subject? 

            #         crops.append(eeg_local_aug(eeg_sample))

            #     # aug_image_features = self.img_features[aug_class_idx]
            #     # crops.append(aug_eeg)
            #     # crops_img_feats.append(aug_image_features) # note, this could be same image feature of the main idx since same image is shown to different subjects

            # # # ---- Now get Same-class Augmentations (across subject/session) ----
            # # # same_class_augs = []

            # # List of (subject_id, session_idx) for the class
            # candidates = self.class_sample_lookup[class_idx].copy()

            # # Get one sample from same candidate 
            # same_candidate_samples = [(sid, sess) for (sid, sess) in candidates if (sid == subject_id and sess != session_idx)]

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
            "eeg": crops[0] if len(crops)==1 else crops,
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
        img_metadata = np.load(os.path.join(args.IMG_DATA_PATH, 'image_metadata.npy'), allow_pickle=True).item()

        data_key = "train" if subset in {"train", "val"} else "test"
        self.img_file_names = img_metadata[f'{data_key}_img_files']
        self.img_concepts = img_metadata[f'{data_key}_img_concepts']
        self.class_to_id = {c: i for i, c in enumerate(sorted(set(self.img_concepts)))}

        if self.subset == "train" or self.subset == "val":
            img_feature_path = os.path.join(args.TEST_CENTER_PATH, self.args.dnn + '_feature_maps_training.npy')
        else:
            img_feature_path = os.path.join(args.TEST_CENTER_PATH, self.args.dnn + '_feature_maps_test.npy')
        self.img_features = np.load(img_feature_path, allow_pickle=True).squeeze()

        # --- Build master index: (subject, class_idx, session_idx) for random access ---
        self.master_index = []
        for subject_idx, subject_id in enumerate(subject_ids):
            eegfilekey = "training" if data_key == "train" else data_key
            eeg_path = os.path.join(self.args.EEG_DATA_PATH, f"sub-{subject_id:02d}", f"preprocessed_eeg_{eegfilekey}.npy")
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
        eeg_path = os.path.join(self.args.EEG_DATA_PATH, f"sub-{subject_id:02d}", f"preprocessed_eeg_{eegfilekey}.npy")
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
        # crops.append(torch.tensor(eeg_sample, dtype=torch.float32))
        crops.append(eeg_global_aug(eeg_sample))


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