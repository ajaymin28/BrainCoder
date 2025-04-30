import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import copy # Import copy for deepcopy
import gc # Import gc for garbage collection

# Assuming these paths are defined elsewhere in your project
# from utils.eeg_utils import EEG_DATA_PATH, IMG_DATA_PATH, TEST_CENTER_PATH
# For this standalone code block, let's define placeholders if they aren't available
from utils.common import TrainConfig, memory_stats
try:
    from utils.eeg_utils import EEG_DATA_PATH, IMG_DATA_PATH, TEST_CENTER_PATH
except ImportError:
    print("Warning: utils.eeg_utils not found. Using placeholder paths.")
    EEG_DATA_PATH = "/path/to/your/eeg_data"
    IMG_DATA_PATH = "/path/to/your/img_data"
    TEST_CENTER_PATH = "/path/to/your/test_center"


class EEG_Dataset3(Dataset):
    """
    Code used from : # https://github.com/eeyhsong/NICE-EEG/blob/main/nice_stand.py#L194
    Modified to load data from multiple subjects using numpy.memmap
    and provide samples from different subjects within a batch.
    """

    def __init__(self, args:TrainConfig, nsubs=[1], subset="train",
                 data_root="/home/jbhol/EEG/gits/NICE-EEG/Data",
                 agument_data=True, # This might need re-evaluation in multi-subject context
                 load_individual_files=False, # This mode is not fully supported with multi-subject memmap here
                 preTraning=False,
                 include_neg_sample=False, # This might need re-evaluation in multi-subject context
                 cache_data=False, # Use cache_data flag to enable memmapping
                 mean_eeg_data=False, # This should likely be False for VAE input [batch, sessions, channels, time]
                 keep_dim_after_mean=False, # This should likely be False for VAE input
                 saved_data_path=None, # Not directly used for loading main EEG files in memmap mode
                 eeg_data_shape=(4, 63, 250), # Expected shape of EEG data per sample (sessions, channels, time)
                 eeg_data_dtype=np.float32 # Expected dtype of EEG data
                 ):
        assert subset=="train" or subset=="test" or subset=="val"
        assert isinstance(nsubs, list) and len(nsubs) > 0, "nsubs must be a list of subject IDs"

        self.args = args # Store args for access to dnn type etc.
        self.nsubs = nsubs # List of subject IDs to include
        self.subset = subset
        self.data_root = data_root
        self.agument_data = agument_data # Augmentation flag
        self.preTraning = preTraning
        self.include_neg_sample = include_neg_sample # Negative sample flag
        self.cache_data = cache_data # Flag to use memmap
        # constrastive_subject is less relevant now, contrastive pairs can be formed within batch
        # self.constrastive_subject = constrastive_subject # Kept for potential compatibility, but usage changes
        self.mean_eeg_data = mean_eeg_data
        self.keep_dim_after_mean = keep_dim_after_mean

        # load_individual_files mode is not integrated with multi-subject memmap
        if load_individual_files:
             raise NotImplementedError("load_individual_files mode is not supported with multi-subject memmap in this version.")
        self.load_individual_files = False # Force False


        data_key = "test"
        if self.subset=="val" or self.subset=="train":
            data_key = "train"

        self.eeg_data_path = EEG_DATA_PATH
        self.img_data_path = IMG_DATA_PATH
        self.test_center_path = TEST_CENTER_PATH

        ## Image Data (Loaded once, shared across subjects)
        self.img_parent_dir  = os.path.join(args.global_config.DATA_BASE_DIR, "Data", "Things-EEG2", 'Image_set')
        self.img_metadata = np.load(os.path.join(self.img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()
        # self.img_metadata = np.load(os.path.join(args.global_config.IMG_DATA_PATH, 'image_metadata.npy'), allow_pickle=True).item()

        self.img_file_names_all = self.img_metadata[f'{data_key}_img_files']
        self.img_concepts_all = self.img_metadata[f'{data_key}_img_concepts']

        # Determine the number of samples per image concept across all subjects
        # Assuming each subject has data for the same set of images
        num_samples_per_image = len(self.img_file_names_all)

        # Shuffle indices for train and val (applies to image features and EEG data mapping)
        # This shuffle is global across all samples for all subjects
        self.global_index_shuffle = list(range(num_samples_per_image))
        random.seed(2025) # Use a fixed seed for reproducibility of shuffle
        random.shuffle(self.global_index_shuffle)


        # --- Load EEG Data for Multiple Subjects using memmap or standard load ---
        self.subject_data = {} # Dictionary to store data (memmap or array) for each subject
        self.subject_img_features = {} # Dictionary to store image features for each subject
        self.subject_labels = {} # Dictionary to store labels for each subject
        self.subject_sample_indices = {} # Dictionary to store original indices for each subject after shuffle/split

        self.global_sample_map = [] # List of tuples: (subject_id, index_within_subject_subset)

        for sub_id in self.nsubs:
            eegfilekey = data_key
            if eegfilekey=="train":
                eegfilekey = "training"
            print(f"Loading data for subject {sub_id}...")
            eeg_file_path = os.path.join(args.global_config.EEG_DATA_PATH, 'sub-'+ format(sub_id, '02') , f'preprocessed_eeg_{eegfilekey}.npy')
            eeg_only_file_path = os.path.join(args.global_config.EEG_DATA_PATH, 'sub-'+ format(sub_id, '02') , f'preprocessed_eeg_only_{eegfilekey}.npy')

            if not os.path.exists(eeg_file_path):
                print(f"Warning: EEG data file not found for subject {sub_id}: {eeg_file_path}. Skipping subject.")
                continue # Skip this subject if data file is missing

            # Load the data dictionary once to inspect shape and dtype
            try:
                loaded_dict = np.load(eeg_file_path, allow_pickle=True) # Load dictionary
                raw_eeg_array = loaded_dict['preprocessed_eeg_data']
                full_data_shape = raw_eeg_array.shape # Expected shape (num_total_samples, sessions, channels, time)
                full_data_dtype = raw_eeg_array.dtype
                if not os.path.exists(eeg_only_file_path):
                    np.save(eeg_only_file_path, raw_eeg_array)
                del loaded_dict, raw_eeg_array # Free memory
            except Exception as e:
                 print(f"Warning: Could not load EEG data dictionary from {eeg_file_path} for subject {sub_id}: {e}. Skipping subject.")
                 continue # Skip this subject if loading fails


            subject_eeg_data = None
            if self.cache_data: # Use memmap
                print(f"Using memmap for subject {sub_id} data from {eeg_only_file_path}")
                try:
                    # Attempting memmap directly on the .npy file
                    subject_eeg_data = np.memmap(eeg_only_file_path, dtype=full_data_dtype, mode='r', shape=full_data_shape)
                    print(f"Memmap successful for subject {sub_id}.")
                except Exception as e:
                     print(f"Warning: Could not memmap {eeg_only_file_path} directly for subject {sub_id}. Falling back to standard load. Error: {e}")
                     self.cache_data = False # Disable memmap for this subject if it fails


            if not self.cache_data or subject_eeg_data is None: # Standard load into memory if memmap is not used or failed
                print(f"Loading subject {sub_id} data into memory from {eeg_file_path}")
                try:
                    loaded_dict = np.load(eeg_file_path, allow_pickle=True)
                    subject_eeg_data = loaded_dict['preprocessed_eeg_data']
                    del loaded_dict # Free memory
                    print(f"Standard load successful for subject {sub_id}.")
                except Exception as e:
                     print(f"Error: Could not perform standard load for EEG data from {eeg_file_path} for subject {sub_id}: {e}. Skipping subject.")
                     continue # Skip this subject if standard loading fails


            # Load image features (same for all subjects, but we'll store a reference per subject for consistency)
            if subset=="train" or subset=="val":
                img_feature_path = os.path.join(args.global_config.IMG_DATA_PATH, self.args.dnn + '_feature_maps_training.npy')
            else:
                img_feature_path = os.path.join(args.global_config.IMG_DATA_PATH, self.args.dnn + '_feature_maps_test.npy')

            subject_img_features_all = np.load(img_feature_path, allow_pickle=True)
            subject_img_features_all = np.squeeze(subject_img_features_all) # (num_total_samples, feature_dim)


            # Apply the global shuffle to the loaded data and image features for this subject
            subject_eeg_data_shuffled = [subject_eeg_data[i] for i in self.global_index_shuffle]
            subject_img_features_shuffled = [subject_img_features_all[i] for i in self.global_index_shuffle]
            subject_img_concepts_shuffled = [self.img_concepts_all[i] for i in self.global_index_shuffle]

            del subject_eeg_data, subject_img_features_all # Free memory from original loads/memmaps before subsetting


            # Apply subset slicing (train/val split) after global shuffling
            if not self.preTraning:
                if subset=="val":
                    subject_eeg_data_subset = subject_eeg_data_shuffled[:740]
                    subject_img_features_subset = subject_img_features_shuffled[:740]
                    subject_img_concepts_subset = subject_img_concepts_shuffled[:740]
                elif subset=="train":
                    subject_eeg_data_subset = subject_eeg_data_shuffled[740:]
                    subject_img_features_subset = subject_img_features_shuffled[740:]
                    subject_img_concepts_subset = subject_img_concepts_shuffled[740:]
                else: # Test set
                    subject_eeg_data_subset = subject_eeg_data_shuffled
                    subject_img_features_subset = subject_img_features_shuffled
                    subject_img_concepts_subset = subject_img_concepts_shuffled
            else: # Pretraining uses all data
                 subject_eeg_data_subset = subject_eeg_data_shuffled
                 subject_img_features_subset = subject_img_features_shuffled
                 subject_img_concepts_subset = subject_img_concepts_shuffled


            # Store the subset data for this subject
            self.subject_data[sub_id] = subject_eeg_data_subset
            self.subject_img_features[sub_id] = subject_img_features_subset
            self.subject_labels[sub_id] = [] # Store labels for this subject's subset
            self.subject_sample_indices[sub_id] = list(range(len(subject_eeg_data_subset))) # Indices within this subject's subset

            # Build labels and class-wise data for this subject's subset
            subject_class_to_id = {}
            subject_id_to_class = {}
            subject_class_wise_data = {}

            for i in range(len(subject_img_concepts_subset)):
                cls = subject_img_concepts_subset[i]
                if cls not in subject_class_to_id.keys():
                    subject_class_to_id[cls] = len(subject_class_to_id.keys())
                    subject_id_to_class[len(subject_id_to_class.keys())] = cls
                self.subject_labels[sub_id].append(cls)

            for cls_idx, cls_label in enumerate(self.subject_labels[sub_id]):
                if cls_label not in subject_class_wise_data.keys():
                    subject_class_wise_data[cls_label] = []
                subject_class_wise_data[cls_label].append(cls_idx)

            # Store subject-specific class mappings and class-wise indices
            # This might be complex if classes vary between subjects.
            # Assuming classes are consistent, we can use a global mapping later.
            # For now, just store the class-wise data for potential negative sampling within a subject.
            self.subject_class_wise_data = subject_class_wise_data # Overwriting for simplicity, consider merging

            # Append to the global sample map
            for i in range(len(subject_eeg_data_subset)):
                self.global_sample_map.append((sub_id, i)) # (subject_id, index_within_subset)


            gc.collect() # Explicit garbage collection after processing each subject
            memory_stats()


        # Rebuild global labels and class mappings based on all included subjects
        self.global_labels = []
        self.global_class_to_id = {}
        self.global_id_to_class = {}
        self.global_class_wise_data = {} # Class-wise indices across all subjects

        for sub_id, sample_idx_in_subset in self.global_sample_map:
             img_concept = self.subject_labels[sub_id][sample_idx_in_subset] # Get the original concept string
             self.global_labels.append(img_concept)

             if img_concept not in self.global_class_to_id.keys():
                 self.global_class_to_id[img_concept] = len(self.global_class_to_id.keys())
                 self.global_id_to_class[len(self.global_id_to_class.keys())] = img_concept

             global_index = len(self.global_labels) - 1 # The current global index
             if img_concept not in self.global_class_wise_data.keys():
                 self.global_class_wise_data[img_concept] = []
             self.global_class_wise_data[img_concept].append(global_index)


        print(f"Dataset init done for subjects : {self.nsubs} subset: {subset}")
        print("Total number of samples across all subjects:", len(self.global_sample_map))


    def __len__(self):
        return len(self.global_sample_map) # Total number of samples across all subjects

    def getLabel(self, global_index):
        """
        Get the class label (name and ID) for a sample based on its global index.
        """
        cls_label_name = self.global_labels[global_index]
        cls_label_id = self.global_class_to_id[cls_label_name]
        return cls_label_name, cls_label_id


    def __getitem__(self, global_index):
        """
        Retrieves a sample based on its global index.
        Returns data for the primary sample, and optionally for augmented and negative samples.
        """
        # Get the subject ID and the index within that subject's subset
        sub_id, sample_idx_in_subset = self.global_sample_map[global_index]

        # Retrieve the primary EEG data and image features
        eeg_feat = self.subject_data[sub_id][sample_idx_in_subset] # Shape: (sessions, channels, time)

        # Apply mean over sessions if mean_eeg_data is True
        if self.mean_eeg_data:
             eeg_feat = np.mean(eeg_feat, axis=0, keepdims=self.keep_dim_after_mean) # Shape depends on keepdims

        img_feat = self.subject_img_features[sub_id][sample_idx_in_subset] # Shape: (feature_dim,)

        # Get the class label and subject ID for the primary sample
        cls_label_name = self.subject_labels[sub_id][sample_idx_in_subset]
        cls_label_id = self.global_class_to_id[cls_label_name] # Use global class ID
        primary_sub_id = sub_id - 1 # Subject ID of the primary sample

        # Prepare the primary data tuple (data1)
        data1 = (eeg_feat, img_feat, cls_label_id, primary_sub_id)

        # --- Handle Augmented Data (Contrastive Sample) ---
        # In a multi-subject setting, a contrastive sample could be:
        # 1. EEG from a *different* subject for the *same* image.
        # 2. EEG from the *same* subject for a *different* (but related?) image.
        # The original code used EEG from a 'constrastive_subject' for the same image.
        # Let's adapt this: find a sample with the same image concept from a *different* subject.

        data2 = ([], [], [], []) # Default empty tuple for data2
        if self.agument_data:
            # Find other samples with the same image concept across all subjects
            same_concept_global_indices = copy.deepcopy(self.global_class_wise_data.get(cls_label_name, []))

            # Remove the current global index to ensure we pick a different sample
            if global_index in same_concept_global_indices:
                 same_concept_global_indices.pop(same_concept_global_indices.index(global_index))

            # Filter to find samples from a *different* subject than the primary one
            different_subject_same_concept_indices = [
                idx for idx in same_concept_global_indices
                if self.global_sample_map[idx][0] != primary_sub_id # Check if subject ID is different
            ]

            if different_subject_same_concept_indices:
                # Randomly sample one index from a different subject with the same concept
                contrastive_global_index = random.sample(different_subject_same_concept_indices, 1)[0]

                # Get the subject ID and index within subset for the contrastive sample
                contrastive_sub_id, contrastive_sample_idx_in_subset = self.global_sample_map[contrastive_global_index]
                

                # Retrieve contrastive EEG data
                eeg_feat2 = self.subject_data[contrastive_sub_id][contrastive_sample_idx_in_subset]
                if self.mean_eeg_data:
                     eeg_feat2 = np.mean(eeg_feat2, axis=0, keepdims=self.keep_dim_after_mean)

                contrastive_sub_id = contrastive_sub_id - 1
                # The image feature and class label are the same as the primary sample
                data2 = (eeg_feat2, img_feat, cls_label_id, contrastive_sub_id)


        # --- Handle Negative Sample ---
        # In a multi-subject setting, a negative sample could be:
        # 1. EEG from the same subject, different class image. (Original logic)
        # 2. EEG from a different subject, different class image.
        # 3. EEG from a different subject, same class image (could be hard negative).
        # Let's adapt the original logic: EEG from the same subject, different class image.

        data3 = ([], [], [], []) # Default empty tuple for data3
        if self.include_neg_sample:
            # Find samples from the *same* primary subject but a *different* class
            same_subject_indices = [
                idx for idx, (s_id, _) in enumerate(self.global_sample_map)
                if s_id == primary_sub_id
            ]

            # Find samples from the same subject but a different class than the primary sample
            neg_sample_global_indices = [
                idx for idx in same_subject_indices
                if self.global_labels[idx] != cls_label_name # Check if class label is different
            ]

            if neg_sample_global_indices:
                # Randomly sample one index from the same subject with a different concept
                neg_global_index = random.sample(neg_sample_global_indices, 1)[0]

                # Get the subject ID and index within subset for the negative sample (should be primary_sub_id)
                neg_sub_id, neg_sample_idx_in_subset = self.global_sample_map[neg_global_index]
                assert neg_sub_id == primary_sub_id # Sanity check

                # Retrieve negative EEG data
                neg_eeg_feat = self.subject_data[neg_sub_id][neg_sample_idx_in_subset]
                if self.mean_eeg_data:
                     neg_eeg_feat = np.mean(neg_eeg_feat, axis=0, keepdims=self.keep_dim_after_mean)

                # Get negative image feature and label
                neg_img_feat = self.subject_img_features[neg_sub_id][neg_sample_idx_in_subset]
                neg_cls_label_name = self.subject_labels[neg_sub_id][neg_sample_idx_in_subset]
                neg_cls_label_id = self.global_class_to_id[neg_cls_label_name] # Use global class ID

                neg_sub_id = neg_sub_id - 1
                data3 = (neg_eeg_feat, neg_img_feat, neg_cls_label_id, neg_sub_id)


        return data1, data2, data3

    def close_memmap(self):
        """
        Explicitly close the memmap files if they were opened.
        This is important to release file handles, especially on Windows.
        Call this when the dataset object is no longer needed.
        """
        if self.cache_data:
            for sub_id, data_source in self.subject_data.items():
                if isinstance(data_source, np.memmap):
                    try:
                        data_source._mmap.close()
                        print(f"Closed memmap for subject {sub_id}")
                    except AttributeError:
                         # Handle cases where slicing returned a non-memmap object
                         pass
            # No need to close c_data separately as it's now part of subject_data


# Example usage in your main script:
# Assuming TrainConfig is defined and paths are set
# t_config = TrainConfig()
# t_config.channels = 63
# t_config.time_points = 250
# t_config.sessions = 4
# t_config.batch_size = 32
# t_config.MultiSubject = True # Keep this flag
# t_config.TestSubject = 1
# t_config.Contrastive_augmentation = True
# t_config.mean_eeg_data = False # Important for VAE input shape
# t_config.keep_dim_after_mean = False # Important for VAE input shape
# t_config.cache_data = True # Enable memmapping

# In your multi-subject training loop:
# # Define the list of training subject IDs
# total_subjects_list = [i for i in range(1, t_config.num_subjects + 1)]
# if t_config.TestSubject in total_subjects_list:
#      total_subjects_list.pop(total_subjects_list.index(t_config.TestSubject))
# print("Subjects to be trained:", total_subjects_list)

# # Create a single dataset instance including all training subjects
# # Pass the list of subject IDs to nsubs
# dataset = EEG_Dataset3(
#     args=t_config,
#     nsubs=total_subjects_list, # Pass the list of subjects
#     subset="train",
#     agument_data=t_config.Contrastive_augmentation,
#     cache_data=t_config.cache_data,
#     mean_eeg_data=t_config.mean_eeg_data,
#     keep_dim_after_mean=t_config.keep_dim_after_mean,
#     saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata", # Ensure this path is correct
#     eeg_data_shape=(t_config.sessions, t_config.channels, t_config.time_points),
#     eeg_data_dtype=np.float32
# )

# # Create validation dataset similarly (for the test subject or a subset of training subjects)
# # If validating on the test subject, create a separate dataset instance for that subject.
# # If validating on a subset of training subjects, pass that subset list to nsubs.
# val_subjects_list = [t_config.TestSubject] # Example: validate on the test subject
# val_dataset = EEG_Dataset2(
#     args=t_config,
#     nsubs=val_subjects_list, # Pass the list of validation subjects
#     subset="val", # Or "test" if using test set
#     agument_data=False, # Usually no augmentation in validation
#     cache_data=t_config.cache_data,
#     mean_eeg_data=t_config.mean_eeg_data,
#     keep_dim_after_mean=t_config.keep_dim_after_mean,
#     saved_data_path="/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/mydata", # Ensure this path is correct
#     eeg_data_shape=(t_config.sessions, t_config.channels, t_config.time_points),
#     eeg_data_dtype=np.float32
# )


# # Create DataLoaders
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=t_config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=t_config.batch_size, shuffle=False, num_workers=0, pin_memory=True) # Shuffle False for consistent validation


# # Now, the training loop will iterate through batches from the combined dataloader.
# # Each batch will likely contain samples from different subjects due to shuffling.
# # The process_batch function will receive batches with mixed subject data.
# # The subject ID for each sample is included in the data tuples returned by __getitem__.

# # The train function will be called once per global epoch, not per subject pair.
# # The inner subject loop in the main script is no longer needed for data loading/training.
# # The adversarial loss and contrastive loss in process_batch need to correctly handle
# # batches with mixed subject IDs. The current vae_loss function already takes subject_labels,
# # which is good. The contrastive loss calculates logits between projected_z and image_features
# # within the batch, which should still work for positive pairs (same image, potentially different subjects).

# # After training is complete (all global epochs), close memmap files
# # del dataloader, val_dataloader # Delete dataloaders first
# # dataset.close_memmap()
# # val_dataset.close_memmap()
# # del dataset, val_dataset
# # gc.collect()
# # torch.cuda.empty_cache()

