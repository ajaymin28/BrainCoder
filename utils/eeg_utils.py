from torch.utils.data import Dataset
import random
# from copy import copy
# import matplotlib.pyplot as plt
# from PIL import Image
import numpy as np
import os
import gc
import pickle
from tqdm import tqdm
import copy

EEG_DATA_PATH = '/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz'
IMG_DATA_PATH = '/home/jbhol/EEG/gits/NICE-EEG/dnn_feature'
TEST_CENTER_PATH = '/home/jbhol/EEG/gits/NICE-EEG/dnn_feature'




def get_eeg_data(eeg_data_path, nSub, subset=None):
    if subset is not None:
        assert subset=="train" or subset=="test"

    def load_data(path, subj, train=True):
        file_name = "preprocessed_eeg_training.npy"
        if not train:
            file_name = "preprocessed_eeg_test.npy"
        data = np.load(path + '/sub-' + format(subj, '02') + f'/{file_name}', allow_pickle=True)
        # train_data_original = copy.deepcopy(train_data['preprocessed_eeg_data'])
        data = data['preprocessed_eeg_data']
        data = np.mean(data, axis=1)
        data = np.expand_dims(data, axis=1)
        return data

    train_data = []
    train_label = []
    test_data = []
    test_label = np.arange(200)

    if subset is None:
        test_data = load_data(path=eeg_data_path,subj=nSub,train=False)
        train_data = load_data(path=eeg_data_path,subj=nSub,train=True)
    elif subset=="train":
        train_data = load_data(path=eeg_data_path,subj=nSub,train=True)
    elif subset=="test":
        test_data = load_data(path=eeg_data_path,subj=nSub,train=False)

    return train_data, train_label, test_data, test_label

class EEG_Dataset(Dataset):
    """
    Code used from : # https://github.com/eeyhsong/NICE-EEG/blob/main/nice_stand.py#L194
    """

    def __init__(self, args, nsub=1, subset="train", 
                 data_root="/home/jbhol/EEG/gits/NICE-EEG/Data", 
                 agument_data=True,
                 load_individual_files=False,
                 saved_data_path=None):
        assert subset=="train" or subset=="test" or subset=="val"

        if self.load_individual_files:
            assert saved_data_path!=None

        self.subset = subset
        self.data_root = data_root
        self.agument_data = agument_data
        self.load_individual_files = load_individual_files

        self.eeg_data_path = '/home/jbhol/EEG/gits/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz'
        self.img_data_path = '/home/jbhol/EEG/gits/NICE-EEG/dnn_feature/'
        self.test_center_path = '/home/jbhol/EEG/gits/NICE-EEG/dnn_feature/'
        self.nSub = nsub

        if not self.load_individual_files:
            if subset=="train" or subset=="val":
                self.data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
                self.data = self.data['preprocessed_eeg_data']
            else:
                self.data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
                self.data = self.data['preprocessed_eeg_data']

        # print("loaded eeg data")
        gc.collect()

        self.shuffle_var = np.random.permutation(len(self.data)) # to be used later to shuffle for train and val spit
        
        ## Image Data
        self.img_parent_dir  = os.path.join(self.data_root, "Things-EEG2", 'Image_set')
        self.img_metadata = np.load(os.path.join(self.img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()

        # print("loaded metadata")

        # train_img_concepts = np.array(copy(self.img_metadata['train_img_concepts']))  # Convert to numpy array
        # shuffled_train_img_concepts = train_img_concepts[self.shuffle_var]  # Shuffle using indices
        # self.img_metadata['train_img_concepts_shuffled'] = shuffled_train_img_concepts.tolist()  # Convert back to list

        # train_img_files = np.array(copy(self.img_metadata['train_img_files']))  # Convert to numpy array
        # shuffled_train_img_files = train_img_files[self.shuffle_var]  # Shuffle using indices
        # self.img_metadata['train_img_files_shuffled'] = shuffled_train_img_files.tolist()  # Convert back to list

        self.data_indexes = []
        self.labels = []
        self.class_to_id = {}
        self.id_to_class = {}
        self.class_wise_data = {}

        if subset=="train" or subset=="val":
            for i in range(len(self.img_metadata['train_img_concepts'])):
                self.data_indexes.append(i)
                t_cls = self.img_metadata['train_img_concepts'][i]
                if t_cls not in self.class_to_id.keys():
                    self.class_to_id[t_cls] = len(self.class_to_id.keys())
                    self.id_to_class[len(self.id_to_class.keys())] = t_cls
                self.labels.append(t_cls)
        else:
            for i in range(len(self.img_metadata['test_img_concepts'])):
                self.data_indexes.append(i)
                test_cls = self.img_metadata['test_img_concepts'][i]
                if test_cls not in self.class_to_id.keys():
                    self.class_to_id[test_cls] = len(self.class_to_id.keys())
                    self.id_to_class[len(self.id_to_class.keys())] = test_cls
                self.labels.append(test_cls)

        if not self.load_individual_files:
            # image features
            if subset=="train" or subset=="val":
                self.img_feature = np.load(self.img_data_path + args.dnn + '_feature_maps_training.npy', allow_pickle=True) # (16540, 1, 768)
            else:
                self.img_feature = np.load(self.img_data_path + args.dnn + '_feature_maps_test.npy', allow_pickle=True)
            self.img_feature = np.squeeze(self.img_feature) # (16540, 768)


            if subset=="train" or subset=="val":
                # shuffle everything
                self.img_feature = self.img_feature[self.shuffle_var]
                # print("shuffling")

                self.data = self.data[self.shuffle_var]
                self.labels = np.array(self.labels)
                self.labels = self.labels[self.shuffle_var]
                self.labels = self.labels.tolist()

                # print("shuffle done")

                if subset=="train":
                    self.img_feature = self.img_feature[740:]
                    self.labels = self.labels[740:]
                    self.data = self.data[740:]
                elif subset=="val":
                    self.img_feature = self.img_feature[:740]
                    self.labels = self.labels[:740]
                    self.data = self.data[:740]

            # gather same class data togather to use them as augmentation like self-supervised(DINO)
            for cls_idx, cls_label in enumerate(self.labels):
                if cls_label not in self.class_wise_data.keys():
                    self.class_wise_data[cls_label] = []
                self.class_wise_data[cls_label].append(cls_idx)
            gc.collect()
            # print("init ds complete")
        else:
            self.load_saved_ds(root_path=saved_data_path)
            self.data_indexes = self.convert_and_shuffle(self.data_indexes)



    def convert_and_shuffle(self, data_list, shuffle_var, convert_data_np=False):
        if convert_data_np:
            data_list = np.array(data_list)
        data_list = data_list[shuffle_var]
        data_list = data_list.tolist()    
        return data_list
    
    def load_saved_ds(self, root_path):
        self.image_features_path = os.path.join(root_path, "image_features")
        self.eeg_features_path = os.path.join(root_path, "eeg_features")
        self.labels_path = os.path.join(root_path, "labels")
        with open(os.path.join(self.labels_path,"labels_data.pickle"), "wb") as f:
            label_data = pickle.loads(f)
            self.labels = label_data["labels"]
            self.data_indexes = []
            for i in range(len(self.labels)):
                self.data_indexes.append(i)
            self.class_wise_data = label_data["class_wise_data"]
            self.class_to_id = label_data["class_to_id"]
            self.id_to_class = label_data["id_to_class"]

    def __len__(self):
        return len(self.labels)
    
    
    def save_ds_files(self, save_path):
        image_features_path = os.path.join(save_path, "image_features") 
        eeg_features_path = os.path.join(save_path, "eeg_features") 
        labels_path = os.path.join(save_path, "labels") 
        os.makedirs(image_features_path,exist_ok=True)
        os.makedirs(eeg_features_path,exist_ok=True)
        os.makedirs(labels_path,exist_ok=True)

        self.class_to_id = {}
        self.id_to_class = {}

        label_data = {
            "labels": self.labels,
            "class_wise_data": self.class_wise_data,
            "class_to_id": self.class_to_id,
            "id_to_class": self.id_to_class
        }

        with open(os.path.join(labels_path,"labels_data.pickle"), "wb") as f:
            pickle.dump(label_data,f)

        for index in range(len(self.data)):
            # image_features = self.img_feature[index]
            cls_label_name = self.labels[index]
            # print(cls_label_name)
            # cls_label_id = self.class_to_id[cls_label_name]
            eeg = self.data[index]  # contains 4 sessions (1,4,63,250)
            image_features = self.img_feature[index]

            with open(os.path.join(image_features_path, f"{cls_label_name}_{index:05d}.npy"), 'wb') as f:
                np.save(f, image_features)
            
            with open(os.path.join(eeg_features_path, f"{cls_label_name}_{index:05d}.npy"), 'wb') as f:
                np.save(f, eeg)



    
    def __getitem__(self, index):
        #read image
        # img_dir = os.path.join(img_parent_dir, 'training_images', img_metadata[f'{self.subset}_img_concepts'][index], img_metadata[f'{self.subset}_img_files'][index])
        # img = Image.open(train_img_dir).convert('RGB')

        # image_features = self.img_feature[index]
        cls_label_name = self.labels[index]
        # print(cls_label_name)
        cls_label_id = self.class_to_id[cls_label_name]
        eeg = self.data[index]  # contains 4 sessions (1,4,63,250)
        image_features = self.img_feature[index]

        # if augment data then we have for one class 10 images(train) and 4 EEG sessions for each. we can pick any conbinations from these.
        if self.agument_data:

            class_sample_indexes = self.class_wise_data[cls_label_name]
            sampled_index = random.sample(class_sample_indexes,1)

            image_features_2 = self.img_feature[sampled_index[0]]
            # image_features_2 = []
            cls_label_name_2 = self.labels[sampled_index[0]]
            # print(cls_label_name_2)
            cls_label_id_2 = self.class_to_id[cls_label_name_2]
            eeg_2 = self.data[sampled_index[0]]  # contains 4 sessions (1,4,63,250)

            # selected_indices = np.random.choice(eeg.shape[0], size=2, replace=False) # select any 2 indexes out of 4
            # eeg =  eeg[selected_indices]

            # Randomly determine start index ensuring at least 200 samples
            # min_length = 100
            # start = np.random.randint(0, eeg_2.shape[2] - min_length + 1, size=1).item()  # Ensure valid start range
            # end = start + min_length

            # Apply selection and slicing
            # selected_indices = np.random.choice(eeg_2.shape[0], size=2, replace=False)
            # eeg_2 = eeg_2[selected_indices][:, :, start:end]

            return (eeg, image_features, cls_label_id), (eeg_2, image_features_2, cls_label_id_2)

        return eeg, image_features, cls_label_id




class EEG_Dataset2(Dataset):
    """
    Code used from : # https://github.com/eeyhsong/NICE-EEG/blob/main/nice_stand.py#L194
    """

    def __init__(self, args, nsub=1, subset="train", 
                 data_root="/home/jbhol/EEG/gits/NICE-EEG/Data", 
                 agument_data=True,
                 load_individual_files=False,
                 preTraning=False,
                 include_neg_sample=False,
                 cache_data=False,
                 constrastive_subject=0,
                 mean_eeg_data=False,
                 keep_dim_after_mean=False,
                 saved_data_path=None):
        assert subset=="train" or subset=="test" or subset=="val"

        self.nSub = nsub
        self.subset = subset
        self.data_root = data_root
        self.agument_data = agument_data
        self.preTraning = preTraning
        self.include_neg_sample = include_neg_sample
        self.cache_data = cache_data
        self.constrastive_subject = constrastive_subject
        self.preloaded_data = False
        self.mean_eeg_data = mean_eeg_data
        self.keep_dim_after_mean = keep_dim_after_mean

        self.load_individual_files = load_individual_files
        if self.load_individual_files:
            assert saved_data_path!=None
            self.saved_data_path = saved_data_path
            self.image_features_path = os.path.join(self.saved_data_path, "image_features") 
            self.eeg_features_root_path = os.path.join(self.saved_data_path , "eeg_features") 
            self.eeg_features_path = os.path.join(self.saved_data_path , "eeg_features", str(self.nSub)) 
            self.labels_path = os.path.join(self.saved_data_path, "labels") 

        data_key = "test"
        if self.subset=="val" or self.subset=="train":
            data_key = "train"

        self.eeg_data_path = EEG_DATA_PATH
        self.img_data_path = IMG_DATA_PATH
        self.test_center_path = TEST_CENTER_PATH

        ## Image Data
        self.img_parent_dir  = os.path.join(self.data_root, "Things-EEG2", 'Image_set')
        self.img_metadata = np.load(os.path.join(self.img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()

        self.img_file_names = self.img_metadata[f'{data_key}_img_files']
        self.img_concepts = self.img_metadata[f'{data_key}_img_concepts']

        #shuffle for train and val
        index_shuffle = list(range(len(self.img_file_names)))
        random.seed(2025)
        random.shuffle(index_shuffle)

        self.data_indexes = []
        self.labels = []
        self.class_to_id = {}
        self.id_to_class = {}
        self.class_wise_data = {}
    
        # image features
        if not self.load_individual_files:
            if subset=="train" or subset=="val":
                data_path = os.path.join(self.eeg_data_path, 'sub-'+ format(self.nSub, '02') , 'preprocessed_eeg_training.npy')
                self.data = np.load(data_path, allow_pickle=True)
                self.data = self.data['preprocessed_eeg_data']

                c_data_path = os.path.join(self.eeg_data_path, 'sub-'+ format(self.constrastive_subject, '02'), 'preprocessed_eeg_training.npy')
                self.c_data = np.load(c_data_path, allow_pickle=True)
                self.c_data = self.c_data['preprocessed_eeg_data']

            else:
                data_path = os.path.join(self.eeg_data_path, 'sub-'+ format(self.nSub, '02'), 'preprocessed_eeg_test.npy')
                self.data = np.load(data_path, allow_pickle=True)
                self.data = self.data['preprocessed_eeg_data']

                c_data_path = os.path.join(self.eeg_data_path, 'sub-'+ format(self.constrastive_subject, '02'), 'preprocessed_eeg_test.npy')
                self.c_data = np.load(c_data_path, allow_pickle=True)
                self.c_data = self.c_data['preprocessed_eeg_data']

            self.data = [self.data[i] for i in index_shuffle]
            self.c_data = [self.c_data[i] for i in index_shuffle]
            gc.collect()

            if subset=="train" or subset=="val":
                img_feature_path = os.path.join(self.img_data_path, args.dnn + '_feature_maps_training.npy')
                self.img_feature = np.load(img_feature_path, allow_pickle=True) # (16540, 1, 768)
            else:
                img_feature_path = os.path.join(self.img_data_path, args.dnn + '_feature_maps_test.npy')
                self.img_feature = np.load(img_feature_path, allow_pickle=True)
            self.img_feature = np.squeeze(self.img_feature) # (16540, 768)

            self.img_feature = [self.img_feature[i] for i in index_shuffle]
            gc.collect()
        

        # Reorder train_eeg and train_img_feature using the shuffled indices
        self.img_file_names = [self.img_file_names[i] for i in index_shuffle]
        self.img_concepts = [self.img_concepts[i] for i in index_shuffle]

        for i in range(len(self.img_concepts)):
            cls = self.img_concepts[i]
            if cls not in self.class_to_id.keys():
                self.class_to_id[cls] = len(self.class_to_id.keys())
                self.id_to_class[len(self.id_to_class.keys())] = cls
            self.labels.append(cls)

        if not self.preTraning:
            if subset=="val" or subset=="train":
                if subset=="val":
                    self.img_file_names  = self.img_file_names[:740]
                    self.img_concepts  = self.img_concepts[:740]
                    self.labels = self.labels[:740]
                    self.data = self.data[:740]
                    self.c_data = self.c_data[:740]
                else:
                    self.img_file_names  = self.img_file_names[740:]
                    self.img_concepts  = self.img_concepts[740:]
                    self.labels = self.labels[740:]
                    self.data = self.data[740:]
                    self.c_data = self.c_data[740:]

        for cls_idx, cls_label in enumerate(self.labels):
            if cls_label not in self.class_wise_data.keys():
                self.class_wise_data[cls_label] = []
            self.class_wise_data[cls_label].append(cls_idx)


        self.indexes_loaded = {}
        self.loaded_indexes_features = {}

        if str(self.nSub) not in self.indexes_loaded.keys():
            self.indexes_loaded[str(self.nSub)] = []

        if str(self.constrastive_subject) not in self.indexes_loaded.keys():
            self.indexes_loaded[str(self.constrastive_subject)] = []

        if str(self.nSub) not in self.loaded_indexes_features:
            self.loaded_indexes_features[str(self.nSub)] = {}

        if str(self.constrastive_subject) not in self.loaded_indexes_features:
            self.loaded_indexes_features[str(self.constrastive_subject)] = {}

        print(f"Dataset init done for subject : {self.nSub} subset: {subset}")
        print("data len", len(self.data))
        

    def __len__(self):
        return len(self.img_file_names)
    
    
    def save_ds_files(self, save_path):
        image_features_path = os.path.join(save_path, "image_features")
        eeg_features_path = os.path.join(save_path , "eeg_features", str(self.nSub)) 
        labels_path = os.path.join(save_path, "labels")
        os.makedirs(image_features_path,exist_ok=True)
        os.makedirs(eeg_features_path,exist_ok=True)
        os.makedirs(labels_path,exist_ok=True)

        for index in tqdm(range(len(self.img_file_names))):
            # image_features = self.img_feature[index]

            concept_name = self.img_concepts[index]
            concept_img_name = self.img_file_names[index]
            concept_img_name_wo_ext = concept_img_name.split(".")[0]

            img_save_path = os.path.join(image_features_path, concept_name)
            os.makedirs(img_save_path,exist_ok=True)

            eeg_save_path = os.path.join(eeg_features_path, concept_name)
            os.makedirs(eeg_save_path,exist_ok=True)

            img_feat_save_path = os.path.join(img_save_path, f"{concept_img_name_wo_ext}_img.npy")
            eeg_feat_save_path = os.path.join(eeg_save_path, f"{concept_img_name_wo_ext}_eeg.npy")

            eeg = self.data[index]  # contains 4 sessions (1,4,63,250)
            image_features = self.img_feature[index]

            with open(img_feat_save_path, 'wb') as f:
                np.save(f, image_features)
            
            with open(eeg_feat_save_path, 'wb') as f:
                np.save(f, eeg)

    
    def _load_npy(self, index, subject=None):
        img_file_name  = self.img_file_names[index]
        img_concept  = self.img_concepts[index]
        # label = self.labels[index]
        img_name_wo_ext = img_file_name.split(".")[0]

        img_feat_path = os.path.join(self.image_features_path,img_concept, f"{img_name_wo_ext}_img.npy")
        eeg_feat_path = os.path.join(self.eeg_features_path,img_concept, f"{img_name_wo_ext}_eeg.npy")
        if subject is not None and type(subject)==int:
            # load based on subject
            eeg_feat_path = os.path.join(f"{self.eeg_features_root_path}",str(subject),img_concept,f"{img_name_wo_ext}_eeg.npy")
        
        img_feat = np.load(img_feat_path, allow_pickle=True)
        eeg_feat = np.load(eeg_feat_path, allow_pickle=True)

        if self.mean_eeg_data:
            eeg_feat = np.mean(eeg_feat,axis=0,keepdims=False)

        return img_feat, eeg_feat

    
    def getLabel(self, index):
        cls_label_name_2 = self.labels[index]
        cls_label_id_2 = self.class_to_id[cls_label_name_2]
        return cls_label_name_2, cls_label_id_2

    def preload_data(self):
        for ind in tqdm(range(len(self.img_file_names))):
            img_feat, eeg_feat = self._load_npy(ind)

            cls_label_name = self.labels[ind]
            cls_label_id = self.class_to_id[cls_label_name]

            self.loaded_indexes_features[str(self.nSub)][ind] = [img_feat, eeg_feat, cls_label_id]
            self.indexes_loaded[str(self.nSub)].append(ind)
        
            if self.agument_data:
                class_sample_indexes = copy.deepcopy(self.class_wise_data[cls_label_name]) 
                if ind in class_sample_indexes:
                    index_to_del =  class_sample_indexes.index(ind)
                    class_sample_indexes.pop(index_to_del)

                sampled_index = random.sample(class_sample_indexes,1)
                # print("Index loaded", sampled_index, class_sample_indexes)

                if sampled_index[0] in self.indexes_loaded[str(self.constrastive_subject)]:
                    pass
                    # img_feat2, eeg_feat2, cls_label_id_2 = self.loaded_indexes_features[str(self.constrastive_subject)][sampled_index[0]]
                else:
                    img_feat2, eeg_feat2 = self._load_npy(sampled_index[0],subject=self.constrastive_subject)
                    cls_label_name_2,cls_label_id_2=  self.getLabel(index=sampled_index[0])

                    self.indexes_loaded[str(self.constrastive_subject)].append(sampled_index[0])
                    self.loaded_indexes_features[str(self.constrastive_subject)][sampled_index[0]] = [img_feat2, eeg_feat2, cls_label_id_2]

                
                if self.include_neg_sample:
                    # get negative pair
                    classkeys = copy.deepcopy(list(self.class_wise_data.keys())) # get all classes
                    del classkeys[classkeys.index(cls_label_name)] # drop current class

                    sampled_class = random.sample(classkeys,1) # sample one class
                    neg_class_sample_indexes = copy.deepcopy(self.class_wise_data[sampled_class[0]])
                    neg_sampled_index = random.sample(neg_class_sample_indexes,1)

                    if neg_sampled_index[0] in self.indexes_loaded:
                        pass
                        # neg_img_feat, neg_eeg_feat, neg_cls_label_id = self.loaded_indexes_features[str(self.nSub)][neg_sampled_index[0]]
                    else:
                        neg_img_feat, neg_eeg_feat = self._load_npy(neg_sampled_index[0])
                        if self.cache_data:
                            if "neg" not in self.loaded_indexes_features[str(self.nSub)].keys():
                                self.loaded_indexes_features[str(self.nSub)]["neg"] = {}

                            if "neg" not in self.indexes_loaded[str(self.nSub)].keys():
                                self.indexes_loaded[str(self.nSub)]["neg"] = {}

                            neg_cls_label_name,neg_cls_label_id=  self.getLabel(index=neg_sampled_index[0])

                            self.indexes_loaded[str(self.nSub)]["neg"].append(neg_sampled_index[0])
                            self.loaded_indexes_features[str(self.nSub)]["neg"][neg_sampled_index[0]] = [neg_img_feat, neg_eeg_feat,neg_cls_label_id]

        self.preloaded_data = True
        print(f"Dataset preload complete")

        
    def __getitem__(self, index):
        # if self.preloaded_data:
        #     img_feat, eeg_feat, cls_label_id = self.loaded_indexes_features[str(self.nSub)][index]

        #     if self.agument_data:
        #         img_feat2, eeg_feat2, cls_label_id_2 = self.loaded_indexes_features[str(self.constrastive_subject)][index]

        #         if not self.include_neg_sample:
        #             return (eeg_feat, img_feat, cls_label_id, self.nSub), (eeg_feat2, img_feat2, cls_label_id_2, self.constrastive_subject), ([], [], [],[])

        #         neg_img_feat, neg_eeg_feat, neg_cls_label_id = self.loaded_indexes_features[str(self.nSub)]["neg"][index]
                
        #         return (eeg_feat, img_feat, cls_label_id,self.nSub),\
        #                 (eeg_feat2, img_feat2, cls_label_id_2,self.constrastive_subject), \
        #                 (neg_eeg_feat, neg_img_feat, neg_cls_label_id,self.nSub)
        #     else:
        #         return (eeg_feat, img_feat, cls_label_id,self.nSub),([],[],[],[]),([],[],[],[])

        if not self.load_individual_files:
            cls_label_name = self.labels[index]
            cls_label_id = self.class_to_id[cls_label_name]

            # random_session = random.randint(0, 3)
            # random_session = 3
            # print(self.data[index].shape)

            eeg_feat = self.data[index]
            if self.mean_eeg_data:
                eeg_feat = np.mean(eeg_feat,axis=0,keepdims=self.keep_dim_after_mean) # contains 4 sessions (4,63,250)
            img_feat = self.img_feature[index]

            if not self.agument_data:
                return (eeg_feat, img_feat, cls_label_id,self.nSub),([],[],[],[]),([],[],[],[])

            # random_session = random.randint(0, 3)
            eeg_feat2 = self.c_data[index]
            if self.mean_eeg_data:
                eeg_feat2 = np.mean(eeg_feat2,axis=0,keepdims=self.keep_dim_after_mean)


            if not self.include_neg_sample:
                return (eeg_feat, img_feat, cls_label_id, self.nSub), (eeg_feat2, img_feat, cls_label_id, self.constrastive_subject), ([], [], [],[])


            # get negative pair
            classkeys = copy.deepcopy(list(self.class_wise_data.keys())) # get all classes
            del classkeys[classkeys.index(cls_label_name)] # drop current class

            sampled_class = random.sample(classkeys,1)[0] # sample one class
            neg_class_sample_indexes = copy.deepcopy(self.class_wise_data[sampled_class])
            neg_sampled_index = random.sample(neg_class_sample_indexes,1)[0]
            # random_session = random.randint(0, 3)
            neg_eeg_feat = self.data[neg_sampled_index]
            if self.mean_eeg_data:
                neg_eeg_feat = np.mean(neg_eeg_feat,axis=0,keepdims=self.keep_dim_after_mean)

            neg_img_feat = self.img_feature[neg_sampled_index]
            neg_cls_label_name,neg_cls_label_id=  self.getLabel(index=neg_sampled_index)

            return (eeg_feat, img_feat, cls_label_id,self.nSub),\
            (eeg_feat2, img_feat, cls_label_id,self.constrastive_subject), \
                (neg_eeg_feat, neg_img_feat, neg_cls_label_id,self.nSub)
        


        if index in self.indexes_loaded[str(self.nSub)]:
            img_feat, eeg_feat = self.loaded_indexes_features[str(self.nSub)][index]
        else:
            img_feat, eeg_feat = self._load_npy(index)
            if self.cache_data:
                self.indexes_loaded[str(self.nSub)].append(index)
                self.loaded_indexes_features[str(self.nSub)][index] = [img_feat, eeg_feat]

        cls_label_name = self.labels[index]
        cls_label_id = self.class_to_id[cls_label_name]
        
        # if augment data then we have for one class 10 images(train) and 4 EEG sessions for each. we can pick any conbinations from these.
        if self.agument_data:

            class_sample_indexes = copy.deepcopy(self.class_wise_data[cls_label_name]) 
            if index in class_sample_indexes:
                index_to_del =  class_sample_indexes.index(index)
                class_sample_indexes.pop(index_to_del)

            sampled_index = random.sample(class_sample_indexes,1)
            # print("Index loaded", sampled_index, class_sample_indexes)

            if sampled_index[0] in self.indexes_loaded[str(self.constrastive_subject)]:
                img_feat2, eeg_feat2 = self.loaded_indexes_features[str(self.constrastive_subject)][sampled_index[0]]
            else:
                img_feat2, eeg_feat2 = self._load_npy(sampled_index[0],subject=self.constrastive_subject)
                if self.cache_data:
                    self.indexes_loaded[str(self.constrastive_subject)].append(sampled_index[0])
                    self.loaded_indexes_features[str(self.constrastive_subject)][sampled_index[0]] = [img_feat2, eeg_feat2]


            cls_label_name_2,cls_label_id_2=  self.getLabel(index=sampled_index[0])
            # cls_label_name_2 = self.labels[sampled_index[0]]
            # cls_label_id_2 = self.class_to_id[cls_label_name_2]

            if not self.include_neg_sample:
                return (eeg_feat, img_feat, cls_label_id, self.nSub), (eeg_feat2, img_feat2, cls_label_id_2, self.constrastive_subject), ([], [], [],[])

            # get negative pair
            classkeys = copy.deepcopy(list(self.class_wise_data.keys())) # get all classes
            del classkeys[classkeys.index(cls_label_name)] # drop current class

            sampled_class = random.sample(classkeys,1) # sample one class
            neg_class_sample_indexes = copy.deepcopy(self.class_wise_data[sampled_class[0]])
            neg_sampled_index = random.sample(neg_class_sample_indexes,1)

            if neg_sampled_index[0] in self.indexes_loaded:
                neg_img_feat, neg_eeg_feat = self.loaded_indexes_features[str(self.nSub)][neg_sampled_index[0]]
            else:
                neg_img_feat, neg_eeg_feat = self._load_npy(neg_sampled_index[0])
                if self.cache_data:
                    self.indexes_loaded[str(self.nSub)].append(neg_sampled_index[0])
                    self.loaded_indexes_features[str(self.nSub)][neg_sampled_index[0]] = [neg_img_feat, neg_eeg_feat]

            neg_cls_label_name,neg_cls_label_id=  self.getLabel(index=neg_sampled_index[0])


            return (eeg_feat, img_feat, cls_label_id,self.nSub),\
            (eeg_feat2, img_feat2, cls_label_id_2,self.constrastive_subject), \
                (neg_eeg_feat, neg_img_feat, neg_cls_label_id,self.nSub)

        return (eeg_feat, img_feat, cls_label_id,self.nSub),([],[],[],[]),([],[],[],[])