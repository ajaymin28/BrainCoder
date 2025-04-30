import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class EEGDataset:
    # Constructor
    def __init__(self, eeg_signals_path, eeg_splits_path, subset='train',subject=1, time_low=20,time_high=480, model_type="cnn", imagesRoot="./data/images/imageNet_images", preprocessin_fn=None):
        # Load EEG signals

        assert subset=='train' or subset=='val' or subset=='test'

        self.time_low = time_low
        self.time_high = time_high
        self.model_type = model_type
        self.imagesRoot = imagesRoot

        splits = torch.load(eeg_splits_path)
        subset_indexes = splits["splits"][0][f"{subset}"]

        loaded = torch.load(eeg_signals_path)

        self.subsetData = []
        self.labels = []
        self.images = []

        self.class_labels = loaded["labels"]
        image_names = loaded['images']

        EEGSelectedImageNetClasses = []
        for imageP in image_names:
            class_folder_name = imageP.split("_")[0]
            EEGSelectedImageNetClasses.append(class_folder_name)

        self.class_labels_names = {}
        self.class_id_to_str = {}
        self.class_str_to_id = {}

        lines = []
        with open(f"{imagesRoot}/labels.txt") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                line = line.strip()
                line = line.split(" ")
                imagenetDirName = line[0]
                imagenetClassId = line[1]
                imagenetClassName = line[-1]
                if imagenetDirName in EEGSelectedImageNetClasses:
                    indexOfClass = self.class_labels.index(imagenetDirName)
                    self.class_labels_names[imagenetDirName] = {"ClassId": int(indexOfClass), "ClassName": imagenetClassName}
                    self.class_id_to_str[int(indexOfClass)]= imagenetClassName
                    self.class_str_to_id[imagenetClassName]= int(indexOfClass)

        for sub_idx in subset_indexes:
            if subject!=0:
                if loaded['dataset'][sub_idx]['subject']==subject:
                    self.subsetData.append(loaded['dataset'][sub_idx])
                    self.labels.append(loaded["dataset"][sub_idx]['label'])
                    self.images.append(image_names[loaded["dataset"][sub_idx]['image']])
            else:
                sub_idx = int(sub_idx)
                self.subsetData.append(loaded['dataset'][sub_idx])
                self.labels.append(loaded["dataset"][sub_idx]['label'])
                self.images.append(image_names[loaded["dataset"][sub_idx]['image']])

        # Compute size
        self.size = len(self.subsetData)

        self.preprocessin_fn = None
        if preprocessin_fn is not None:
            self.preprocessin_fn = preprocessin_fn

        self.trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)

        self.isDataTransformed = False


    def __len__(self):
        return self.size
    
    def getOriginalImage(self, idx):
        class_folder_name = self.images[idx].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[idx]}.JPEG"
        imageOriginal = Image.open(ImagePath).convert('RGB')
        return imageOriginal
    
    def getImagePath(self, idx):
        class_folder_name = self.images[idx].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[idx]}.JPEG"
        return ImagePath
    
    def transformEEGData(self, resnet_model, resnet_to_eeg_model, device, isVIT=False):
        print("Transforming EEG data")
        for i, image_path in enumerate(self.images):

            class_folder_name = image_path.split("_")[0]
            ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"

            image = Image.open(ImagePath).convert('RGB')
            if self.preprocessin_fn is not None:
                image = self.preprocessin_fn(image)

            # eeg, label, image, idxs = data
            with torch.no_grad():
                if isVIT:
                    features = resnet_model(image.unsqueeze(0).to(device)).last_hidden_state[:, 0]
                else:
                    features = resnet_model(image.unsqueeze(0).to(device))
                    features = features.view(-1, features.size(1))
                outputs = resnet_to_eeg_model(features)
                self.subsetData[i]["eeg"] = outputs
                # print("FC features shape", outputs.view(128,-1).size(), "original eeg shape: ", eeg.reshape(128,-1).size())
        self.isDataTransformed = True
        print("Transforming EEG data (done)")

    def __getitem__(self, i):
        eeg = self.subsetData[i]["eeg"].float().t()
        # print("EE sample size",self.subsetData[i]["eeg"].size())
        # eeg = self.subsetData[i]["eeg"].T
        if not self.isDataTransformed:
            eeg = eeg[self.time_low:self.time_high,:]

        class_folder_name = self.images[i].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"

        label = self.class_labels_names[class_folder_name]

        image = Image.open(ImagePath).convert('RGB')
        if self.preprocessin_fn is not None:
            image = self.preprocessin_fn(image)

        return eeg, label, image, i