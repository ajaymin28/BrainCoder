from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from imutils import paths
from PIL import Image
from tqdm import tqdm
import os

# custom dataset class
class Caltech101Dataset(Dataset):
    def __init__(self, images_path, preprocessin_fn = None, filter_label=None):

        image_paths = list(paths.list_images(images_path))
        self.str_labels = []
        self.images = []

        for img_path in tqdm(image_paths):
            label = img_path.split(os.path.sep)[-2]
            if label == "BACKGROUND_Google":
                continue

            if filter_label is not None:
                if label == filter_label: 
                    self.images.append(img_path.replace("\\", "/"))
                    self.str_labels.append(label)
            else:
                self.images.append(img_path.replace("\\", "/"))
                self.str_labels.append(label)

        lb = LabelEncoder()
        self.labels = lb.fit_transform(self.str_labels)

        self.class_str_to_id = {}
        for intlab, strlab in zip(self.labels,self.str_labels):
            if strlab not in self.class_str_to_id:
                self.class_str_to_id[strlab] = intlab

        # print(self.class_str_to_id)

        self.int_to_str_labels = {y: x for x, y in self.class_str_to_id.items()}
        # print(self.int_to_str_labels)

        # print(f"Total Number of Classes: {len(lb.classes_)}")

        self.transforms = preprocessin_fn


    def getOriginalImage(self, idx):
        ImagePath = self.images[idx][:]
        imageOriginal = Image.open(ImagePath).convert('RGB')
        return imageOriginal
    
    def getImagePath(self, idx):
        ImagePath = self.images[idx][:]
        return ImagePath
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ImagePath = self.images[idx][:]

        image = Image.open(ImagePath).convert('RGB')
        LabelClassName = self.int_to_str_labels[self.labels[idx]] 
        LabelClasId = self.class_str_to_id[LabelClassName]

        if self.transforms:
            image = self.transforms(image)

        return None,{"ClassName": LabelClassName, "ClassId": LabelClasId}  ,image, idx 