"""
Package DINO features for center images

"""

# import argparse
# import torch.nn as nn
# import numpy as np
# import torch
# import os
# from PIL import Image
# from torchvision import transforms

# gpus = [0]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

# parser = argparse.ArgumentParser()
# parser.add_argument('--pretrained', default=True, type=bool)
# parser.add_argument('--project_dir', default='/home/ja882177/EEG/gits/NICE-EEG/Data/Things-EEG2', type=str)
# args = parser.parse_args()

# print('Extract feature maps DINOv2 of images for center <<<')
# print('\nInput arguments:')
# for key, val in vars(args).items():
#     print('{:16} {}'.format(key, val))

# # Set random seed for reproducible results
# seed = 20200220
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# # Load DINOv2 model
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# model = model.cuda()
# model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
# model.eval()

# # DINOv2 image processor
# processor = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# img_set_dir = os.path.join(args.project_dir, 'Image_set/center_images/')
# condition_list = os.listdir(img_set_dir)
# condition_list.sort()

# all_centers = []

# for cond in condition_list:
#     one_cond_dir = os.path.join(args.project_dir, 'Image_set/center_images/', cond)
#     cond_img_list = os.listdir(one_cond_dir)
#     cond_img_list.sort()
#     cond_center = []
#     for img_name in cond_img_list:
#         img_path = os.path.join(one_cond_dir, img_name)
#         img = Image.open(img_path).convert('RGB')
#         img_tensor = processor(img).unsqueeze(0).cuda()  # Shape: (1, 3, 224, 224)
#         with torch.no_grad():
#             # DINOv2 output: get global pooled feature (see doc: model(img, return_class_token=True))
#             feats = model(img_tensor)  # (1, feat_dim)
#         cond_center.append(np.squeeze(feats.cpu().numpy()))
#     if len(all_centers)==0:
#         print(np.array(cond_center).shape)
#     all_centers.append(np.array(cond_center))

# np.save(os.path.join(args.project_dir, 'center_all_image_dinov2.npy'), all_centers)

# import argparse
# import torch
# import torch.nn as nn
# import numpy as np
# import os
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader

# # ========== DATASET ==========
# class ImageFolderDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.samples = []
#         self.labels = []
#         self.transform = transform
#         self.categories = sorted(os.listdir(root))
#         for idx, cat in enumerate(self.categories):
#             cat_dir = os.path.join(root, cat)
#             if not os.path.isdir(cat_dir):
#                 continue
#             imgs = sorted(os.listdir(cat_dir))
#             for img in imgs:
#                 self.samples.append(os.path.join(cat_dir, img))
#                 self.labels.append(idx)  # optional: store category index

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img = Image.open(self.samples[idx]).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img, self.labels[idx], self.samples[idx]

# # ========== MAIN SCRIPT ==========
# gpus = [0]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

# parser = argparse.ArgumentParser()
# parser.add_argument('--pretrained', default=True, type=bool)
# parser.add_argument('--project_dir', default='/home/ja882177/EEG/gits/NICE-EEG/Data/Things-EEG2', type=str)
# parser.add_argument('--batch_size', default=32, type=int)
# parser.add_argument('--num_workers', default=8, type=int)
# args = parser.parse_args()

# print('Extract feature maps DINOv2 of images for center <<<')
# for key, val in vars(args).items():
#     print('{:16} {}'.format(key, val))

# seed = 20200220
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# # Model
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# model = model.cuda()
# model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
# model.eval()

# processor = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# img_set_dir = os.path.join(args.project_dir, 'Image_set/center_images/')
# dataset = ImageFolderDataset(img_set_dir, transform=processor)
# dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
#                         num_workers=args.num_workers, pin_memory=True)

# # Determine feature dim by running a single forward pass
# with torch.no_grad():
#     for batch_imgs, _, _ in dataloader:
#         batch_imgs = batch_imgs.cuda(non_blocking=True)
#         feats = model(batch_imgs)
#         feat_dim = feats.shape[1]
#         print("feature dim: {}")
#         break

# # Allocate shared CUDA tensor for features (shape: N_images, feat_dim)
# total_images = len(dataset)
# all_features = torch.zeros((total_images, feat_dim), dtype=torch.float32, device='cuda')

# # Extract features in batches and write to tensor
# img_idx = 0
# with torch.no_grad():
#     for batch_imgs, _, _ in dataloader:
#         bs = batch_imgs.shape[0]
#         batch_imgs = batch_imgs.cuda(non_blocking=True)
#         feats = model(batch_imgs)  # (bs, feat_dim)
#         all_features[img_idx:img_idx+bs] = feats
#         img_idx += bs
#         print(f"Processed {img_idx}/{total_images}")

# # Move to CPU and save as numpy
# all_features_cpu = all_features.cpu().numpy()
# np.save(os.path.join(args.project_dir, 'center_all_image_dinov2.npy'), all_features_cpu)
# print(f"Saved {all_features_cpu.shape} features to npy file.")

# Optionally: Save file paths and/or labels if needed

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# ========== DATASET ==========
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.labels = []
        self.categories = []
        self.cat_to_idx = {}
        self.transform = transform

        # build categories
        self.categories = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        for idx, cat in enumerate(self.categories):
            cat_dir = os.path.join(root, cat)
            imgs = sorted(os.listdir(cat_dir))
            for img in imgs:
                self.samples.append(os.path.join(cat_dir, img))
                self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label, self.samples[idx]

# ========== MAIN SCRIPT ==========
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/home/ja882177/EEG/gits/NICE-EEG/Data/Things-EEG2', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=8, type=int)
args = parser.parse_args()

print('Extract feature maps DINOv2 of images for center <<<')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
model.eval()

processor = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_set_dir = os.path.join(args.project_dir, 'Image_set/center_images/')
dataset = ImageFolderDataset(img_set_dir, transform=processor)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

# Determine feature dim by running a single forward pass
with torch.no_grad():
    for batch_imgs, _, _ in dataloader:
        batch_imgs = batch_imgs.cuda(non_blocking=True)
        feats = model(batch_imgs)
        feat_dim = feats.shape[1]
        print(f"feature dim: {feat_dim}")
        break

# Collect features per category
features_by_cat = defaultdict(list)

img_idx = 0
with torch.no_grad():
    for batch_imgs, batch_labels, _ in dataloader:
        bs = batch_imgs.shape[0]
        batch_imgs = batch_imgs.cuda(non_blocking=True)
        feats = model(batch_imgs)  # (bs, feat_dim)
        feats_cpu = feats.cpu().numpy()
        for i in range(bs):
            features_by_cat[batch_labels[i].item()].append(feats_cpu[i])
        img_idx += bs
        print(f"Processed {img_idx}/{len(dataset)}")

# Average features for each category
n_categories = len(dataset.categories)
avg_features = []
for cat_idx in range(n_categories):
    feats = np.stack(features_by_cat[cat_idx])  # [num_imgs_in_cat, feat_dim]
    avg_feat = feats.mean(axis=0)
    avg_features.append(avg_feat)
avg_features = np.stack(avg_features)  # [n_categories, feat_dim]

# Save result
np.save(os.path.join(args.project_dir, 'center_all_image_dinov2_avg_by_cat.npy'), avg_features)
print(f"Saved averaged features: {avg_features.shape}")

# Save category order for reference
with open(os.path.join(args.project_dir, 'center_image_category_order.txt'), 'w') as f:
    for cat in dataset.categories:
        f.write(f"{cat}\n")

