"""
Obtain ViT features of training and test images in Things-EEG.

using huggingface pretrained ResNet model

"""

import argparse
import torch.nn as nn
import numpy as np
import torch
import os
from PIL import Image
# from transformers import AutoImageProcessor, ResNetForImageClassification
from tqdm import tqdm
from torchvision import transforms

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/home/ja882177/EEG/gits/NICE-EEG/Data/Things-EEG2', type=str)
args = parser.parse_args()

print('Extract feature maps ResNet <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
# model = model.cuda()
# model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])
model.eval()


# processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
processor =  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])

# Image directories
img_set_dir = os.path.join(args.project_dir, 'Image_set')
img_partitions = os.listdir(img_set_dir)
for p in img_partitions:
	part_dir = os.path.join(img_set_dir, p)
	image_list = []
	for root, dirs, files in os.walk(part_dir):
		for file in files:
			if file.endswith(".jpg") or file.endswith(".JPEG"):
				image_list.append(os.path.join(root,file))
	image_list.sort()
	# Create the saving directory if not existing
	save_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
		'full_feature_maps', 'dinov2', 'pretrained-'+str(args.pretrained), p)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)

	# Extract and save the feature maps
	# * better to use a dataloader
	for i, image in tqdm(enumerate(image_list),total=len(image_list)):
		img = Image.open(image).convert('RGB')

		image_tensors = processor(img).unsqueeze(0).cuda()

		with torch.no_grad():

			image_features = model(image_tensors) #384 small, #768 base, #1024 large
			image_features /= image_features.norm(dim=-1, keepdim=True)
			image_features = image_features.cpu().numpy()

			# x = model(**inputs).logits[0]
			# feats = x.detach().cpu().numpy()


			file_name = p + '_' + format(i+1, '07')
			np.save(os.path.join(save_dir, file_name), image_features)
