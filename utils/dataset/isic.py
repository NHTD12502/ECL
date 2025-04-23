#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: isic.py
@author:zyl
@contact:zhangyilan@buaa.edu.cn
@time:2023/7/8 17:33
'''

import os
import h5py
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch

augmentation_rand = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
    transforms.ToTensor()]
    # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))]
    )

augmentation_sim = transforms.Compose(
    [transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()]
    # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))]
    )


augmentation_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
    ])

class isic2019_dataset(Dataset):
    def __init__(self,path,transform,mode='train'):
        self.path = path
        self.transform = transform
        self.mode = mode

        if self.mode == 'train':
            self.df = pd.read_csv(os.path.join(path,'ISIC2019_train.csv'))
        elif self.mode == 'valid':
            self.df = pd.read_csv(os.path.join(path,'ISIC2019_val.csv'))
        else:
            self.df = pd.read_csv(os.path.join(path,'ISIC2019_test.csv'))

    def __getitem__(self, item):
        img_path = os.path.join(self.path,'ISIC2019_Dataset',self.df.iloc[item]['category'],f"{self.df.iloc[item]['image']}.jpg")
        img = Image.open(img_path)
        if (img.mode != 'RGB'):
            img = img.convert("RGB")

        label = int(self.df.iloc[item]['label'])
        label = torch.LongTensor([label])

        if self.transform is not None:
            if self.mode == 'train':
                img1 = self.transform[0](img)
                img2 = self.transform[1](img)

                return [img1,img2],label
            else:
                img1 = self.transform(img)
                return img1, label
        else:
            raise Exception("Transform is None")

    def __len__(self):
        return len(list(self.df['image']))


class isic2018_dataset(Dataset):
    def __init__(self,path,transform,mode='train', dataset_type='h5_file'):
        self.dataset_type = dataset_type
        self.path = path
        self.transform = transform
        self.mode = mode

        if self.dataset_type != 'h5_file':
            if self.mode == 'train':
                self.df = pd.read_csv(os.path.join(path,'ISIC2018_train.csv'))
            elif self.mode == 'valid':
                self.df = pd.read_csv(os.path.join(path,'ISIC2018_val.csv'))
            else:
                self.df = pd.read_csv(os.path.join(path,'ISIC2018_test.csv'))


        else:
            with h5py.File(self.path, 'r') as f:
                # Get original image
                self.num_samples = len(f['images'])


    def __getitem__(self, item):
        if self.dataset_type != 'h5_file':
            #=============================================== old
            img_path = os.path.join(self.path,'ISIC2018_Dataset',self.df.iloc[item]['category'],f"{self.df.iloc[item]['image']}.jpg")
            img = Image.open(img_path)
            if (img.mode != 'RGB'):
                img = img.convert("RGB")

            label = int(self.df.iloc[item]['label'])
            label = torch.LongTensor([label])

            if self.transform is not None:
                if self.mode == 'train':
                    img1 = self.transform[0](img)
                    img2 = self.transform[1](img)

                    return [img1,img2],label
                else:
                    img1 = self.transform(img)
                    return img1, label
            else:
                raise Exception("Transform is None")
                        #=============================================== old

        else:
            #item as index
            idx_str = str(item)
            with h5py.File(self.path, 'r') as f:
                # Get original image
                # self.num_samples = len(f['images'])
                image = f['images'][idx_str][:]
                
                # Get mask (equivalent to seg_prior in your original code)
                mask = f['masks'][idx_str][:].astype(np.float32)
                
                # Get contour (equivalent to boundary_prior in your original code)
                contour = f['contours'][idx_str][:].astype(np.float32)
                
                # Convert to tv_tensors for compatibility with transforms
                # mask = tv_tensors.Mask(mask)
                # contour = tv_tensors.Mask(contour)
                
                # Use a placeholder label (you should replace this with actual labels if available)
                label = self.labels[item]
                
                if self.transform is not None:
                    if self.mode == 'train':
                        img1 = self.transform[0](image)
                        img2 = self.transform[1](image)

                        return [img1,img2],label
                    else:
                        img1 = self.transform(image)
                        return img1, label
                else:
                    return image, label, mask, contour

    def __len__(self):
        if self.dataset_type != 'h5_file':
            return len(list(self.df['image']))
        else:
            return self.num_samples

# class ISIC2018Dataset_enhanced(Dataset):
#     def __init__(self, h5_path, geo_transform=None, normalize_transform=None, task="train"):
#         self.h5_path = h5_path
#         self.geo_transform = geo_transform
#         self.normalize_transform = normalize_transform
#         self.task = task
        
#         # Standard mapping for ISIC classes
#         self.standard_mapping = {
#             'AKIEC': 0,
#             'BCC': 1,
#             'BKL': 2, 
#             'DF': 3,
#             'MEL': 4,
#             'NV': 5,
#             'VASC': 6
#         }

#         # Read metadata and determine dataset size
#         with h5py.File(self.h5_path, 'r') as f:
#             # Get number of samples by counting entries in the 'images' group
#             self.num_samples = len(f['images'])
            
#             # Print dataset information
#             print(f"Dataset: {h5_path}")
#             print(f"Number of samples: {self.num_samples}")
#             print(f"Data structure: images ({f['images']['0'].shape}), masks ({f['masks']['0'].shape}), contours ({f['contours']['0'].shape})")
#             print("-" * 50)
            
#             # For compatibility, we'll set these attributes 
#             # You may need to add actual class labels if they're available elsewhere
#             self.labels = f['metadata/labels'][:]

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         idx_str = str(idx)  # Convert index to string for accessing h5 file
        
#         with h5py.File(self.h5_path, 'r') as f:
#             # Get original image
#             image = f['images'][idx_str][:]
            
#             # Get mask (equivalent to seg_prior in your original code)
#             mask = f['masks'][idx_str][:].astype(np.float32)
            
#             # Get contour (equivalent to boundary_prior in your original code)
#             contour = f['contours'][idx_str][:].astype(np.float32)
            
#             # Convert to tv_tensors for compatibility with transforms
#             # mask = tv_tensors.Mask(mask)
#             # contour = tv_tensors.Mask(contour)
            
#             # Use a placeholder label (you should replace this with actual labels if available)
#             label = self.labels[idx]
            
#             # # Apply geometric transformations if provided
#             # if self.geo_transform:
#             #     transformed_tensors = self.geo_transform(image, mask, contour)
#             #     image, mask, contour = transformed_tensors
            
#             # # Apply normalization to the image if provided
#             # if self.normalize_transform:
#             #     image = self.normalize_transform(image)
            
#             if self.transform is not None:
#                 if self.mode == 'train':
#                     img1 = self.transform[0](image)
#                     img2 = self.transform[1](image)

#                     return [img1,img2],label
#                 else:
#                     img1 = self.transform(image)
#                     return img1, label
            
#             return image, label, mask, contour