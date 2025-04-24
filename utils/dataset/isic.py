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
from torchvision import tv_tensors
from collections import Counter
import torch
import numpy as np

import torchvision.transforms.v2 as transforms_v2

'''Function to enhance image using mask and contour 3channel and 2 channel'''
def enhance_image(image, mask, contour, enhanced_type='3channel'):
    if enhanced_type == '3channel':
        # print("shape of image:", image.shape)
        # print("shape of mask:", mask.shape)
        # print("shape of contour:", contour.shape)
        image[ 0, :, :] = image[ 0, :, :] + mask + contour
        image[ 1, :, :] = image[ 0, :, :] + mask + contour
        image[ 2, :, :] = image[ 0, :, :] + mask + contour
    elif enhanced_type == '2channel':
        image[:, 1, :, :] += mask 
        image[:, 2, :, :] += contour
    else:
        raise ValueError("Invalid enhanced_type. Choose '3channel' or '2channel'.")

    return image, mask, contour

#=====================old version=====================
# augmentation_rand = transforms.Compose(
#     [transforms.Resize((224,224)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#         ], p=0.8),
#     transforms.ToTensor()]
#     # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))]
#     )

# augmentation_sim = transforms.Compose(
#     [transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomRotation(90),
#     transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#         ], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor()]
#     # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))]
#     )


# augmentation_test = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
#     ])
#=====================old version=====================

#transform_v2
# augmentation_rand = transforms_v2.Compose(
#     [
#         transforms_v2.Resize((224,224)),
#         transforms_v2.RandomHorizontalFlip(p=0.5),
#         transforms_v2.RandomApply([
#                 transforms_v2.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#             ], p=0.8),
#         transforms_v2.ToTensor()
#     ]
# )

# augmentation_sim = transforms_v2.Compose(
#     [
#         transforms_v2.RandomResizedCrop(224,scale=(0.8,1.0)),
#         transforms_v2.RandomHorizontalFlip(p=0.5),
#         transforms_v2.RandomVerticalFlip(p=0.5),
#         transforms_v2.RandomRotation(90),
#         transforms_v2.RandomApply([
#                 transforms_v2.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#             ], p=0.8),
#         transforms_v2.RandomGrayscale(p=0.2),
#         transforms_v2.ToTensor()
#     ]
# )

# augmentation_test = transforms_v2.Compose(
#     [
#         transforms_v2.Resize(224),
#         transforms_v2.ToTensor(),
#     ]
# )

'''Transfrom with normalization'''
augmentation_rand = transforms_v2.Compose(
    [
        transforms_v2.Resize((224,224)),
        transforms_v2.RandomHorizontalFlip(p=0.5),
        # transforms_v2.RandomApply([
        #         transforms_v2.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        #     ], p=0.8),
        
        transforms_v2.ToTensor(),
        # transforms_v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
        ]
)

augmentation_sim = transforms_v2.Compose(
    [
        transforms_v2.RandomResizedCrop(224,scale=(0.8,1.0)),
        transforms_v2.RandomHorizontalFlip(p=0.5),
        transforms_v2.RandomVerticalFlip(p=0.5),
        transforms_v2.RandomRotation(90),
        # transforms_v2.RandomApply([
        #         transforms_v2.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        #     ], p=0.8),
        transforms_v2.RandomGrayscale(p=0.2),
        transforms_v2.ToTensor(),
        # transforms_v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

augmentation_test = transforms_v2.Compose(
    [
        transforms_v2.Resize(224),
        
        transforms_v2.ToTensor(),
        # transforms_v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)


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
    def __init__(self,path,transform,mode='train', dataset_type='h5_file', enhanced=True):
        self.dataset_type = dataset_type
        self.path = path
        self.transform = transform
        self.mode = mode
        self.enhanced = enhanced
        self.enhanced_flag = False
        self.normalize = transforms_v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

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
                self.labels = f['metadata/labels'][:]


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

        elif self.dataset_type == 'h5_file' :
            #item as index
            idx_str = str(item)
            with h5py.File(self.path, 'r') as f:
                # Get original image
                # self.num_samples = len(f['images'])
                image = f['images'][idx_str][:]
                image_pil = Image.fromarray(image)
                # Get mask (equivalent to seg_prior in your original code)
                mask = f['masks'][idx_str][:].astype(np.float32)
                
                # Get contour (equivalent to boundary_prior in your original code)
                contour = f['contours'][idx_str][:].astype(np.float32)
                
                # Convert to tv_tensors for compatibility with transforms
                mask = tv_tensors.Mask(mask)
                contour = tv_tensors.Mask(contour)
                
                # Use a placeholder label (you should replace this with actual labels if available)
                label = self.labels[item]
                label = torch.LongTensor([label]).unsqueeze(0)

                                    #====================  

                if self.enhanced: # new code: enhanced
                    if self.enhanced_flag == False:
                        self.enhanced_flag = True
                        print("Enhancement mode")
                    
                    if self.transform is not None:
                        if self.mode == 'train':
                            original_image = image_pil.copy()  # Save the original image for later use
                            original_image = transforms_v2.ToTensor()(original_image)
                            img1,mask1,contour1 = self.transform[0](image_pil,mask,contour)
                            img2,mask2,contour2 = self.transform[1](image_pil,mask,contour)
                            img1 = self.normalize(img1)
                            img2 = self.normalize(img2)


                            # Enhance the image using mask and contour
                            img1, mask, contour = enhance_image(img1, mask1, contour1, enhanced_type='3channel')
                            img2, mask, contour = enhance_image(img2, mask2, contour2, enhanced_type='3channel')
                            
                           

                            return [img1,img2],label, original_image
                        else:
                            img1,mask1,contour1 = self.transform(image_pil,mask,contour)

                            img1 = self.normalize(img1)

                            # Enhance the image using mask and contour
                            img1, mask1, contour1 = enhance_image(img1, mask1, contour1, enhanced_type='3channel')

                            return img1, label
                    else:
                        return image, label, mask, contour
                    
                    #====================  
                else:# original code: not enhanced
                    if self.enhanced_flag == False:
                        print("Original mode")
                        self.enhanced_flag = True
                    if self.transform is not None:
                        if self.mode == 'train':
                            original_image = image_pil.copy() 
                            original_image = transforms_v2.ToTensor()(original_image)
                            img1 = self.transform[0](image_pil)
                            img2 = self.transform[1](image_pil)

                            return [img1,img2],label,original_image
                        else:
                            img1 = self.transform(image_pil)
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