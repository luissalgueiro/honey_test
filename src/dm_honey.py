import torch
# import imageio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations 
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
import os
from skimage.transform import resize
import cv2

import matplotlib.pyplot as plt
# from kornia.filters import gaussian_blur2d, blur_pool2d
# from kornia.geometry.transform import scale, rescale

####################################################
####################################################

# From https://juansensio.com/blog/062_multihead_attention
class HoneyDataset(torch.utils.data.Dataset):
    def __init__(self, mode, df):
        self.mode = mode
        self.df = df 
        self.mean_img = (0.485, 0.456, 0.406 )
        self.std_img = (0.229, 0.224, 0.225)
        self.classes = ['Pinus','Erica.m', 'Cistus sp', 'Lavandula', 'Citrus sp', 'Helianthus annuus',
                        'Eucalyptus sp.', 'Rosmarinus officinalis', 'Brassica', 'Cardus', 'Tilia', 'Taraxacum']
    def __crop_padding(self,img):
        ## convert to gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ## set threshold for 0
        _,thresh = cv2.threshold(img_gray,10,255,cv2.THRESH_BINARY)
        ## find contours
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        crop = img[y:y+h,x:x+w,:]
        return crop
    def __getitem__(self, index):
        name_img = self.df['name'].iloc[index]
        label    = self.df['labels'].iloc[index]
        ## READ IMAGE
        image = plt.imread(name_img)
        image = self.__crop_padding(image)
        target = torch.tensor(self.classes.index(label))
        # print(f'Image shape: {image.shape} \t Target:{target}')
        if self.mode=='train':
            train_augm = albumentations.Compose(
              [
               albumentations.Resize(height=320,width=320),
               albumentations.Normalize(self.mean_img, self.std_img, max_pixel_value=255.0, always_apply=True),
               albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
               albumentations.HorizontalFlip(p=0.5),
               albumentations.VerticalFlip(p=0.5)
            #    albumentations.Flip(p=0.5),
              ]
            )
            transformed = train_augm(image=image)
            image=transformed['image']
        else:
            valid_augm = albumentations.Compose(
              [
               albumentations.Resize(height=320,width=320),
               albumentations.Normalize(self.mean_img, self.std_img, max_pixel_value=255.0, always_apply=True)
              ]
            )
            transformed = valid_augm(image=image)
            image=transformed['image']
        image = torch.from_numpy(image.transpose()).float()
        target_oh = torch.nn.functional.one_hot(target, num_classes=12).float()
        data = {"image":image,
                "target_oh":target_oh,
                'target':target,
                'class_name':label } 
        # print(f'Image shape: {image.shape} \t Target:{target}')
        return data
    def __len__(self):
        return len(self.df)

class HoneyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4,\
        Dataset = HoneyDataset,\
        sampler=None,\
        df_train=None,\
        df_val=None,\
        workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.Dataset    = Dataset
        self.sampler    = sampler
        self.workers    = workers
        self.train_ds   =  self.Dataset(mode='train', df = df_train)
        self.val_ds     =  self.Dataset(mode='val',   df = df_val)
        # self.test_ds    =  self.Dataset(mode='test',  df = df_test)
        print(f'*** Init training with BatchSize[Train={self.batch_size}] and {self.workers} workers')

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          # shuffle=True,
                          num_workers=self.workers,
                          pin_memory=True,
                          drop_last=True,
                          sampler=self.sampler
                          )
    def val_dataloader(self):
        return DataLoader(self.val_ds,\
            batch_size=4,\
            shuffle=False,\
            num_workers=self.workers,\
            pin_memory=True,\
            drop_last=True,\
            )
    # def test_dataloader(self):
    #     return DataLoader(self.test_ds,\
    #         batch_size=1,\
    #         shuffle=False,\
    #         num_workers=0,\
    #         pin_memory=True,\
    #         drop_last=False )
    
# dm = HoneyDataModule(Dataset=HoneyDataset)
