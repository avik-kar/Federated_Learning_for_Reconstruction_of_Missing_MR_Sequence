import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils import data
from torchvision import transforms

class dataLoader(data.Dataset):

    def __init__(
        self,
        is_transform=False,
        augmentations=None,
        img_size=256
    ):
        
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_size = img_size
        self.path = './Data/'
        self.patientList = os.listdir(self.path)
        self.files = []
        for patient in self.patientList:
            self.files += os.listdir(self.path+patient+'/t1/')
        self.tf = transforms.Compose(
            [   transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index].split('_')
        fid = fname[0]+fname[1]+'/'
        
        im_t1 = torch.load(self.path+fid+'t1/'+self.files[index])
        im_t1ce = torch.load(self.path+fid+'t1ce/'+self.files[index])
        im_t2 = torch.load(self.path+fid+'t2/'+self.files[index])

        if self.augmentations is not None:
            im_t1, im_t1ce, im_t2 = self.augmentations(im_t1, im_t1ce, im_t2)
        if self.is_transform:
            im_t1, im_t1ce, im_t2 = self.transform(im_t1, im_t1ce, im_t2)
        
        mis_seq = torch.zeros(im_t1.shape)
        
        return torch.cat((mis_seq, im_t1, im_t1ce, im_t2), 0), torch.cat((mis_seq, mis_seq, im_t1ce, im_t2), 0), torch.cat((mis_seq, im_t1, mis_seq, im_t2), 0), torch.cat((mis_seq, im_t1, im_t1ce, mis_seq), 0)

    def transform(self, im_t1, im_t1ce, im_t2):
        
        im_t1 = self.tf(Image.fromarray((im_t1 - np.min(im_t1))/(np.max(im_t1) - np.min(im_t1))))
        im_t1ce = self.tf(Image.fromarray((im_t1ce - np.min(im_t1ce))/(np.max(im_t1ce) - np.min(im_t1ce))))
        im_t2 = self.tf(Image.fromarray((im_t2 - np.min(im_t2))/(np.max(im_t2) - np.min(im_t2))))
        
        return im_t1, im_t1ce, im_t2

