import torch
from torch.utils.data import DataLoader, Dataset
import glob
import torchvision.transforms as torch_transform

from PIL import Image
import numpy as np
import os

def hr_transform(rotate=0, mode = 'train'):
    transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.ToTensor()
    ])
    if mode == 'train' & rotate>0.7:
        transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.RandomRotation((90,90)),
        torch_transform.ToTensor()
    ])
    return transform

def lr_transform(image_size, rotate=0, upscale_factor = 4, mode = 'train'):

    transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.Resize(int(image_size//upscale_factor),interpolation=Image.BICUBIC),
        torch_transform.RandomRotation((90, 90)),
        torch_transform.ToTensor()
    ])

    if mode == 'train' & rotate>0.7:
        transform = torch_transform.Compose([
            torch_transform.ToPILImage(),
            torch_transform.Resize(int(image_size // upscale_factor), interpolation=Image.BICUBIC),
            torch_transform.ToTensor()
        ])
    return transform


class Dataset_Train(Dataset):
    def __init__(self, dirpath_input, dirpath_ref, upscale_factor = 4):
        super(Dataset_Train, self).__init__()
        self.imagelist_input = glob.glob(os.path.join(dirpath_input,"*.png"))
        self.imagelist_ref = glob.glob(os.path.join(dirpath_ref, "*.png"))
        self.upscalefactor = upscale_factor

    def __getitem__(self, index):
        input_image = Image.open(self.imagelist_input[index])
        ref_image = Image.open(self.imagelist_ref[index])

        npimage_input = np.array(input_image)
        npimage_ref = np.array(ref_image)

        imagesize = npimage_ref.shape()
        imagesize = imagesize[1]

        rotatenum = np.random.rand()

        self.hr_transform = hr_transform(rotate=rotatenum, mode = 'train')
        self.lr_transform = lr_transform(image_size=imagesize, rotate=rotatenum, upscale_factor=self.upscalefactor, mode = 'train')

        lr_image = self.lr_transform(npimage_input)
        hr_image = self.hr_transform(npimage_input)
        ref_image = self.hr_transform(npimage_ref)

        return lr_image, hr_image, ref_image

    def __len__(self):
        return len(self.imagelist_input)


class Dataset_Vaild(Dataset):
    def __init__(self, dirpath, upscale_factor = 4):
        super(Dataset_Vaild, self).__init__()
        self.upscale_factor = upscale_factor
        self.imagelist = glob.glob(os.path.join(dirpath,"*.png"))

    def __getitem__(self, index):
        image = Image.open(self.imagelist[index])
        image = np.array(image)

        imagesize = image.shape
        imagesize = imagesize[1]
        self.hr_transform = hr_transform(mode = 'evaluation')
        self.lr_transform = lr_transform(image_size=imagesize, upscale_factor = self.upscale_factor, mode = 'evaluation')

        hr_image = self.hr_transform(image)
        lr_image = self.lr_transform(hr_image)
        print("size of hr_image : {}".format(hr_image.shape))
        print("size of hr_image : {}".format(lr_image.shape))
        return lr_image, hr_image

    def __len__(self):
        return len(self.imagelist)
