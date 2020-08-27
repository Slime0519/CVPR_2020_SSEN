import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import torchvision.transforms as torch_transform
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import os

def hr_transform(rotate=0, mode = 'train'):
    transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.ToTensor()
    ])
    if mode == 'train' and rotate>0.7:
        transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.RandomRotation((90,90)),
        torch_transform.ToTensor()
    ])
    return transform

def lr_transform(image_size, rotate=0, upscale_factor = 4, mode = 'train'):

    transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
     #  torch_transform.Resize(int(image_size//upscale_factor),interpolation=Image.BICUBIC),
        torch_transform.Resize(40, interpolation=Image.BICUBIC),
        torch_transform.ToTensor()
    ])

    if (mode == 'train') and (rotate>0.7):
        transform = torch_transform.Compose([
            torch_transform.ToPILImage(),
            #torch_transform.Resize(int(image_size // upscale_factor), interpolation=Image.BICUBIC),
            torch_transform.Resize(40, interpolation=Image.BICUBIC),
            torch_transform.RandomRotation((90, 90)),
            torch_transform.ToTensor()
        ])

    return transform


def testset_hr_transform(imagesize = (192,192)):
    transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.RandomCrop(imagesize),
        torch_transform.ToTensor()
    ])
    return transform

def testset_lr_transform(image_size, upscale_factor = 4):

    transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.Resize(image_size//upscale_factor, interpolation=Image.BICUBIC),
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
        input_image = Image.open(self.imagelist_input[index]).convert("RGB")
        ref_image = Image.open(self.imagelist_ref[index]).convert("RGB")

        npimage_input = np.array(input_image)
        npimage_ref = np.array(ref_image)

        imagesize = npimage_ref.shape
        imagesize = imagesize[1]

        rotatenum = np.random.rand()

        refimage_size = npimage_ref.shape
        if refimage_size[1] <= 160 or refimage_size[0]<=160:
            npimage_ref = np.pad(npimage_ref,((0,160-refimage_size[0]),(0,160-refimage_size[1]),(0,0)))

        self.hr_transform = hr_transform(rotate=rotatenum, mode = 'train')
        self.lr_transform = lr_transform(image_size=imagesize, rotate=rotatenum, upscale_factor=self.upscalefactor, mode = 'train')

        lr_image = self.lr_transform(npimage_input)
        hr_image = self.hr_transform(npimage_input)
        ref_image = self.hr_transform(npimage_ref)
       # print("{} : lr : {}  / hr : {} / ref : {} ".format(index,lr_image.shape, hr_image.shape, ref_image.shape))

        return lr_image, hr_image, ref_image

    def __len__(self):
        return len(self.imagelist_input)


class Dataset_Vaild(Dataset):
    def __init__(self, dirpath, upscale_factor = 4):
        super(Dataset_Vaild, self).__init__()
        self.upscale_factor = upscale_factor
        self.imagelist = glob.glob(os.path.join(dirpath,"*.png"))

    def __getitem__(self, index):
        image = Image.open(self.imagelist[index]).convert("RGB")
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

class Dataset_Test(Dataset):
    def __init__(self,dirpath, upscale_factor = 4, mode= "XH"):
        self.upscale_factor = upscale_factor
        self.imagelist = glob.glob(os.path.join(dirpath,"*.png"))
        self.mode = mode

        self.group_num = len(self.imagelist)//6
        self.original_path = []
        self.reference_path = []
        for i in range(self.group_num):
            self.original_path.append(self.imagelist[i*6])
            self.reference_path.append(self.imagelist[i*6+1:(i+1)*6])

        self.test_hr_transform = testset_hr_transform()
        self.test_lr_transform = testset_lr_transform(image_size=192, upscale_factor=self.upscale_factor)

       # print(self.group_num)
       # print(self.imagelist)

    def __getitem__(self, index):
        inputimage = Image.open(self.original_path[index]).convert('RGB')
        inputimage = np.array(inputimage)

        if inputimage.shape[0]<192 or inputimage.shape[1]<192:
            index = index-1
            inputimage = Image.open(self.original_path[index]).convert('RGB')
            inputimage = np.array(inputimage)

        if inputimage.shape[0] % self.upscale_factor != 0:
            inputimage = inputimage[:-(inputimage.shape[0]%self.upscale_factor),:]
        if inputimage.shape[1] % self.upscale_factor != 0:
            inputimage = inputimage[:,:-(inputimage.shape[1]%self.upscale_factor)]

        if self.mode == "XH":
            referenceimage = Image.open(self.reference_path[index][0]).convert('RGB')
        elif self.mode == "H":
            referenceimage = Image.open(self.reference_path[index][1]).convert('RGB')
        elif self.mode == "M":
            referenceimage = Image.open(self.reference_path[index][2]).convert('RGB')
        elif self.mode == "L":
            referenceimage = Image.open(self.reference_path[index][3]).convert('RGB')
        elif self.mode == "XL":
            referenceimage = Image.open(self.reference_path[index][4]).convert('RGB')

        referenceimage = np.array(referenceimage)

        if referenceimage.shape[0]> inputimage.shape[0]:
            height = inputimage.shape[0]
        else:
            height = referenceimage.shape[0]

        if referenceimage.shape[1] > inputimage.shape[1]:
            width = inputimage.shape[1]
        else:
            width = referenceimage.shape[1]

        self.test_hr_transform = testset_hr_transform(imagesize=(height,width))
        self.test_lr_transform = testset_lr_transform(image_size=(height,width), upscale_factor=self.upscale_factor)

        input_hr = self.test_hr_transform(inputimage)
        input_lr = self.test_lr_transform(input_hr)
        ref_hr = self.test_hr_transform(referenceimage)

        return input_lr, input_hr, ref_hr

    def __len__(self):
        return self.group_num



