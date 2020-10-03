import PIL.Image as Image
import numpy as np
import os
import matplotlib.pyplot as plt


import cv2
from Data_gen import Dataset_Test
from torch.utils.data import DataLoader


def showpatch(imagepatch,  modelname ,foldername=None, istensor = True):
    batchsize = imagepatch.shape[0]
    channelsize = imagepatch.shape[1]
    #print(imagepatch.shape)
    #print(batchsize)

    if istensor:
        imagepatch = np.array(imagepatch.cpu().detach())
    folderpath = os.path.join("Network_patches",modelname, foldername)
    
    print("start visulization {}, channelsize : {}".format(foldername,channelsize))

    if not os.path.isdir(os.path.join("Network_patches",modelname)):
        os.mkdir(os.path.join("Network_patches",modelname))
    if not os.path.isdir(folderpath):
        os.mkdir(folderpath)

    for index in range(batchsize):
        patches = imagepatch[index]
        for channel in range(0,channelsize,4):
        #for channel in range(0,channelsize):
            image = regularization_image(patches[channel])
            image = (image*255).astype(np.uint8)
            #PIL_Input_Image = Image.fromarray(image)
            #PIL_Input_Image.save(os.path.join(folderpath,"image{}.png".format(index)))
            plt.imshow(image, 'gray')
            plt.savefig(os.path.join(folderpath,"image{}.png".format(channel)))



def saveoffset(offsetbatch, modelname, foldername, istensor = False):
    if istensor:
        offsetbatch = np.array(offsetbatch.cpu().detach())
        offsetbatch = np.transpose(offsetbatch, (0, 2, 3, 1))
        offsetbatch = np.squeeze(offsetbatch)
    sizetemp = offsetbatch.shape[:-1]
    offset_coord = np.zeros((*sizetemp, int(offsetbatch.shape[-1] / 2), 2), dtype=np.float32)

    for y in range(offset_coord.shape[0]):
        for x in range(offset_coord.shape[1]):
            for i in range(offset_coord.shape[2]):
                coordtuple = offsetbatch[y,x,i*2:(i+1)*2]
                offset_coord[y,x,i] = coordtuple

    folderpath = os.path.join("Network_patches",modelname, foldername)

    if not os.path.isdir(folderpath):
        os.mkdir(folderpath)

    for i in range(offsetbatch.shape[0]):
        np.save(os.path.join(folderpath, "offset_{}.npy".format(i)), offsetbatch[i])

    for i in range(offsetbatch.shape[0]):
        np.save(os.path.join(folderpath, "offset_{}.npy".format(i)), offsetbatch[i])
    return offset_coord



def getPSNR(image1, image2):
    MSE = (np.square(image1-image2)).mean(axis = None)

    PSNR = 20*np.log10(1/np.sqrt(MSE))
    return PSNR


def regularization_image(image):
    min = np.min(image)
    temp_image = image-min

    max = np.max(temp_image)
    temp_image = temp_image/max

    return temp_image


def regularize_testimage(image, istensor=True):
    if istensor:
        image = np.array(image.cpu().detach())
    else:
        image = np.array(image)
    image = image.squeeze()
    image = regularization_image(image)
    return image


if __name__ == "__main__":

    offset1= np.load("Network_patches/offset/offset_deformconv1.npy")
    offset1 = saveoffset(offset1)
    print(offset1.shape)
    print(offset1[40][40])

