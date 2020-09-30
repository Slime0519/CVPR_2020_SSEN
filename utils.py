import PIL.Image as Image
import numpy as np
import os
import matplotlib.pyplot as plt

from Modules.OridinaryModels.Baseline import Baseline,Baseline_show
from Modules.OridinaryModels.Baseline_big import BigBaseline,BigBaseline_show
from Modules.OridinaryModels.Baseline_small import Baseline_small,Baseline_small_show
from Modules.OridinaryModels.lightbaseline import Baseline_light
from Modules.OridinaryModels.Baseline128 import Baseline128,Baseline128_show

from Modules.EDSR_pretrained_baseline.EDSR_baseline import EDSR_baseline, EDSR_baseline_show

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

def getprefixname(modeltype):
    if modeltype == "normal_concat":
        prefix_resultname = "normalModel_concat"
    elif modeltype == "normal":
        prefix_resultname = "normalModel"
    elif modeltype == "normal_cosine":
        prefix_resultname = "normalModel_cosine"
    elif modeltype == "normal128":
        prefix_resultname = "normalModel_model128"
    elif modeltype == "normal_cosine_concat":
        prefix_resultname = "normalModel_cosine_concat"
    elif modeltype == "normal_light":
        prefix_resultname = "normalModel_light"
    elif modeltype == "big":
        prefix_resultname = "bigModel"
    elif modeltype == "EDSR_pretrained_baseline":
        prefix_resultname = "EDSR_pretrained_baseline"
    else:
        prefix_resultname = "smallModel"

    return prefix_resultname

def loadmodel(modeltype):
    if modeltype == "normal_concat" or modeltype == "normal_cosine_concat":
        print("load concat baseline module")
        Model = Baseline(mode="concat")
    elif modeltype == "normal" or modeltype == "normal_cosine":
        print("load original baseline module")
        Model = Baseline()
    elif modeltype == "normal128":
        print("load normal128 model")
        Model = Baseline128(mode="concat")
    elif modeltype == "normal_light":
        print("load light extraction model")
        Model = Baseline_light()
    elif modeltype == "big":
        print("load big baseline module")
        Model = BigBaseline()
    elif modeltype == "EDSR_pretrained_baseline":
        print("load EDSR_pretrained_baseline baseline")
        Model = EDSR_baseline()
        Model.load_pretrained_model()
    else:
        print("load small baseline module")
        Model = Baseline_small()

    return Model

def loadshowmodel(modeltype):
    if modeltype == "normal":
        testmodel = Baseline_show()
    elif modeltype == "normal_concat" or modeltype == "normal_cosine_concat":
        print("load concat baseline module")
        testmodel = Baseline_show(mode="concat")
    elif modeltype == "big":
        print("load big baseline module")
        testmodel = BigBaseline_show()
    elif modeltype == "normal128":
        testmodel = Baseline128_show(mode="concat")
    elif modeltype == "EDSR_pretrained_baseline":
        print("load EDSR_pretrained_baseline baseline")
        print("load EDSR_Show")
        testmodel = EDSR_baseline_show()
    else:
        print("load small baseline module")
        testmodel = Baseline_small_show()
    return testmodel

if __name__ == "__main__":

    offset1= np.load("Network_patches/offset/offset_deformconv1.npy")
    offset1 = saveoffset(offset1)
    print(offset1.shape)
    print(offset1[40][40])

