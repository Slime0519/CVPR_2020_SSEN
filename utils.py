import PIL.Image as Image
import numpy as np
import os
import matplotlib.pyplot as plt

def showpatch(imagepatch, foldername=None, istensor = True):
    batchsize = imagepatch.shape[0]
    channelsize = imagepatch.shape[1]
    #print(imagepatch.shape)
    #print(batchsize)

    if istensor:
        imagepatch = np.array(imagepatch.cpu().detach())
    folderpath = os.path.join("Network_patches",foldername)

    if not os.path.isdir(folderpath):
        os.mkdir(folderpath)

    for index in range(batchsize):
        patches = imagepatch[index]
        for channel in range(channelsize):
            image = regularization_image(patches[channel])
            image = (image*255).astype(np.uint8)
            #PIL_Input_Image = Image.fromarray(image)
            #PIL_Input_Image.save(os.path.join(folderpath,"image{}.png".format(index)))
            plt.imshow(image, 'gray')
            plt.savefig(os.path.join(folderpath,"image{}.png".format(channel)))

def regularization_image(image):
    min = np.min(image)
    temp_image = image-min

    max = np.max(temp_image)
    temp_image = temp_image/max

    return temp_image


def getPSNR(image1, image2):
    shape = image1.shape[0]

#    MSE = (image1-image2)**2
 #   MSE = np.sum(MSE)
    
   # MSE = MSE/(shape**2)
    MSE = (np.square(image1-image2)).mean(axis = None)

    PSNR = 10*np.log10(255**2/MSE)
    return PSNR

if __name__ == "__main__":
    array1 = np.random.rand(3,100,100)
    array1 = np.expand_dims(array1,axis=0)

    array2 = np.random.rand(3, 100, 100)
    array2 = np.expand_dims(array2,axis=0)
    array3 = np.concatenate((array1,array2),axis=0)
    print(array3.shape)
    showpatch(array3,foldername="test",istensor=False)
