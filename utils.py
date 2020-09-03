import PIL.Image as Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from Data_gen import Dataset_Test
from torch.utils.data import DataLoader

def showpatch(imagepatch, foldername=None, istensor = True):
    batchsize = imagepatch.shape[0]
    channelsize = imagepatch.shape[1]
    #print(imagepatch.shape)
    #print(batchsize)

    if istensor:
        imagepatch = np.array(imagepatch.cpu().detach())
    folderpath = os.path.join("Network_patches",foldername)
    
    print("start visulization {}, channelsize : {}".format(foldername,channelsize))

    if not os.path.isdir(folderpath):
        os.mkdir(folderpath)

    for index in range(batchsize):
        patches = imagepatch[index]
        for channel in range(channelsize//4):
            image = regularization_image(patches[channel])
            image = (image*255).astype(np.uint8)
            #PIL_Input_Image = Image.fromarray(image)
            #PIL_Input_Image.save(os.path.join(folderpath,"image{}.png".format(index)))
            plt.imshow(image, 'gray')
            plt.savefig(os.path.join(folderpath,"image{}.png".format(channel)))


def saveoffset(offsetbatch, foldername=None, istensor = False):
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


    #for i in range(offsetbatch.shape[0]):
     #   np.save(os.path.join(foldername, "offset_{}.npy".format(i)), offsetbatch[i])
    return offset_coord


def regularization_image(image):
    min = np.min(image)
    temp_image = image-min

    max = np.max(temp_image)
    temp_image = temp_image/max

    return temp_image


def getPSNR(image1, image2):
#    shape = image1.shape[0]

#    MSE = (image1-image2)**2
 #   MSE = np.sum(MSE)

   # MSE = MSE/(shape**2)
    MSE = (np.square(image1-image2)).mean(axis = None)

    PSNR = 20*np.log10(1/np.sqrt(MSE))
    return PSNR

def pointdot(img, coordtuple = (10,10)):
    dpi = 10

    # Set red pixel value for RGB image
    red = [1, 0, 0]
    height, width, bands = img.shape

    # Update figure size based on image size
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    figure = plt.figure(figsize=figsize)
    axes = figure.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    axes.axis('off')

    # Draw a red dot at pixel (62,62) to (66, 66)
    #for i in range(62, 67):
     #   for j in range(62, 67):
      #    img[i][j] = red
    img[coordtuple[0]:coordtuple[0]+5,coordtuple[1]:coordtuple[1]+5,:] = red
    # Draw the image
    axes.imshow(img, interpolation='nearest')
    plt.show()
    #figure.savefig("test.png", dpi=dpi, transparent=True)

#def pointdeformoffset():



if __name__ == "__main__":
    """
    array1 = np.random.rand(3,100,100)
    array1 = np.expand_dims(array1,axis=0)

    array2 = np.random.rand(3, 100, 100)
    array2 = np.expand_dims(array2,axis=0)
    array3 = np.concatenate((array1,array2),axis=0)
    print(array3.shape)
    showpatch(array3,foldername="test",istensor=False)
    
    testset_dirpath = "CUFED_SRNTT/CUFED5"
    Test_Dataset = Dataset_Test(dirpath=testset_dirpath, upscale_factor=4, mode="XH")
    Test_Dataloader = DataLoader(dataset=Test_Dataset, shuffle=False, batch_size=1, num_workers=0)

    for i,(input, target, ref) in enumerate(Test_Dataloader):
        if i == 0:
            continue
        inputarray = np.array(ref.cpu().detach())
        inputarray = np.transpose(np.squeeze(inputarray), (1, 2, 0))
        print(inputarray.shape)
        inputarray = regularization_image(inputarray)
        pointdot(inputarray)

        break
    """
    offset1= np.load("Network_patches/offset/offset_deformconv1.npy")
    offset1 = saveoffset(offset1)
    print(offset1.shape)
    print(offset1[40][40])

"""
    array1 = torch.zeros([2, 18, 40, 40], dtype=torch.float32)
    for i in range(9):
        array1[0, 2 * i:2 * (i + 1), :, :] = i
    for i in range(9):
        array1[1, 2 * i:2 * (i + 1), :, :] = 8 - i
    # print(array1[:,:,0,0])
    saveoffset(array1)
"""
