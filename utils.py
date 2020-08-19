import PIL.Image as Image
import numpy as np
import os

def showpatch(imagepatch, foldername=None, istensor = True):
    batchsize = imagepatch.shape[0]

    if istensor:
        imagepatch = np.array(imagepatch.cpu().detach())
    patch = np.transpose(imagepatch,(0,2,3,1))
    folderpath = os.path.join("Network_patches",foldername)

    if not os.path.isdir(folderpath):
        os.mkdir(folderpath)

    for index in range(batchsize):
        image = patch[index]
        image = regularization_image(image)
        image = (image*255).astype(np.uint8)
        PIL_Input_Image = Image.fromarray(image)
        PIL_Input_Image.save(os.path.join(folderpath,"image{}.png".format(index)))

def regularization_image(image):
    min = np.min(image)
    temp_image = image-min

    max = np.max(temp_image)
    temp_image = temp_image/max

    return temp_image


def getPSNR(image1, image2):
    MSE = (image1-image2)**2
    MSE = np.sum(MSE)

    PSNR = 10*np.log10(1/MSE)
    return PSNR

if __name__ == "__main__":
    array1 = np.random.rand(3,100,100)
    array1 = np.expand_dims(array1,axis=0)

    array2 = np.random.rand(3, 100, 100)
    array2 = np.expand_dims(array2,axis=0)
    array3 = np.concatenate((array1,array2),axis=0)
    print(array3.shape)
    showpatch(array3,foldername="test",istensor=False)