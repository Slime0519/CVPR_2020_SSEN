import torch
import Data_gen
from Baseline import Baseline
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as pltimage
import numpy as np
import os
import cv2
from PIL import Image

savedir = "Result_image"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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
    testset_dirpath = "CUFED_SRNTT/CUFED5"

    model_dirpath = "Trained_model"
    model_epoch = 800

    testmodel = Baseline().to(device)
    Test_Dataset = Data_gen.Dataset_Test(dirpath=testset_dirpath,upscale_factor=4, mode = "XH")
    Test_Dataloader = DataLoader(dataset=Test_Dataset, shuffle=False, batch_size=1, num_workers=0)

    testmodel.load_state_dict(
        torch.load(os.path.join(model_dirpath,"Model_epoch{}.pth".format(model_epoch))))
    testmodel = testmodel.to(device)
    testmodel.eval()

    for i, (input, target, refimage) in enumerate(Test_Dataloader):
        # if not i == 33:
        #    continue
        input, refimage = input.to(device), refimage.to(device)
        output = testmodel(input,refimage)
        output_image = np.array(output.cpu().detach())
        output_image = output_image.squeeze()
        regularized_output_image = regularization_image(output_image)
        regularized_output_image = (regularized_output_image * 255).astype(np.uint8)

        target_image = np.array(target)
        target_image = np.squeeze(target_image)
        target_image = regularization_image(target_image)
        PSNR = getPSNR(regularized_output_image, target_image)
        print("PSNR : {}".format(PSNR))
        input_temp = np.array(input.cpu().detach())
        input_bicubic = cv2.resize(np.transpose(np.squeeze(input_temp), (1, 2, 0)), dsize=(0, 0), fx=4, fy=4,
                                   interpolation=cv2.INTER_CUBIC)
        regularized_input_image = regularization_image(input_bicubic)
        regularized_input_image = (regularized_input_image * 255).astype(np.uint8)

        regularized_output_image = np.transpose(regularized_output_image, (1, 2, 0))
        target_image = np.transpose(target_image,(1,2,0))
        target_image = (target_image*255).astype(np.uint8)
      #  plt.imshow(regularized_input_image)
      #  plt.show()
       # plt.imshow(regularized_output_image)
      #  plt.show()
      #  plt.imshow(target_image)
      #  plt.show()

        # PNG Image 저장
        PIL_Input_Image = Image.fromarray(regularized_input_image)
        #PIL_Input_Image.save("Result_image/bicubic/epoch{}_image{}.png".format(model_epoch,i))
        PIL_Input_Image.save("Result_image/input/image{}.png".format(i))  # save large size image

        PIL_output_Image = Image.fromarray(regularized_output_image)
        # PIL_output_Image.save("Result_image/srgan/epoch{}_image{}.png".format(model_epoch, i))
        PIL_output_Image.save("Result_image/output/image{}.png".format(i))

        PIL_target_Image = Image.fromarray(target_image)
        # PIL_output_Image.save("Result_image/srgan/epoch{}_image{}.png".format(model_epoch, i))
        PIL_target_Image.save("Result_image/target/image{}.png".format(i))




