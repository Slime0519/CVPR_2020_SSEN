import torch
import Data_gen
from Models.Train.Baseline import Baseline
from Models.Train.Baseline_big import BigBaseline
from torch.utils.data import DataLoader
import argparse
from Models.Train.Baseline_small import Baseline_small

from utils import regularization_image, getPSNR
import numpy as np
import os
import cv2
from PIL import Image
from collections import OrderedDict

savedir = "Result_image"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser(description="RefSR Network with SSEN Training module")
parser.add_argument('--model_size', type = str, default="normal", help = "select model size")
parser.add_argument('--model_epoch', type = int, default = 700 , help = "pretrained model's epoch")
parser.add_argument('--showmode', type = str, default= None, help = "show patches of each levels" )

if __name__ == "__main__":

    opt = parser.parse_args()

    Modelsize = opt.model_size
    model_epoch = opt.model_epoch
    showmode = opt.showmode

    if Modelsize == "normal":
        prefix_resultname = "normalModel"
    elif Modelsize == "normal_concat":
        prefix_resultname = "normalModel_concat"
    elif Modelsize == "big":
        prefix_resultname = "bigModel"
    else:
        prefix_resultname = "smallModel"

    testset_dirpath = "CUFED_SRNTT/CUFED5"

    model_dirpath = os.path.join("Trained_model",prefix_resultname)

    image_savepath = os.path.join(savedir,prefix_resultname,"epoch{}".format(model_epoch))

    if not os.path.isdir(os.path.join(savedir,prefix_resultname)):
        os.mkdir(os.path.join(savedir,prefix_resultname))
    if not os.path.isdir(image_savepath):
        os.mkdir(image_savepath)

    if not os.path.isdir(os.path.join(image_savepath,"input")):
        os.mkdir(os.path.join(image_savepath,"input"))
    if not os.path.isdir(os.path.join(image_savepath,"output")):
        os.mkdir(os.path.join(image_savepath,"output"))
    if not os.path.isdir(os.path.join(image_savepath,"reference")):
        os.mkdir(os.path.join(image_savepath,"reference"))
    if not os.path.isdir(os.path.join(image_savepath,"target")):
        os.mkdir(os.path.join(image_savepath,"target"))


    if Modelsize == "normal":
        print("load original baseline module")
        testmodel = Baseline()
        if showmode == "show":
            testmodel = Baseline_show()
    elif Modelsize == "normal_concat":
        print("load concat baseline module")
        testmodel = Baseline(mode="concat")
        if showmode == "show":
            testmodel = Baseline_show(mode = "concat")
    elif Modelsize == "big":
        print("load big baseline module")
        testmodel = BigBaseline()
        if showmode == "show":
            testmodel = BigBaseline_show()
    else :
        print("load small baseline module")
        testmodel = Baseline_small()
        if showmode == "show":
            testmodel = Baseline_small_show()
    
    testmodel.to(device)
    Test_Dataset = Data_gen.Dataset_Test(dirpath=testset_dirpath,upscale_factor=4, mode = "XH")
    Test_Dataloader = DataLoader(dataset=Test_Dataset, shuffle=False, batch_size=1, num_workers=0)

    # original saved file with DataParallel
    checkpoint = torch.load(os.path.join(model_dirpath,prefix_resultname+"_epoch{}.pth".format(model_epoch)))
    loadedmodel = checkpoint['model']
    state_dict =  loadedmodel.state_dict()
    # create new OrderedDict that does not contain `module.`

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    testmodel.load_state_dict(new_state_dict)


 #   testmodel.load_state_dict(
  #      torch.load(os.path.join(model_dirpath,"larger_Model_epoch{}.pth".format(model_epoch))))
    #testmodel = testmodel.to(device)
    testmodel.eval()
    PSNRarr = np.zeros(len(Test_Dataloader))

    for i, (input, target, refimage) in enumerate(Test_Dataloader):
        # if not i == 33:
        #    continue
        input, refimage = input.to(device), refimage.to(device)
#        output = testmodel(input,refimage, showmode = True)
        output = testmodel(input, refimage,showmode = True )

        output_image = np.array(output.cpu().detach())
        output_image = output_image.squeeze()
        regularized_output_image = regularization_image(output_image)
        
        target_image = np.array(target)
        target_image = np.squeeze(target_image)
        target_image = regularization_image(target_image)

        ref_image = np.array(refimage.cpu().detach())
        ref_image = ref_image.squeeze()
        regularized_ref_image = regularization_image(ref_image)

        PSNR = getPSNR(regularized_output_image, target_image)

        print("PSNR : {}".format(PSNR))
        PSNRarr[i] = PSNR

        input_temp = np.array(input.cpu().detach())
        input_bicubic = cv2.resize(np.transpose(np.squeeze(input_temp), (1, 2, 0)), dsize=(0, 0), fx=4, fy=4,
                                   interpolation=cv2.INTER_CUBIC)
        regularized_input_image = regularization_image(input_bicubic)
        regularized_input_image = (regularized_input_image * 255).astype(np.uint8)
        
        regularized_ref_image = (ref_image * 255).astype(np.uint8)
        regularized_ref_image = np.transpose(regularized_ref_image,(1,2,0))
        
        target_image = (target_image * 255).astype(np.uint8)
        target_image = np.transpose(target_image, (1, 2, 0))

        regularized_output_image = (regularized_output_image * 255).astype(np.uint8)
        regularized_output_image = np.transpose(regularized_output_image, (1, 2, 0))


        # PNG Image 저장
        PIL_Input_Image = Image.fromarray(regularized_input_image)
        PIL_Input_Image.save(os.path.join(image_savepath,"input/image{}.png".format(i)))  # save large size image
        #PIL_Input_Image.save("Result_image/input/image{}.png".format(i))
        
        PIL_output_Image = Image.fromarray(regularized_output_image)
        PIL_output_Image.save(os.path.join(image_savepath,"output/image{}.png".format(i)))
        #PIL_Input_Image.save("Result_image/output/image{}.png".format(i))

        PIL_ref_Image = Image.fromarray(regularized_ref_image)
        #PIL_Input_Image.save("Result_image/reference/image{}.png".format(i))
        PIL_ref_Image.save(os.path.join(image_savepath,"reference/image{}.png".format(i)))

        PIL_target_Image = Image.fromarray(target_image)
        #PIL_Input_Image.save("Result_image/target/image{}.png".format(i))
        PIL_target_Image.save(os.path.join(image_savepath,"target/image{}.png".format(i)))

        np.save(os.path.join(image_savepath,"{}_PSNRlist.npy".format(prefix_resultname)),PSNRarr)


