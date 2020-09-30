import torch
import Data_gen

from torch.utils.data import DataLoader
import argparse
import utils
from utils import regularization_image, getPSNR,regularize_testimage

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
parser.add_argument('--model_type', type = str, default="normal", help = "select model size")
parser.add_argument('--model_epoch', type = int, default = 700 , help = "pretrained model's epoch")
parser.add_argument('--showmode', type = str, default= None, help = "show patches of each levels" )

if __name__ == "__main__":

    opt = parser.parse_args()

    modeltype = opt.model_typ
    model_epoch = opt.model_epoch
    showmode = opt.showmode

    testset_dirpath = "CUFED_SRNTT/CUFED5"

    prefix_resultname = utils.getprefixname(modeltype)
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

    if showmode != "show":
        testmodel = utils.loadmodel(modeltype)
    else:
        testmodel = utils.loadshowmodel(modeltype)

    testmodel.to(device)

    Test_Dataset = Data_gen.Dataset_Test(dirpath=testset_dirpath,upscale_factor=4, mode = "XH")
    if modeltype == "EDSR":
       Test_Dataset = Data_gen.Dataset_Test(dirpath = testset_dirpath, upscale_factor =2 ,mode = "XH")

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

    testmodel.eval()
    PSNRarr = np.zeros(len(Test_Dataloader))

    for i, (input, target, refimage) in enumerate(Test_Dataloader):
        # if not i == 33:
        #    continue
        if showmode == "show" and i!= 53:
            continue;
        input, refimage = input.to(device), refimage.to(device)
#        output = testmodel(input,refimage, showmode = True)
        output = testmodel(input, refimage)
        
        regularized_output = regularize_testimage(output, istensor=True)
        regularized_target = regularize_testimage(target, istensor=False)
        regularized_ref = regularize_testimage(refimage, istensor=True)
        
        PSNR = getPSNR(regularized_output, regularized_target)

        print("PSNR : {}".format(PSNR))
        PSNRarr[i] = PSNR

        input_temp = np.array(input.cpu().detach())
        input_bicubic = cv2.resize(np.transpose(np.squeeze(input_temp), (1, 2, 0)), dsize=(0, 0), fx=2, fy=2,
                                   interpolation=cv2.INTER_CUBIC)
        regularized_input_image = regularization_image(input_bicubic)
        regularized_input_image = (regularized_input_image * 255).astype(np.uint8)
        
        regularized_ref = (regularized_ref * 255).astype(np.uint8)
        regularized_ref = np.transpose(regularized_ref,(1,2,0))
        
        regularized_target = (regularized_target * 255).astype(np.uint8)
        regularized_target = np.transpose(regularized_target, (1, 2, 0))

        regularized_output = (regularized_output * 255).astype(np.uint8)
        regularized_output = np.transpose(regularized_output, (1, 2, 0))


        # PNG Image 저장
        PIL_Input_image = Image.fromarray(regularized_input_image)
        PIL_Input_image.save(os.path.join(image_savepath,"input/image{}.png".format(i)))  # save large size image
        #PIL_Input_Image.save("Result_image/input/image{}.png".format(i))
        
        PIL_output_image = Image.fromarray(regularized_output)
        PIL_output_image.save(os.path.join(image_savepath,"output/image{}.png".format(i)))
        #PIL_Input_Image.save("Result_image/output/image{}.png".format(i))

        PIL_ref_image = Image.fromarray(regularized_ref)
        #PIL_Input_Image.save("Result_image/reference/image{}.png".format(i))
        PIL_ref_image.save(os.path.join(image_savepath,"reference/image{}.png".format(i)))

        PIL_target_image = Image.fromarray(regularized_target)
        #PIL_Input_Image.save("Result_image/target/image{}.png".format(i))
        PIL_target_image.save(os.path.join(image_savepath,"target/image{}.png".format(i)))

        np.save(os.path.join(image_savepath,"{}_PSNRlist.npy".format(prefix_resultname)),PSNRarr)
        if showmode == "show":
            break;

