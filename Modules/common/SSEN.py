import torch.nn as nn

from Modules.common.DeformableBlock import DeformableConvBlock,DeformableConvBlock_show
from utils import showpatch

class SSEN(nn.Module):
    def __init__(self,in_channels,mode):
        super(SSEN, self).__init__()
        self.deformblock1 = DeformableConvBlock(input_channels= in_channels,mode=mode)
        self.deformblock2 = DeformableConvBlock(input_channels= in_channels,mode=mode)
        self.deformblock3 = DeformableConvBlock(input_channels=in_channels,mode=mode)

    def forward(self,lr_batch, init_hr_batch):

        hr_out1 = self.deformblock1(lr_batch, init_hr_batch)
        hr_out2 = self.deformblock2(lr_batch, hr_out1)
        hr_out3 = self.deformblock3(lr_batch, hr_out2)
        return hr_out3

class SSEN_show(nn.Module):
    def __init__(self,in_channels,mode):
        super(SSEN_show, self).__init__()
        self.deformblock1 = DeformableConvBlock_show(input_channels= in_channels,mode = mode)
        self.deformblock2 = DeformableConvBlock_show(input_channels= in_channels, mode = mode)
        self.deformblock3 = DeformableConvBlock_show(input_channels=in_channels, mode = mode)

    def forward(self,lr_batch, init_hr_batch, modelname, showmode = False):

        hr_out1 = self.deformblock1(lr_batch, init_hr_batch,showmode = showmode,num_block = 1, modelname = modelname)
        hr_out2 = self.deformblock2(lr_batch, hr_out1, showmode = showmode, num_block = 2, modelname = modelname)
        hr_out3 = self.deformblock3(lr_batch, hr_out2, showmode = showmode, num_block = 3, modelname = modelname )

        if showmode:
            showpatch(hr_out1, foldername="extracted_features_by_deformconv1", modelname=modelname)
            showpatch(hr_out2, foldername="extracted_features_by_deformconv2", modelname=modelname)
            showpatch(hr_out3, foldername="extracted_features_by_deformconv3", modelname=modelname)

        return hr_out3