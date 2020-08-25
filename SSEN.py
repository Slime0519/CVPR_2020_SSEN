import torch.nn as nn
import torch
import numpy as np
from Dynamic_offset_estimator import Dynamic_offset_estimator
from torch_deform_conv.layers import ConvOffset2D
from mmcv.ops.deform_conv import DeformConv2d
from utils import showpatch

class SSEN(nn.Module):
    def __init__(self,in_channels):
        super(SSEN, self).__init__()
        self.deformblock1 = Deformable_Conv_Block(input_channels= in_channels)
        self.deformblock2 = Deformable_Conv_Block(input_channels= in_channels)
        self.deformblock3 = Deformable_Conv_Block(input_channels=in_channels)

    def forward(self,lr_batch, init_hr_batch, showmode = False):

        hr_out1 = self.deformblock1(lr_batch, init_hr_batch,showmode = showmode,num_block = 1)
        hr_out2 = self.deformblock2(lr_batch, hr_out1, showmode = showmode, num_block = 2)
        hr_out3 = self.deformblock3(lr_batch, hr_out2, showmode = showmode, num_block = 3)
        if showmode:
            showpatch(hr_out1, foldername="extracted_features_by_deformconv1")
            showpatch(hr_out2, foldername="extracted_features_by_deformconv2")
            showpatch(hr_out3, foldername="extracted_features_by_deformconv3")

        return hr_out3

class Deformable_Conv_Block(nn.Module):
    def __init__(self,input_channels):
        super(Deformable_Conv_Block, self).__init__()
        self.offset_estimator = Dynamic_offset_estimator(input_channelsize=input_channels*2)
        self.offset_conv = nn.Conv2d(in_channels=input_channels*2, out_channels=1*2*9,kernel_size =3 ,padding=1,bias = False)
        self.deformconv = DeformConv2d(in_channels=input_channels,out_channels=input_channels, kernel_size=3, padding=1, bias=False)

    def forward(self,lr_features, hr_features,showmode = False,num_block =None):
        input_offset = torch.cat((lr_features,hr_features),dim=1)

        estimated_offset = self.offset_estimator(input_offset)
        estimated_offset = self.offset_conv(estimated_offset)

        if showmode:
            showpatch(estimated_offset,foldername="extracted_offset_deformconv{}".format(num_block))
        #print(estimated_offset.shape)
        #print(hr_features.shape)
        output = self.deformconv( x = hr_features, offset = estimated_offset)

        return output
