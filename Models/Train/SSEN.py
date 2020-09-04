import torch.nn as nn
import torch
from Models.Train.Dynamic_offset_estimator import Dynamic_offset_estimator
from mmcv.ops.deform_conv import DeformConv2d

class SSEN(nn.Module):
    def __init__(self,in_channels):
        super(SSEN, self).__init__()
        self.deformblock1 = Deformable_Conv_Block(input_channels= in_channels)
        self.deformblock2 = Deformable_Conv_Block(input_channels= in_channels)
        self.deformblock3 = Deformable_Conv_Block(input_channels=in_channels)

    def forward(self,lr_batch, init_hr_batch):

        hr_out1 = self.deformblock1(lr_batch, init_hr_batch)
        hr_out2 = self.deformblock2(lr_batch, hr_out1)
        hr_out3 = self.deformblock3(lr_batch, hr_out2)
        return hr_out3

class Deformable_Conv_Block(nn.Module):
    def __init__(self,input_channels):
        super(Deformable_Conv_Block, self).__init__()
        self.offset_estimator = Dynamic_offset_estimator(input_channelsize=input_channels*2)
        self.offset_conv = nn.Conv2d(in_channels=input_channels * 2, out_channels=1 * 2 * 9, kernel_size=3, padding=1, bias=False)

        self.deformconv = DeformConv2d(in_channels=input_channels,out_channels=input_channels, kernel_size=3, padding = 1,  bias=False)

    def forward(self,lr_features, hr_features):
        input_offset = torch.cat((lr_features,hr_features),dim=1)
        
        estimated_offset = self.offset_estimator(input_offset)
        estimated_offset = self.offset_conv(estimated_offset)
        output = self.deformconv( x = hr_features, offset = estimated_offset)

        return output
