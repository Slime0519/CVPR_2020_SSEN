import torch.nn as nn
import torch

from Dynamic_offset_estimator import Dynamic_offset_estimator
from torch_deform_conv.layers import ConvOffset2D

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
        self.offset_generator = ConvOffset2D(filters = input_channels*2)

        self.conv1 = nn.Conv2d(64, out_channels=input_channels, kernel_size=3, padding=1, bias = False)

    def forward(self,lr_features, hr_features):
        print("lr feature shape : {}".format(lr_features.shape))
        print("hr feature shape : {}".format(hr_features.shape))
        input_offset = torch.cat((lr_features,hr_features),dim=1)

        estimated_offset = self.offset_estimator(input_offset)

        print("offset size : {}".format(estimated_offset.shape))
        gend_offset = self.offset_generator(hr_features, estimated_offset)
        print("generated offset size : {}".format(gend_offset.shape))
        output = self.conv1(gend_offset)

        return output
