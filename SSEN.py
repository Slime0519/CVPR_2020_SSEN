import torch.nn as nn
import torch

from Dynamic_offset_estimator import Dynamic_offset_estimator
from torch_deform_conv.layers import ConvOffset2D

class SSEN(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(SSEN, self).__init__()



class Deformable_Conv_Block(nn.Module):
    def __init__(self,input_channels):
        self.offset_estimator = Dynamic_offset_estimator(input_channelsize=input_channels)
        self.offset_generator = ConvOffset2D(filters = input_channels)

        self.conv1 = nn.Conv2d(input_channels, out_channels=2*input_channels, kernel_size=3, padding='same', bias = False)

    def forward(self,lr_features, hr_features):
        input_offset = torch.cat((lr_features,hr_features),dim=0)
        estimated_offset = self.offset_estimator(input_offset)
        gend_offset = self.offset_generator(hr_features, estimated_offset)

        output = self.conv1(gend_offset)

        return output
