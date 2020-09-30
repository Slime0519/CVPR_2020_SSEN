import torch
import torch.nn as nn

from Modules.common.Dynamic_offset_estimator import Dynamic_offset_estimator, Dynamic_offset_estimator_concat
from mmcv.ops.deform_conv import DeformConv2d
from utils import saveoffset, showpatch

class DeformableConvBlock(nn.Module):
    def __init__(self, input_channels, mode):
        super(DeformableConvBlock, self).__init__()
        if mode == "concat":
            self.offset_estimator = Dynamic_offset_estimator_concat(input_channelsize=input_channels * 2)
        else:
            self.offset_estimator = Dynamic_offset_estimator(input_channelsize=input_channels * 2)
        self.offset_conv = nn.Conv2d(in_channels=input_channels * 2, out_channels=1 * 2 * 9, kernel_size=3, padding=1,
                                     bias=True)

        self.deformconv = DeformConv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3,
                                       padding=1, bias=True)

    def forward(self, lr_features, hr_features):
        input_offset = torch.cat((lr_features, hr_features), dim=1)

        estimated_offset = self.offset_estimator(input_offset)
        estimated_offset = self.offset_conv(estimated_offset)
        output = self.deformconv(x=hr_features, offset=estimated_offset)

        return output


class DeformableConvBlock_show(nn.Module):
    def __init__(self, input_channels, mode):
        super(DeformableConvBlock_show, self).__init__()
        if mode == "concat":
            self.offset_estimator = Dynamic_offset_estimator_concat(input_channelsize=input_channels * 2)
        else:
            self.offset_estimator = Dynamic_offset_estimator(input_channelsize=input_channels * 2)
        self.offset_conv = nn.Conv2d(in_channels=input_channels * 2, out_channels=1 * 2 * 9, kernel_size=3, padding=1,
                                     bias=False)

        self.deformconv = DeformConv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3,
                                       padding=1, bias=False)

    def forward(self, lr_features, hr_features, modelname, showmode=False, num_block=None):
        input_offset = torch.cat((lr_features, hr_features), dim=1)

        estimated_offset = self.offset_estimator(input_offset)

        if showmode:
            showpatch(estimated_offset, foldername="DOE_output{}".format(num_block), modelname=modelname)
            showpatch(input_offset, foldername="offsetinput{}".format(num_block), modelname=modelname)

        estimated_offset = self.offset_conv(estimated_offset)

        if showmode:
            # showpatch(estimated_offset,foldername="extracted_offset_deformconv{}".format(num_block))
            saveoffset(estimated_offset, modelname=modelname, foldername="resultoffset_deformconv{}".format(num_block),
                       istensor=True)
        output = self.deformconv(x=hr_features, offset=estimated_offset)

        return output