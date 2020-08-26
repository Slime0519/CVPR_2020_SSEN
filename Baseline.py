import torch
import torch.nn as nn
from SSEN import SSEN
from utils import showpatch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
import torchsummary


def make_residual_block(blocknum=32, input_channel = 64, output_channel = 64):
    residual_layers = []
    residual_layers.append(residual_block(input_channel=input_channel, output_channel = output_channel))
    for i in range(blocknum-1):
        residual_layers.append(residual_block(input_channel=output_channel, output_channel = output_channel))
    blockpart_model = nn.Sequential(*residual_layers)
    return blockpart_model

def make_downsampling_network(layernum = 2, in_channels = 3, out_channels = 64):
    layers = []
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, bias=False, padding=1))
    for _ in range(layernum-1):
        layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, bias=False,padding=1))
    print(layers)
    model = nn.Sequential(*layers)
    return model


class Baseline(nn.Module):
    def __init__(self, num_channel = 64):
        super(Baseline, self).__init__()
        # referenced by EDVR paper implementation code
        # https://github.com/xinntao/EDVR/blob/master/basicsr/models/archs/edvr_arch.py line 251
        self.downsampling_network = make_downsampling_network(layernum=2, in_channels=3, out_channels=64)
        self.lrfeature_scaler = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, bias=False)
     #   self.lrfeature_scaler2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False)
        self.feature_extractor = make_residual_block(blocknum=5, input_channel=64, output_channel=64)

       # self.conv1 = nn.Conv2d(in_channels=3,out_channels=num_channel, kernel_size=3, padding=1, bias=False)

        #self.feature_extractor = Feature_extractor_in_SSEN(input_channel=3, output_channel=num_channel)
        self.SSEN_Network = SSEN(in_channels=num_channel)

        self.preprocessing_residual_block = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False)
        self.residual_blocks = make_residual_block(blocknum=16, input_channel=64, output_channel=64)

        self.upscaling_4x = nn.Sequential(
            nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding=1, bias = False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding=1, bias = False),
            nn.PixelShuffle(2),
        )

        self.outconv = nn.Conv2d(in_channels=num_channel, out_channels=3, kernel_size=3, padding=1, bias=False)


    def forward(self,input_lr, ref_input , showmode = False):
        ref_input = self.downsampling_network(ref_input)
        input_lr = self.lrfeature_scaler(input_lr)
        lr_feature_out = self.feature_extractor(input_lr)
        ref_feature_out = self.feature_extractor(ref_input)

        SSEN_out = self.SSEN_Network(lr_batch = lr_feature_out ,init_hr_batch = ref_feature_out, showmode=showmode)
        residual_input = torch.cat((lr_feature_out, SSEN_out), dim = 1)
        residual_input_scaled = self.preprocessing_residual_block(residual_input)
        out = self.residual_blocks(residual_input_scaled)
        out = torch.add(out,lr_feature_out)

        if showmode:
            showpatch(lr_feature_out,foldername="extracted_features_lr_image")
            showpatch(ref_feature_out,foldername="extracted_features_ref_image")
            showpatch(out, foldername="features_after_reconstruction_blocks")

        out = self.upscaling_4x(out)
        out = self.outconv(out)
        return out

class residual_block(nn.Module):
    def __init__(self, input_channel = 256, output_channel = 256):
        super(residual_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=input_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=input_channel,out_channels=output_channel, kernel_size=3, padding=1, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out *= 0.1
        out = torch.add(out,x)

        return out

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


if __name__ == "__main__":
    testmodel = Baseline(num_channel=256)
    model_layerlist = (list(testmodel.children()))
    print(model_layerlist)
   # torchsummary.summary(testmodel,(3,160,160),(3,160,160),device='cpu')