import torch
import torch.nn as nn
from Modules.common.SSEN import SSEN
from Modules.Model_utils import make_residual_block, make_downsampling_network
from utils import showpatch

class Baseline_small(nn.Module):
    def __init__(self, num_channel = 64):
        super(Baseline_small, self).__init__()
        # referenced by EDVR paper implementation code
        # https://github.com/xinntao/EDVR/blob/master/basicsr/models/archs/edvr_arch.py line 251
        self.downsampling_network = make_downsampling_network(layernum=2, in_channels=3, out_channels=32)
        self.lrfeature_scaler = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.feature_extractor = make_residual_block(blocknum=5, input_channel=32, output_channel=32)
        self.SSEN_Network = SSEN(in_channels=32)

        self.residual_blocks = make_residual_block(blocknum=16, input_channel=64, output_channel=64)

        self.extracted_lr_scaler = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0, bias=False)
        self.upscaling_4x = nn.Sequential(
            nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding=1, bias = False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding=1, bias = False),
            nn.PixelShuffle(2),
        )

        self.outconv = nn.Conv2d(in_channels=num_channel, out_channels=3, kernel_size=3, padding=1, bias=False)


    def forward(self,input_lr, ref_input):

        ref_input = self.downsampling_network(ref_input)
        input_lr = self.lrfeature_scaler(input_lr)

        lr_feature_out = self.feature_extractor(input_lr)
        ref_feature_out = self.feature_extractor(ref_input)

        SSEN_out = self.SSEN_Network(lr_batch = lr_feature_out ,init_hr_batch = ref_feature_out)
        residualblock_input = torch.cat((lr_feature_out, SSEN_out), dim = 1)

        lr_feature_scaled = self.extracted_lr_scaler(lr_feature_out)
        out = self.residual_blocks(residualblock_input)
        out = torch.add(out,lr_feature_scaled)

        out = self.upscaling_4x(out)
        out = self.outconv(out)
        return out


class Baseline_small_show(nn.Module):
    def __init__(self, num_channel = 64):
        super(Baseline_small_show, self).__init__()
        # referenced by EDVR paper implementation code
        # https://github.com/xinntao/EDVR/blob/master/basicsr/models/archs/edvr_arch.py line 251
        self.downsampling_network = make_downsampling_network(layernum=2, in_channels=3, out_channels=32)
        self.lrfeature_scaler = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.feature_extractor = make_residual_block(blocknum=5, input_channel=32, output_channel=32)
        self.SSEN_Network = SSEN_show(in_channels=32, mode= "small")

        self.residual_blocks = make_residual_block(blocknum=16, input_channel=64, output_channel=64)

        self.extracted_lr_scaler = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0, bias=False)
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
        residualblock_input = torch.cat((lr_feature_out, SSEN_out), dim = 1)

        lr_feature_scaled = self.extracted_lr_scaler(lr_feature_out)
        out = self.residual_blocks(residualblock_input)
        out = torch.add(out,lr_feature_scaled)

        if showmode:
            showpatch(lr_feature_out,foldername="extracted_features_lr_image")
            showpatch(ref_feature_out,foldername="extracted_features_ref_image")
            showpatch(out, foldername="features_after_reconstruction_blocks")

        out = self.upscaling_4x(out)
        out = self.outconv(out)
        return out

if __name__ == "__main__":
    testmodel = Baseline_small(num_channel=256)
    model_layerlist = (list(testmodel.children()))
    print(model_layerlist)
   # torchsummary.summary(testmodel,(3,160,160),(3,160,160),device='cpu')