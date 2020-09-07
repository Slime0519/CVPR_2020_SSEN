import torch
import torch.nn as nn
from Models.Train.SSEN import SSEN
from Models.Train_utils import make_residual_block, make_downsampling_network

class BigBaseline(nn.Module):
    def __init__(self, num_channel = 256):
        super(BigBaseline, self).__init__()
        self.feature_extractor = make_residual_block(blocknum=5, input_channel=3, output_channel=64)

        #referenced by EDVR paper implementation code
        #https://github.com/xinntao/EDVR/blob/master/basicsr/models/archs/edvr_arch.py line 251
        self.downsampling_network = make_downsampling_network(layernum=2, in_channels=3, out_channels=256)
        self.lrfeature_scaler1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding = 1, bias=False)
        self.lrfeature_scaler2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding = 1, bias= False)
        self.feature_extractor = make_residual_block(blocknum=5, input_channel=256, output_channel=256)
        self.SSEN_Network = SSEN(in_channels=num_channel)

        self.preprocessing_residual_block = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False)
        self.residual_blocks = make_residual_block(blocknum=32, input_channel=256, output_channel=256)

        self.upscaling_4x = nn.Sequential(
            nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding=1, bias = False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding=1, bias = False),
            nn.PixelShuffle(2),
        )

        self.outconv = nn.Conv2d(in_channels=num_channel, out_channels=3, kernel_size=3, padding=1, bias=False)


    def forward(self,input_lr, ref_input):

        ref_input = self.downsampling_network(ref_input)
        input_lr = self.lrfeature_scaler1(input_lr)
        input_lr = self.lrfeature_scaler2(input_lr)

        lr_feature_out = self.feature_extractor(input_lr)
        ref_feature_out = self.feature_extractor(ref_input)
        SSEN_out = self.SSEN_Network(lr_batch = lr_feature_out ,init_hr_batch = ref_feature_out,showmode = showmode)
        residual_input = torch.cat((lr_feature_out, SSEN_out), dim = 1)
        residual_input_scaled = self.preprocessing_residual_block(residual_input)
        out = self.residual_blocks(residual_input_scaled)
        out = torch.add(out,lr_feature_out)

        out = self.upscaling_4x(out)
        out = self.outconv(out)
        return out


if __name__ == "__main__":
    testmodel = BigBaseline(num_channel=256)
    model_layerlist = (list(testmodel.children()))
    print(model_layerlist)
   # torchsummary.summary(testmodel,(3,160,160),(3,160,160),device='cpu')