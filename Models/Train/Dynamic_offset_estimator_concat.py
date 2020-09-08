import torch.nn as nn
import torch
from non_local_embedded_gaussian import NONLocalBlock2D

class Dynamic_offset_estimator_concat(nn.Module):
    def __init__(self,input_channelsize):
        super(Dynamic_offset_estimator_concat, self).__init__()
        self.downblock1 = self.downsample_block(input_channelsize)
        self.downblock2 = self.downsample_block(64)
        self.downblock3 = self.downsample_block(64)

        self.attentionblock1 = NONLocalBlock2D(in_channels=64)
        self.attentionblock2 = NONLocalBlock2D(in_channels=64)
        self.attentionblock3 = NONLocalBlock2D(in_channels=64)

        self.upblock1 = self.upsample_block(in_channels=128,out_channels=64)
        self.upblock2 = self.upsample_block(in_channels=128,out_channels=64)
        self.upblock3 = self.upsample_block(in_channels=128,out_channels=64)

        self.channelscaling_block = nn.Conv2d(in_channels= 64, out_channels=input_channelsize, kernel_size=3, padding=1, bias=False)

    def forward(self,x):
        halfscale_feature = self.downblock1(x)
        quarterscale_feature = self.downblock2(halfscale_feature)
        octascale_feature = self.downblock3(quarterscale_feature)

        octascale_NLout = self.attentionblock1(octascale_feature)
        octascale_NLout = torch.cat((octascale_NLout, octascale_feature),dim=1)
        octascale_upsampled = self.upblock1(octascale_NLout)

        quarterscale_NLout = self.attentionblock2(octascale_upsampled)
        quarterscale_NLout = torch.cat((quarterscale_NLout, quarterscale_feature),dim=1)
        quarterscale_upsampled = self.upblock2(quarterscale_NLout)

        halfscale_NLout = self.attentionblock3(quarterscale_upsampled)
        halfscale_NLout = torch.cat((halfscale_NLout,halfscale_feature),dim =1 )
        halfscale_upsampled = self.upblock3(halfscale_NLout)

        out = self.channelscaling_block(halfscale_upsampled)
        return out

    def downsample_block(self, input_channelsize):
        layers = []
        layers.append(
            nn.Conv2d(in_channels=input_channelsize, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))

        pre_model = nn.Sequential(*layers)
        return pre_model

    def upsample_block(self, in_odd = True, in_channels = 64, out_channels = 64):
        layers = []

        if in_odd:
            layers.append(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
        else :
            layers.append(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=0, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))

        post_model = nn.Sequential(*layers)
        return post_model
