import torch
import torch.nn as nn
import torchsummary

class Baseline(nn.Module):
    def __init__(self, num_channel = 256):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=num_channel, kernel_size=3, padding='same', bias=False)

        self.residual_blocks = self.make_residual_block(blocknum=32,num_channel= num_channel)

        self.upscaling_4x = nn.Sequential(
            nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding='same', bias = False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=num_channel, out_channels=4*num_channel, kernel_size=3, padding='same', bias = False),
            nn.PixelShuffle(2),
        )

        self.outconv = nn.Conv2d(in_channels=num_channel, out_channels=3, kernel_size=3, padding='same', bias=False)

    def make_residual_block(self, blocknum = 32, num_channel = 256):
        residual_layers = []

        for i in range(blocknum):
            residual_layers.append(Baseline_residual_block())
        blockpart_model = nn.Sequential(*residual_layers)
        return blockpart_model

    def forward(self,x):
        out = self.conv1(x)
        out = self.residual_blocks(out)
        out = self.upscaling_4x(out)
        out = self.outconv(out)

class Baseline_residual_block(nn.Module):
    def __init__(self, input_channel = 256, output_channel = 256):
        super(Baseline_residual_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=output_channel, kernel_size=3, padding='same', bias=False)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=output_channel, kernel_size=3, padding='same', bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.relu(self.conv1(x))
        out = nn.conv2(out)
        out *= 0.1
        out = torch.add(out,x)

        return out
