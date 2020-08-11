import torch.nn as nn
import torch

class Dynamic_offset_estimator(nn.Module):
    def __init__(self,input_channelsize):
        super(Dynamic_offset_estimator, self).__init__()
        self.downblock1 = self.downsample_block(input_channelsize)
        self.downblock2 = self.downsample_block(64)
        self.downblock3 = self.downsample_block(64)

        self.attentionblock1 = Nonlocal_block(input_channelsize=64)
        self.attentionblock2 = Nonlocal_block(input_channelsize=64)
        self.attentionblock3 = Nonlocal_block(input_channelsize=64)

        self.upblock1 = self.upsample_block()
        self.upblock2 = self.upsample_block()
        self.upblock3 = self.upsample_block()

    def forward(self,x):
        halfscale_feature = self.downblock1(x)
        quarterscale_feature = self.downblock2(halfscale_feature)
        octascale_feature = self.downblock3(quarterscale_feature)

        octascale_NLout = self.attentionblock1(octascale_feature)
        octascale_NLout = octascale_NLout + octascale_feature
        octascale_upsampled = self.upblock1(octascale_NLout)

        quarterscale_NLout = self.attentionblock2(octascale_upsampled)
        quarterscale_NLout = quarterscale_NLout + quarterscale_feature
        quarterscale_upsampled = self.upblock2(quarterscale_NLout)

        halfscale_NLout = self.attentionblock3(quarterscale_upsampled)
        halfscale_NLout = halfscale_NLout + halfscale_feature
        halfscale_upsampled = self.upblock3(halfscale_NLout)

        return halfscale_upsampled

    def downsample_block(self, input_channelsize):
        layers = []
        layers.append(
            nn.Conv2d(in_channels=input_channelsize, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))

        pre_model = nn.Sequential(*layers)
        return pre_model

    def upsample_block(self):
        layers = []
        layers.append(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))

        post_model = nn.Sequential(*layers)
        return post_model


class Nonlocal_block(nn.Module):
    def __init__(self,input_channelsize , mode = 'EmbeddedG', dimension = 2):
        super(Nonlocal_block, self).__init__()
        self.mode = mode
        self.dimension = dimension
        self.bottleneck_channel = input_channelsize//2

        self.thetalayer = nn.Conv2d(in_channels=input_channelsize, out_channels=self.bottleneck_channel, kernel_size=1, bias = False)
        self.philayer = nn.Conv2d(in_channels=input_channelsize, out_channels=self.bottleneck_channel, kernel_size=1, bias = False)

        self.glayer = nn.Conv2d(in_channels=input_channelsize, out_channels=self.bottleneck_channel, kernel_size=1, bias = False)

        self.lastlayer = nn.Conv2d(in_channels=self.bottleneck_channel , out_channels=input_channelsize, kernel_size=1, bias=False)

    def forward(self,x):
        #x= torch.Tensor(x)

        batch_size = x.size(0)

        theta_output = self.thetalayer(x)
        theta_output = theta_output.view(batch_size, self.bottleneck_channel,-1)
        theta_output = theta_output.view((0,2,1))  #convert H x W X channels to HW X channels matrix

        phi_output = self.philayer(x)
        phi_output = phi_output.view(batch_size, self.bottleneck_channel, -1)

        g_output = self.glayer(x)
        g_output = g_output.view(batch_size, self.bottleneck_channel, -1)
        g_output = g_output.permute((0,2,1))

        compressed_matrix = torch.matmul(theta_output,phi_output)
        compressed_matrix = nn.Softmax2d(compressed_matrix)

        third_input = torch.matmul(compressed_matrix,g_output)
        third_input = third_input.permute((0,2,1)).contiguous()
        third_input = third_input.view(batch_size,self.bottleneck_channel,x.size(2),x.size(3))

        final_output = self.lastlayer(third_input)
        residual_output = final_output + x

        return residual_output