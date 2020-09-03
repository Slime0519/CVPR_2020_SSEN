import torch.nn as nn
import torch
from non_local_embedded_gaussian import NONLocalBlock2D

class Dynamic_offset_estimator(nn.Module):
    def __init__(self,input_channelsize):
        super(Dynamic_offset_estimator, self).__init__()
        self.downblock1 = self.downsample_block(input_channelsize)
        self.downblock2 = self.downsample_block(64)
        self.downblock3 = self.downsample_block(64)
        """
        self.attentionblock1 = Nonlocal_block(input_channelsize=64)
        self.attentionblock2 = Nonlocal_block(input_channelsize=64)
        self.attentionblock3 = Nonlocal_block(input_channelsize=64)
        """
        self.attentionblock1 = NONLocalBlock2D(in_channels=64)
        self.attentionblock2 = NONLocalBlock2D(in_channels=64)
        self.attentionblock3 = NONLocalBlock2D(in_channels=64)

        self.upblock1 = self.upsample_block()
        self.upblock2 = self.upsample_block()
        self.upblock3 = self.upsample_block()

        self.channelscaling_block = nn.Conv2d(in_channels= 64, out_channels=input_channelsize, kernel_size=3, padding=1, bias=False)

    def forward(self,x):
        halfscale_feature = self.downblock1(x)
        quarterscale_feature = self.downblock2(halfscale_feature)
        octascale_feature = self.downblock3(quarterscale_feature)

        octascale_NLout = self.attentionblock1(octascale_feature)
        octascale_NLout = torch.add(octascale_NLout, octascale_feature)
       # print("octascale : {}".format(octascale_NLout.shape))
        octascale_upsampled = self.upblock1(octascale_NLout)
      #  print("octascale_up : {}".format(octascale_upsampled.shape))

        quarterscale_NLout = self.attentionblock2(octascale_upsampled)
        quarterscale_NLout = quarterscale_NLout + quarterscale_feature
        quarterscale_upsampled = self.upblock2(quarterscale_NLout)

        halfscale_NLout = self.attentionblock3(quarterscale_upsampled)
        halfscale_NLout = halfscale_NLout + halfscale_feature
        halfscale_upsampled = self.upblock3(halfscale_NLout)
     #   print("offset size : {}".format(halfscale_upsampled.shape))

        out = self.channelscaling_block(halfscale_upsampled)
        return out

    def downsample_block(self, input_channelsize):
        layers = []
        layers.append(
            nn.Conv2d(in_channels=input_channelsize, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))

        pre_model = nn.Sequential(*layers)
        return pre_model

    def upsample_block(self, in_odd = True):
        layers = []
        
        if in_odd:
            layers.append(
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
        else :
            layers.append(
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False))
        
#        layers.append(
 #           nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False))
  #      layers.append(nn.LeakyReLU(inplace=True))

        post_model = nn.Sequential(*layers)
        return post_model

"""
class Nonlocal_block(nn.Module):
    def __init__(self,input_channelsize , mode = 'EmbeddedG', dimension = 2):
        super(Nonlocal_block, self).__init__()
        self.mode = mode
        self.dimension = dimension
        self.bottleneck_channel = input_channelsize//2

        self.thetalayer = nn.Conv2d(in_channels=input_channelsize, out_channels=self.bottleneck_channel, kernel_size=1, bias = False)
        self.philayer = nn.Conv2d(in_channels=input_channelsize, out_channels=self.bottleneck_channel, kernel_size=1, bias = False)

        self.glayer = nn.Conv2d(in_channels=input_channelsize, out_channels=self.bottleneck_channel, kernel_size=1, bias = False)

        self.softmax = nn.Softmax()
        self.lastlayer = nn.Conv2d(in_channels=self.bottleneck_channel , out_channels=input_channelsize, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=input_channelsize)
    def forward(self,x):
        #x= torch.Tensor(x)

        batch_size = x.size(0)

        theta_output = self.thetalayer(x)
        theta_output = theta_output.view(batch_size, self.bottleneck_channel,-1)
        #print("theta output : {}".format(theta_output.shape))
        theta_output = theta_output.permute(0,2,1)  #convert H x W X channels to HW X channels matrix

        phi_output = self.philayer(x)
        phi_output = phi_output.view(batch_size, self.bottleneck_channel, -1)
        #print("phi output : {}".format(phi_output.shape))

        g_output = self.glayer(x)
        g_output = g_output.view(batch_size, self.bottleneck_channel, -1)
        g_output = g_output.permute((0,2,1))
        #print("goutput : {}".format(g_output.shape))
        compressed_matrix = torch.matmul(theta_output,phi_output)
        #print(compressed_matrix.shape)
        compressed_matrix = self.softmax(compressed_matrix)

        third_input = torch.matmul(compressed_matrix,g_output)
        third_input = third_input.permute((0,2,1)).contiguous()
        third_input = third_input.view(batch_size,self.bottleneck_channel,x.size(2),x.size(3))

        final_output = self.lastlayer(third_input)
        final_output = self.bn(final_output)
        residual_output = final_output + x

        return residual_output
"""
