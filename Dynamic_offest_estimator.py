import torch.nn as nn
import torch

class Deformable_Convolution(nn.Module):
    def __init__(self):
        super(Deformable_Convolution, self).__init__()

class Dynamic_offset_estimator(nn.Module):
    def __init__(self,input_channelsize):
        super(Dynamic_offset_estimator, self).__init__()
        self.inputchannel = input_channelsize

        self.halfscale_pre = self.pre_nonlocal_block(input_channelsize)
        self.halfscale_NLblock = Nonlocal_block(input_channelsize=64)
        self.halfscale_post = self.post_nonlocal_block()

        self.quarterscale_pre = self.pre_nonlocal_block(input_channelsize)
        self.quarterscale_NLblock = Nonlocal_block(input_channelsize=64)
        self.quarterscale_post = self.post_nonlocal_block()

        self.octascale_pre = self.pre_nonlocal_block(input_channelsize)
        self.octascale_NLblock = Nonlocal_block(input_channelsize=64)
        self.octascale_post = self.post_nonlocal_block()


    def pre_nonlocal_block(self,input_channelsize):
        layers = []
        layers.append(nn.Conv2d(in_channels=input_channelsize, out_channels=64, kernel_size=3, stride=2, padding=1, bias = False))
        layers.append(nn.LeakyReLU(inplace=True))

        pre_model = nn.Sequential(*layers)
        return pre_model

    def post_nonlocal_block(self):
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,padding=1, bias= False))
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