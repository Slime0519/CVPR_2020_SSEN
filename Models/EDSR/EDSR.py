import torch.nn as nn
import torch
from Models.Train_utils import make_residual_block

import torch.nn as nn
from torchsummary import summary
from utils import showpatch

class EDSR(nn.Module):
    def __init__(self, ):
        super(EDSR, self).__init__()
        self.relu = nn.ReLU(True)

       # self.sub_mean = common.MeanShift(args.rgb_range)
     #   self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        #m_head = [nn.Conv2d(in_channels=3, out_channels=64,kernel_size=3,padding=1, bias=True)]
        self.mhead = nn.Conv2d(in_channels=128, out_channels=64, kernel_size = 3, padding=1,bias = True)
        # define body module
        """
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        
        m_body.append(conv(n_feats, n_feats, kernel_size))
        """
        self.body = make_residual_block(blocknum=16,input_channel=64,output_channel=64)
        # define tail module
        
        """
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        """

        m_tail = [
            nn.Conv2d(in_channels=64, out_channels=4 * 64, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=True)
        ]



        #self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, lr_feature):
        x = self.mhead(x)
        res = self.body(x)
        res = torch.add(res,lr_feature)

        x = self.tail(res)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    print("load weight : {}".format(name))
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSR_show(nn.Module):
    def __init__(self, ):
        super(EDSR_show, self).__init__()
        self.relu = nn.ReLU(True)

        self.mhead = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.body = make_residual_block(blocknum=16, input_channel=64, output_channel=64)
        # define tail module
        m_tail = [
            nn.Conv2d(in_channels=64, out_channels=4 * 64, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=True)
        ]
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, lr_feature):
        x = self.mhead(x)
        res = self.body(x)
        res = torch.add(res, lr_feature)
        showpatch(res, foldername="features_after_reconstruction_blocks", modelname="EDSR")
        x = self.tail(res)
        return x


if __name__ == "__main__":
    model = EDSR()
    dict= torch.load("edsr_baseline_x2-1bc95232.pt")
    model.load_state_dict(dict,strict=False)
    summary(model, (128,160,160),device="cpu")

