import torch.nn as nn
from Modules.EDSR_pretrained_baseline.Feature_extractor import feature_extraction_network
from Modules.EDSR_pretrained_baseline.EDSR import EDSR, EDSR_show
from Modules.common.SSEN import SSEN, SSEN_show
from utils import showpatch

import os
from torchsummary import summary
import torch

class EDSR_baseline(nn.Module):
    def __init__(self):
        super(EDSR_baseline, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample_module = nn.Conv2d(in_channels=3, out_channels=64, stride=2, kernel_size=3, padding=1, bias=True)
        self.lrfeature_ext1 =  nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.lrfeature_ext2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)

        self.feature_extraction_network = feature_extraction_network(in_channels=64)
        self.ssen = SSEN(in_channels = 64, mode = "add")
        self.EDSR = EDSR()

    def forward(self,lrinput,refinput):
        lr_feature = self.relu(self.lrfeature_ext1(lrinput))
        lr_feature = self.relu(self.lrfeature_ext2(lr_feature))
        lr_feature = self.feature_extraction_network(lr_feature)

        ref_feature = self.downsample_module(refinput)
        ref_feature = self.relu(ref_feature)
        ref_feature = self.feature_extraction_network(ref_feature)

        deform_feature = self.ssen(lr_batch = lr_feature, init_hr_batch = ref_feature)
        concated_feature = torch.cat((lr_feature, deform_feature), dim=1)
        reconstruct_image = self.EDSR(x = concated_feature, lr_feature = lr_feature)

        return reconstruct_image

    def load_pretrained_model(self):
        print(os.path.abspath('./edsr_baseline_x2.pt'))
        dict = torch.load('Modules/EDSR_pretrained_baseline/edsr_baseline_x2-1bc95232.pt')
        self.EDSR.load_state_dict(dict, strict=False)


class EDSR_baseline_show(nn.Module):
    def __init__(self):
        super(EDSR_baseline_show, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample_module = nn.Conv2d(in_channels=3, out_channels=64, stride=2, kernel_size=3, padding=1, bias=True)
        self.lrfeature_ext1 =  nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.lrfeature_ext2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)

        self.feature_extraction_network = feature_extraction_network(in_channels=64)
        self.ssen_show = SSEN_show(in_channels = 64, mode = "add")
        self.EDSR_show = EDSR_show()

    def forward(self,lrinput,refinput, showmode = True):
        lr_feature = self.relu(self.lrfeature_ext1(lrinput))
        lr_feature = self.relu(self.lrfeature_ext2(lr_feature))
        lr_feature = self.feature_extraction_network(lr_feature)

        ref_feature = self.downsample_module(refinput)
        ref_feature = self.relu(ref_feature)
        ref_feature = self.feature_extraction_network(ref_feature)

        if showmode:
            showpatch(lr_feature,foldername="extracted_features_lr_image", modelname="EDSR")
            showpatch(ref_feature,foldername="extracted_features_ref_image", modelname = "EDSR")

        deform_feature = self.ssen_show(lr_batch = lr_feature, init_hr_batch = ref_feature, showmode=showmode, modelname = "EDST")

        concated_feature = torch.cat((lr_feature, deform_feature), dim=1)
        reconstruct_image = self.EDSR_show(x = concated_feature, lr_feature = lr_feature)

        return reconstruct_image


if __name__ == "__main__":
    model = EDSR_baseline()
    summary(model, (3,160,160),device ="cpu")



