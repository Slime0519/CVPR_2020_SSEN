import torch.nn as nn
from Models.EDSR.Feature_extractor import downsampling_network, feature_extraction_network
from Models.EDSR.EDSR import EDSR
from Models.Train.SSEN import SSEN

from torchsummary import summary
import torch

class EDSR_baseline(nn.Module):
    def __init__(self):
        super(EDSR_baseline, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample_module = downsampling_network()
        self.lrfeature_ext1 =  nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.lrfeature_ext2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)

        self.feature_extraction_network = feature_extraction_network(in_channels=64)
        self.ssen = SSEN(mode = "concat")
        self.EDSR = EDSR()

    def forward(self,lrinput,refinput):
        lr_feature = self.relu(self.lrfeature_ext1(lrinput))
        lr_feature = self.relu(self.lrfeature_ext2(lr_feature))
        lr_feature = self.feature_extraction_network(lr_feature)

        ref_feature = self.downsample_module(refinput)
        ref_feature = self.feature_extraction_network(ref_feature)

        deform_feature = self.ssen(lr_batch = lr_feature, init_hr_batch = ref_feature)
        concated_feature = torch.cat((lr_feature, deform_feature), dim=1)
        reconstruct_image = self.EDSR(x = concated_feature, lr_feature = lr_feature)

        return reconstruct_image

    def load_pretrained_model(self):
        dict = torch.load("edsr_baseline_x2-1bc95232.pt")
        self.EDSR.load_state_dict(dict, strict=False)

if __name__ == "__main__":
    model = EDSR_baseline()
    summary(model, (3,160,160),device ="cpu")



