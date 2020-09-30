from Modules.Model_utils import make_residual_block
import torch.nn as nn

class feature_extraction_network(nn.Module):
    def __init__(self,in_channels = 64, out_channels = 64):
        super(feature_extraction_network, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels = out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        return out

class downsampling_network(nn.Module):
    def __init__(self):
        super(downsampling_network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, stride=2, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, stride=2, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out




