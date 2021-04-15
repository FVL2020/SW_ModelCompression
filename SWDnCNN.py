import torch.nn as nn
from SWConv import Conv2dSW
from SWConvF import Conv2dSWF


class SWDnCNN(nn.Module):
    def __init__(self, channels, features=64, num_of_layers=17):
        super(SWDnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(Conv2dSW(in_channels=channels, out_channels=features, kernel_radius=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for num in range(num_of_layers - 2):
            if num==0 or num==2 or num==4 or num==6 or num==8 or num==10 or num==12 or num==14:
                layers.append(Conv2dSWF(in_channels=features, kernel_radius=1, dilation=1, bias=False))
            elif num==1 or num==13:
                layers.append(Conv2dSWF(in_channels=features, kernel_radius=1, dilation=2, bias=False))
            elif num==3 or num==11:
                layers.append(Conv2dSWF(in_channels=features, kernel_radius=1, dilation=3, bias=False))
            elif num==5 or num==9:
                layers.append(Conv2dSWF(in_channels=features, kernel_radius=1, dilation=4, bias=False))
            elif num==7:
                layers.append(Conv2dSWF(in_channels=features, kernel_radius=1, dilation=5, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=(1, 1), bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.swdncnn = nn.Sequential(*layers)
        self.Attn = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

    def forward(self, x):
        out = self.swdncnn(x)
        Attn = self.Attn(out)
        out = out*Attn
        return out