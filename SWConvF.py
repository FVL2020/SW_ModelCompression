import torch
import torch.nn as nn


class Conv2dSWF(nn.Module):
    def __init__(self, in_channels, kernel_radius, dilation=1, bias=True):
        super(Conv2dSWF, self).__init__()

        stride = 1
        self.channels = in_channels
        self.padding = kernel_radius + dilation - 1
        self.convLdw = nn.Conv2d(
            in_channels=in_channels//4,
            out_channels=in_channels//4,
            kernel_size=(2 * kernel_radius + 1, kernel_radius + 1),
            stride=stride,
            padding=self.padding,
            groups=in_channels//4,
            bias=bias,
            dilation=dilation)
        self.convRdw = nn.Conv2d(
            in_channels=in_channels//4,
            out_channels=in_channels//4,
            kernel_size=(2 * kernel_radius + 1, kernel_radius + 1),
            stride=stride,
            padding=self.padding,
            groups=in_channels//4,
            bias=bias,
            dilation=dilation)
        self.convUdw = nn.Conv2d(
            in_channels=in_channels//4,
            out_channels=in_channels//4,
            kernel_size=(kernel_radius + 1, 2 * kernel_radius + 1),
            stride=stride,
            padding=self.padding,
            groups=in_channels//4,
            bias=bias,
            dilation=dilation)
        self.convDdw = nn.Conv2d(
            in_channels=in_channels//4,
            out_channels=in_channels//4,
            kernel_size=(kernel_radius + 1, 2 * kernel_radius + 1),
            stride=stride,
            padding=self.padding,
            groups=in_channels//4,
            bias=bias,
            dilation=dilation)

    def forward(self, input):
        out_L = self.convLdw(input[:, 0:int(self.channels/4), :, :])
        out_R = self.convRdw(input[:, int(self.channels/4):int(self.channels*2/4), :, :])
        out_U = self.convUdw(input[:, int(self.channels*2/4):int(self.channels*3/4), :, :])
        out_D = self.convDdw(input[:, int(self.channels*3/4):int(self.channels), :, :])

        out_L = out_L[:, :, :, :-self.padding]
        out_R = out_R[:, :, :, self.padding:]
        out_U = out_U[:, :, :-self.padding, :]
        out_D = out_D[:, :, self.padding:, :]

        out = torch.cat((out_L, out_R, out_U, out_D), 1)

        return out