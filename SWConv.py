import torch
import torch.nn as nn


class Conv2dSW(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_radius, dilation=1, bias=True):
        super(Conv2dSW, self).__init__()

        stride = 1

        self.padding = kernel_radius + dilation - 1
        self.convL = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels//4,
            kernel_size=(2 * kernel_radius + 1, kernel_radius + 1),
            stride=stride,
            padding=self.padding,
            bias=bias,
            dilation=dilation)
        self.convR = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels//4,
            kernel_size=(2 * kernel_radius + 1, kernel_radius + 1),
            stride=stride,
            padding=self.padding,
            bias=bias,
            dilation=dilation)
        self.convU = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels//4,
            kernel_size=(kernel_radius + 1, 2 * kernel_radius + 1),
            stride=stride,
            padding=self.padding,
            bias=bias,
            dilation=dilation)
        self.convD = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels//4,
            kernel_size=(kernel_radius + 1, 2 * kernel_radius + 1),
            stride=stride,
            padding=self.padding,
            bias=bias,
            dilation=dilation)

    def forward(self, input):
        out_L = self.convL(input)
        out_R = self.convR(input)
        out_U = self.convU(input)
        out_D = self.convD(input)

        out_L = out_L[:, :, :, :-self.padding]
        out_R = out_R[:, :, :, self.padding:]
        out_U = out_U[:, :, :-self.padding, :]
        out_D = out_D[:, :, self.padding:, :]

        out = torch.cat((out_L, out_R, out_U, out_D), 1)

        return out