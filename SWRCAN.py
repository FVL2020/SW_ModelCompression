import math
import torch.nn as nn
from SWConv import Conv2dSW
from SWConvF import Conv2dSWF


class SWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(SWConv, self).__init__()
        kernel_radius = kernel_size // 2
        self.SWConv = Conv2dSW(in_channels=in_channels, out_channels=out_channels, kernel_radius=kernel_radius, bias=bias)

    def forward(self, x):
        out = self.SWConv(x)
        return out


class SWConvF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(SWConvF, self).__init__()
        kernel_radius = kernel_size // 2
        self.SWConvF = nn.Sequential(
            Conv2dSWF(in_channels=in_channels, kernel_radius=kernel_radius, dilation=dilation, bias=bias),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False))

    def forward(self, x):
        out = self.SWConvF(x)
        return out


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, swconvf, n_feat, kernel_size, reduction, dilation, bias=True, bn=False, act=nn.ReLU(True)):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(swconvf(n_feat, n_feat, kernel_size, dilation=dilation, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, swconvf, swconv, n_feat, kernel_size, reduction):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=1, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=2, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=1, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=3, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=1, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=4, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=1, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=5, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=1, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=6, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=6, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=1, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=5, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=1, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=4, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=1, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=3, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=1, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=2, bias=True, bn=False, act=nn.ReLU(True)),
            RCAB(swconvf, n_feat, kernel_size, reduction, dilation=1, bias=True, bn=False, act=nn.ReLU(True))]

        modules_body.append(swconv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class SWRCAN(nn.Module):
    def __init__(self, n_colors, scale, swconvf=SWConvF, swconv=SWConv):
        super(SWRCAN, self).__init__()

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = scale

        # define head module
        modules_head = [swconv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [ResidualGroup(swconvf, swconv, n_feats, kernel_size, reduction) for _ in range(n_resgroups)]

        modules_body.append(swconv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(swconvf, scale, n_feats, act=False),
            nn.Conv2d(n_feats, n_colors, kernel_size, padding=(kernel_size // 2))]

        self.Attn = nn.Conv2d(in_channels=n_colors, out_channels=n_colors, kernel_size=1, stride=1, padding=0,
                              groups=1, bias=True)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        Attn = self.Attn(x)
        x = x * Attn

        return x


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)