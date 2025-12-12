
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECABlock(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size

    """

    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1, dilation=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
        dilation=dilation)


def up_conv3x3(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv3x3(in_channels, out_channels))


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, residual=True, norm=nn.BatchNorm2d,
                 act=nn.ReLU, concat=True, use_att=False, dilations=None, out_fuse=False):
        super(UpConv, self).__init__()
        if dilations is None:
            dilations = []
        self.concat = concat
        self.residual = residual
        self.conv2 = []
        self.use_att = use_att

        self.out_fuse = out_fuse
        self.up_conv = up_conv3x3(in_channels, out_channels, transpose=False)
        if isinstance(norm, str):
            if norm == 'bn':
                norm = nn.BatchNorm2d
            elif norm == 'in':
                norm = nn.InstanceNorm2d
            else:
                raise TypeError("Unknown Type:\t{}".format(norm))
        self.norm0 = norm(out_channels)
        if len(dilations) == 0: dilations = [1] * blocks

        if self.concat:
            self.conv1 = conv3x3(2 * out_channels, out_channels)
            self.norm1 = norm(out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
            self.norm1 = norm(out_channels)
        for i in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels, dilation=dilations[i], padding=dilations[i]))

        self.bn = []
        for _ in range(blocks):
            self.bn.append(norm(out_channels))
        self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def forward(self, from_up, from_down, se=None):
        # from_up分辨率比from_down小一倍，但是channel大一倍
        from_up = self.act(self.norm0(self.up_conv(from_up)))
        if self.concat:
            x1 = torch.cat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up

        x1 = self.act(self.norm1(self.conv1(x1)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)

            if (se is not None) and (idx == len(self.conv2) - 1):  # last
                x2 = se(x2)

            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        return x2


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, pooling=True, norm=nn.BatchNorm2d, act=nn.ReLU, residual=True, dilations=None):
        super(DownConv, self).__init__()
        if dilations is None:
            dilations = []
        self.pooling = pooling
        self.residual = residual
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        if isinstance(norm, str):
            if norm == 'bn':
                norm = nn.BatchNorm2d
            elif norm == 'in':
                norm = nn.InstanceNorm2d
            else:
                raise TypeError("Unknown Type:\t{}".format(norm))
        self.norm1 = norm(out_channels)
        if len(dilations) == 0:
            dilations = [1] * blocks
        self.conv2 = []
        for i in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels, dilation=dilations[i], padding=dilations[i]))

        self.bn = []
        for _ in range(blocks):
            self.bn.append(norm(out_channels))
        self.bn = nn.ModuleList(self.bn)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        x1 = self.act(self.norm1(self.conv1(x)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool


class Refinement(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, ngf=32, blocks=3, n_skips=0):
        # in_channels =4,out_channels = 3, shared_depth = 1,
        super(Refinement, self).__init__()

        self.n_skips = n_skips
        # self.act = F.relu
        self.act = nn.LeakyReLU(negative_slope=0.2)

        self.conv_in = nn.Sequential(nn.Conv2d(in_channels, ngf, 3, 1, 1), nn.InstanceNorm2d(ngf), nn.LeakyReLU(0.2))
        self.down1 = DownConv(ngf, ngf, blocks, pooling=True, residual=True, norm=nn.BatchNorm2d, act=self.act)
        self.down2 = DownConv(ngf, ngf * 2, blocks, pooling=True, residual=True, norm=nn.BatchNorm2d, act=self.act)
        self.down3 = DownConv(ngf * 2, ngf * 4, blocks, pooling=True, residual=True, norm=nn.BatchNorm2d, act=self.act)
        self.down4 = DownConv(ngf * 4, ngf * 8, blocks, pooling=False, residual=True, norm=nn.BatchNorm2d, act=self.act)

        self.up1 = UpConv(ngf * 8, ngf * 4, blocks, residual=True, concat=True, norm='bn', act=self.act)
        self.up2 = UpConv(ngf * 4, ngf * 2, blocks, residual=True, concat=True, norm='bn', act=self.act)
        self.up3 = UpConv(ngf * 2, ngf, blocks, residual=True, concat=True, norm='bn', act=self.act)

        if n_skips > 0:
            self.dec_conv2 = nn.Sequential(nn.Conv2d(ngf * 1, ngf * 1, 1, 1, 0))
            self.dec_conv3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 1, 1, 1, 0), nn.LeakyReLU(0.2), nn.Conv2d(ngf, ngf, 3, 1, 1), nn.LeakyReLU(0.2))
            self.dec_conv4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 2, 1, 1, 0), nn.LeakyReLU(0.2), nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1),
                                           nn.LeakyReLU(0.2))

        self.out_conv = nn.Sequential(*[
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf, out_channels, 1, 1, 0)
        ])

    def forward(self, coarse_bg, mask, decoder_outs=None):
        if self.n_skips < 1:
            dec_feat2 = 0
        else:
            dec_feat2 = self.dec_conv2(decoder_outs[0])

        if self.n_skips < 2:
            dec_feat3 = 0
        else:
            dec_feat3 = self.dec_conv3(decoder_outs[1])  # 64

        if self.n_skips < 3:
            dec_feat4 = 0
        else:
            dec_feat4 = self.dec_conv4(decoder_outs[2])  # 64

        xin = torch.cat([coarse_bg, mask], dim=1)
        x = self.conv_in(xin)

        x, d1 = self.down1(x + dec_feat2)
        x, d2 = self.down2(x + dec_feat3)
        x, d3 = self.down3(x + dec_feat4)
        x, d4 = self.down4(x)

        x = self.up1(x, d3)
        x = self.up2(x, d2)
        x = self.up3(x, d1)

        im = self.out_conv(x) + coarse_bg

        return im, d4

