import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.base_module import *
from einops import rearrange
from model.femasr_arch import FeMaSRNet, SA_attention, DecoderBlock, FeMaSRNet2, MaskTransformerBlock
from model.modules.fema_utils import ResBlock


class Encoder(nn.Module):
    def __init__(self, in_ch=4):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, _, _ = x.size()
        # h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class RestorationBlock(nn.Module):
    """
        Restormer
    """

    def __init__(self, dim, ffn_expansion_factor, num_heads, bias, LayerNorm_type='WithBias'):
        super(RestorationBlock, self).__init__()
        self.ln1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Channel_Attention(dim, num_heads, bias)
        self.ln2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x1 = self.attn(self.ln1(x)) + x
        x = self.ffn(self.ln2(x1)) + x1
        return x


class InpaintingBlock(nn.Module):
    """
        a resblock with FFC and conv.
    """

    def __init__(self, channels, kernel_size, ratio_g):
        super(InpaintingBlock, self).__init__()

        self.ffc = FFC_BN_ACT(channels, channels, kernel_size, ratio_gin=ratio_g, ratio_gout=ratio_g, stride=1, padding=kernel_size // 2,
                              norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False), nn.BatchNorm2d(channels), nn.ReLU(True)
        )
        self.split = SplitTupleLayer(ratio_g)
        self.concat = ConcatTupleLayer()

    def forward(self, x):
        x0 = x
        x = self.split(x)
        x = self.concat(self.ffc(x))
        x = self.conv(x) + x0
        return x


class FusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sa = SA_attention(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim * 2, 1)
        )
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, feat, mask, scale, rest_feat, inpa_feat):
        assert rest_feat.shape == inpa_feat.shape
        attn_map = self.sa(feat, mask, scale)
        if not rest_feat.shape[-2:] == attn_map.shape[-2:]:
            attn_map = F.interpolate(attn_map, size=rest_feat.shape[-2:], mode='bilinear')
        attn_map_r, attn_map_i = self.conv(attn_map).chunk(2, dim=1)

        x = rest_feat * F.gelu(attn_map_r) + inpa_feat * F.gelu(attn_map_i)

        return self.out(x)


class FeatureInjectionGate(nn.Module):
    """
        adjust the feature injection ratio
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.sa = MaskTransformerBlock(dim=in_ch, head=8, scale=3)
        self.resblock = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, 1),
        )
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, mask, img_feat):
        """
            x: [N,C,H,W]
        """
        if x is None:
            return self.conv(img_feat)

        attn_map = self.sa(x, mask)
        if not img_feat.shape[-2:] == x.shape[-2:]:
            attn_map = F.interpolate(attn_map, size=img_feat.shape[-2:], mode='bilinear')
        x = self.resblock(attn_map) + attn_map
        output = img_feat * F.gelu(x)

        return self.conv(output)


class ImageGeneratorVQ(BaseNetwork):
    def __init__(self, args: dict, train=True):
        super(ImageGeneratorVQ, self).__init__()
        self.args = args
        self.num_blocks = num_blocks = 3
        self.self_aware_blocks = 1
        channel = 256

        # Restore branch
        self.restbranch = nn.ModuleList([nn.Sequential(*[RestorationBlock(dim=channel, ffn_expansion_factor=2.66, num_heads=8, bias=False)
                                                         for _ in range(2)]) for _ in range(num_blocks)])  # 6
        # VQ branch
        if train:
            # load pretrained model to generate gt_indices
            load_path = self.args['pretrained']
            self.pre_net = FeMaSRNet(WM_stage=False)
            if load_path is not None:
                self.pre_net.load_state_dict(torch.load(load_path))
                print(f'load pretrain FeMaSRNet model.')

            # load pretrained codebook and decoder to match features
            self.VQbranch = FeMaSRNet2(WM_stage=True)
            if load_path is not None:
                self.VQbranch.load_state_dict(torch.load(load_path), strict=False)
                frozen_module_keywords = ['quantize', 'decoder', 'after_quant_group', 'out_conv']
                if frozen_module_keywords is not None:
                    for name, module in self.VQbranch.named_modules():
                        for fkw in frozen_module_keywords:
                            if fkw in name:
                                for p in module.parameters():
                                    p.requires_grad = False
                                break
        else:
            # load
            self.VQbranch = FeMaSRNet2(WM_stage=True)

        # fusion module
        self.fusion = FusionBlock(dim=channel)

        # decoderR
        norm_type = 'gn'
        act_type = 'silu'
        self.decoderR = nn.ModuleList([DecoderBlock(256, 256, norm_type, act_type),
                                       DecoderBlock(256, 128, norm_type, act_type),
                                       DecoderBlock(128, 64, norm_type, act_type),
                                       nn.Conv2d(64, 3, 3, 1, 1)])
        # decoderF
        self.decoderF = nn.ModuleList([DecoderBlock(256, 128, norm_type, act_type),
                                       DecoderBlock(128, 64, norm_type, act_type),
                                       nn.Conv2d(64, 3, 3, 1, 1)])

        # print network parameter number
        self.print_network()

    def forward(self, x, mask, gt=None):
        empty = torch.zeros(mask.shape).to(mask.device)
        if self.training:
            assert gt is not None
            # VQ branch
            with torch.no_grad():
                _, _, _, gt_indices, _ = self.pre_net(torch.concat((gt, empty), dim=1))
            output, l_codebook, _, l_trans, _, share_enc_feat = self.VQbranch(torch.concat((x, mask), dim=1), gt_indices)

            # restore branch
            res = share_enc_feat
            for i in range(self.num_blocks):
                res = self.restbranch[i](res)
            res_feat = self.decoderR[0](res)

            # fusion branch
            out_feat = []
            feat = share_enc_feat
            feat = self.fusion(feat, mask, 2, res_feat, output[0])
            out_feat.append(feat)
            for j in range(3):
                feat = self.decoderF[j](feat)
            fine_frame = feat

            for k in range(3):
                res_feat = self.decoderR[k + 1](res_feat)

            return fine_frame, output[-1], res_feat, l_codebook, l_trans, out_feat

        else:  # infer
            VQ_feat = []
            # VQ branch
            output, _, _, _, _, share_enc_feat = self.VQbranch(torch.concat((x, mask), dim=1))
            VQ_feat.append(output[0])

            # restore branch
            tran_feat = []
            res = share_enc_feat
            for i in range(self.num_blocks):
                res = self.restbranch[i](res)
            res_feat = self.decoderR[0](res)
            tran_feat.append(res_feat)

            # fusion branch
            out_feat = []
            feat = share_enc_feat
            feat = self.fusion(feat, mask, 2, res_feat, output[0])
            out_feat.append(feat)
            for j in range(3):
                feat = self.decoderF[j](feat)
            fine_frame = feat

            for k in range(3):
                res_feat = self.decoderR[k + 1](res_feat)

            return fine_frame, output[-1], res_feat, out_feat, VQ_feat, tran_feat
