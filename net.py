import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv = weight_norm(conv)

        self.init_weight()

    def forward(self, x):
        return self.conv(x)

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')


class ResBlock(nn.Module):

    def __init__(self, channels, n_conv):
        super(ResBlock, self).__init__()

        models = []
        for _ in range(n_conv-1):
            models.append(Conv(channels, channels))
            models.append(nn.ReLU(inplace=True))
        models.append(Conv(channels, channels))

        self.model = nn.Sequential(*models)

    def forward(self, x):
        return self.model(x) + x


class KPN(nn.Module):

    def __init__(self, in_channels, mid_channels, out_kernel_size):
        super().__init__()

        n_params = math.prod(out_kernel_size)
        n_layers = n_params // mid_channels

        assert n_params % mid_channels == 0

        self.out_kernel_size = out_kernel_size  # in OIHW format
        self.proj = nn.Linear(in_channels, mid_channels)
        self.kernel_out = nn.ModuleList([nn.Linear(mid_channels, mid_channels) for _ in range(n_layers)])
        self.bias_out = nn.Linear(mid_channels, 3)

    def forward(self, x):
        x = self.proj(x)
        kernel_params = []
        for i, mlp in enumerate(self.kernel_out):
            if i > 0:
                x = F.relu(x, inplace=False)
            x = mlp(x)
            kernel_params.append(x)

        kernel = torch.cat(kernel_params, dim=1).reshape(self.out_kernel_size)
        bias = self.bias_out(F.relu(kernel_params[-1], inplace=False)).reshape(self.out_kernel_size[0])
        return kernel, bias


class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()

        channels = (9+1+1)*3

        self.res1 = ResBlock(channels, n_conv=2)
        self.res2 = ResBlock(channels*2*4, n_conv=4)

        self.out = nn.Sequential(
            ResBlock(channels*2*2, n_conv=2),
            Conv(channels * 2 * 2, 9*3)
        )

    def forward(self, x):
        cs = self.res1(x)
        hs = self.res2(torch.cat([F.pixel_unshuffle(cs, 2), F.pixel_unshuffle(x, 2)], dim=1))
        out = self.out(torch.cat([F.pixel_shuffle(hs, 2), cs, x], dim=1))
        return out


class FERNet(nn.Module):

    def __init__(self, resolution=9):
        super(FERNet, self).__init__()

        # angular resolution
        self.resolution = resolution

        # stage 1
        self.embedding = nn.Sequential(
            Conv(resolution ** 2, 64),
            nn.ReLU(inplace=True),
            Conv(64, 3),
            nn.ReLU(inplace=True)
        )
        # stage 2
        self.kpn = KPN(in_channels=2, mid_channels=9, out_kernel_size=(3, 6, 3, 3))
        # stage 3
        self.ffn = FFN()

    def forward(self, x):
        b, _, _, c, h, w = x.shape

        # stage 1: get global representation
        inp = rearrange(x, 'b r1 r2 c h w -> b (r1 r2) c h w')[:, :, 1]
        global_repr = self.embedding(inp)  # B, C ,H, W

        result = torch.zeros_like(x)
        cnt = torch.zeros_like(result)

        centers = [it * 3 + 1 for it in range(0, self.resolution // 3)]
        if self.resolution % 3 > 0:
            centers.append(self.resolution - 2)

        for cx, cy in itertools.product(centers, centers):
            norm_cx, norm_cy = cx - self.resolution // 2, cy - self.resolution // 2
            center_indice = torch.tensor([[norm_cx, norm_cy]]).to(x)  # B, 2
            weight, bias = self.kpn(center_indice)
            center_view = x[:, cx, cy]
            combined_repr = torch.cat([global_repr, center_view], dim=1)
            center_repr = F.conv2d(combined_repr, weight, bias, padding=1)
            neighbors = x[:, cx-1:cx+2, cy-1:cy+2].flatten(start_dim=1, end_dim=3)  # B, U*V*C, H, W
            out = self.ffn(torch.cat([neighbors, center_repr, global_repr], dim=1)).view(b, 3, 3, c, h, w)

            result[:, cx-1:cx+2, cy-1:cy+2] += out
            cnt[:, cx-1:cx+2, cy-1:cy+2] += 1

        result = torch.clamp(result / cnt, 0.0, 1.0)
        return result


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    # calculate macs and params
    resolution = 9
    net = FERNet(resolution)
    input_shape = (2, resolution, resolution, 3, 128, 128)
    input = torch.randn(input_shape)
    flops = FlopCountAnalysis(net, input)
    with open(f'summary.txt', 'w', encoding='utf-8') as f:
        f.write(flop_count_table(flops))
    print(net(input).shape)
