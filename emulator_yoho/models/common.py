import torch
import torch.nn as nn


class GLayerNorm(nn.Module):
    """Global Layer Normalization for TasNet."""

    def __init__(self, channels, eps=1e-5):
        super(GLayerNorm, self).__init__()
        self.eps = eps
        self.norm_dim = channels
        self.gamma = nn.Parameter(torch.Tensor(channels))
        self.beta = nn.Parameter(torch.Tensor(channels))
        # self.register_parameter('weight', self.gamma)
        # self.register_parameter('bias', self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, sample):
        """Forward function.

        Args:
            sample: [batch_size, channels, length]
        """
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # Mean and variance [N, 1, 1]
        mean = torch.mean(sample, (1, 2), keepdim=True)
        var = torch.mean((sample - mean) ** 2, (1, 2), keepdim=True)
        sample = (sample - mean) / torch.sqrt(var + self.eps) * \
                 self.gamma + self.beta
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample



class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=7, stride=1, downsample=None, expansion=1):
        super(Bottleneck, self).__init__()
        inplanes_ = inplanes * expansion
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(inplanes, inplanes_, kernel_size=1, bias=False)
        self.bn1 = GLayerNorm(inplanes_)
        self.conv2 = nn.Conv1d(inplanes_, inplanes_, kernel_size=kernel_size, stride=stride,
                               padding=pad, bias=False, groups=inplanes_)
        self.bn2 = GLayerNorm(inplanes_)
        self.conv3 = nn.Conv1d(inplanes_, planes, kernel_size=1, bias=1)
        self.bn3 = GLayerNorm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ConvMulti(nn.Module):
    def __init__(self, inplanes, planes, stride=4):
        super(ConvMulti, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_7 = nn.Conv1d(inplanes, planes//2, kernel_size=7, stride=stride,
                                padding=3, bias=False, groups=planes//2)
        self.bn_7 = GLayerNorm(planes//2)

        self.conv_5 = nn.Conv1d(inplanes, planes//2, kernel_size=5, stride=stride,
                                padding=2, bias=False, groups=planes//2)
        self.bn_5 = GLayerNorm(planes//2)

    def forward(self, x):
        output_7 = self.bn_7(self.conv_7(x))
        output_5 = self.bn_5(self.conv_5(x))
        return self.relu(torch.cat([output_7, output_5], dim=1))


class BottleneckMulti(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=7, stride=1, downsample=None, expansion=1):
        super(BottleneckMulti, self).__init__()
        inplanes_ = inplanes * expansion
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(inplanes, inplanes_, kernel_size=1, bias=False)
        self.bn1 = GLayerNorm(inplanes_)
        # self.conv2 = nn.Conv1d(inplanes_, inplanes_, kernel_size=kernel_size, stride=stride,
        #                        padding=pad, bias=False, groups=inplanes_)
        # self.bn2 = GLayerNorm(inplanes_)
        self.conv2 = ConvMulti(inplanes_, inplanes_, stride)
        self.conv3 = nn.Conv1d(inplanes_, planes, kernel_size=1, bias=1)
        self.bn3 = GLayerNorm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
