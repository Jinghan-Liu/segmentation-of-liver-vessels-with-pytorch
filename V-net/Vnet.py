# this file is to define the network
import torch
import torch.nn as nn
import torch.nn.functional as F


# normalization
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('Expected 5D input, but got {}D input'.format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps
        )


class LUConv(nn.Module):
    def __init__(self, nchan):
        super(LUConv, self).__init__()
        self.relu = nn.PReLU(nchan)
        self.conv = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


def make_Conv(nchan, depth):
    layers = []
    for i in range(depth):
        layers.append(LUConv(nchan))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans):
        super(InputTransition, self).__init__()
        self.conv = nn.Conv3d(1, outChans, kernel_size=5, padding=2)
        self.bn = ContBatchNorm3d(outChans)
        self.relu = nn.PReLU(outChans)

    def forward(self, x):
        out = self.bn(self.conv(x))
        out = self.relu(out)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn = ContBatchNorm3d(outChans)
        self.relu = nn.PReLU(outChans)
        self.do = nn.Dropout3d()
        self.ops = make_Conv(outChans, nConvs)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        if dropout:
            out = self.do(out)
        out = self.ops(out)
        out = self.relu(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn = ContBatchNorm3d(outChans // 2)
        self.do = nn.Dropout3d()
        self.relu1 = nn.PReLU(outChans // 2)
        self.relu2 = nn.PReLU(outChans)
        self.ops = make_Conv(outChans, nConvs)

    def forward(self, x, skipx):
        if dropout:
            x = self.do(x)
        skipxdo = self.do(skipx)
        out = self.up_conv(x)
        out = self.relu1(self.bn(out))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu = nn.PReLU(2)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = torch.sigmoid(out)
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds to what is in the actual prototxt
    def __init__(self):
        super(VNet, self).__init__()
        self.input_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16, 1)
        self.down_tr64 = DownTransition(32, 2)
        self.down_tr128 = DownTransition(64, 3, dropout=True)
        self.down_tr256 = DownTransition(128, 2, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1)
        self.up_tr32 = UpTransition(64, 32, 1)
        self.output_tr = OutputTransition(32)

    def forward(self, x):
        out16 = self.input_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.output_tr(out)
        return out
