import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
from network.inceptionv4 import inceptionv4
# from inceptionv4 import inceptionv4
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)
    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y
def fft2d(input):
    fft_out = fft.fftn(input, dim=(2, 3), norm='ortho')
    return fft_out


def fftshift2d(input):
    b, c, h, w = input.shape
    fs11 = input[:, :, -h // 2:h, -w // 2:w]
    fs12 = input[:, :, -h // 2:h, 0:w // 2]
    fs21 = input[:, :, 0:h // 2, -w // 2:w]
    fs22 = input[:, :, 0:h // 2, 0:w // 2]
    output = torch.cat([torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3)
    return output
class FAD_Head(nn.Module):
    def __init__(self, size=299):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

        # 4 kernel
        # y_list = []
        # for i in range(4):
        #     x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
        #     y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
        #     y_list.append(y)
        # out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        x_pass = self.filters[2](x_freq)  # [N, 3, 299, 299]
        y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
        # out= y
        # ============================
        input = y
        x_fft = fft2d(input)
        x_phase = torch.atan2(x_fft.imag + 1e-8, x_fft.real + 1e-8)
        phase = fftshift2d(x_phase)
        phase_1 = 0.11 * phase[:, 0, :, :] + 0.59 * phase[:, 1, :, :] + 0.3 * phase[:, 2, :, :]
        x_amplitude = torch.abs(x_fft)
        x_amplitude = torch.pow(x_amplitude + 1e-8, 0.8)
        amplitude = fftshift2d(x_amplitude)
        amplitude_1 = 0.11 * amplitude[:, 0, :, :] + 0.59 * amplitude[:, 1, :, :] + 0.3 * amplitude[:, 2, :, :]

        amplitude = torch.unsqueeze(amplitude_1, dim=1)
        phase = torch.unsqueeze(phase_1, dim=1)
        x = torch.cat((phase, amplitude), dim=1)
        # x = torch.cat((x, phase), dim=1)
        return x
class new_F3Net(nn.Module):
    def __init__(self, num_classes=1, img_width=299, img_height=299,device=None,pretrained=False):
        super(new_F3Net, self).__init__()
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.FAD_Head=FAD_Head(img_size)
        self.FAD_incep = inceptionv4(self.num_classes, self.pretrained)
        self.FAD_incep.features[0].conv = nn.Conv2d(2, 32, 3, 2, 0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(p=0.2)
    def forward(self,x):
        fea_FAD = self.FAD_Head(x)
        fea_FAD = self.FAD_incep(fea_FAD)
        y = fea_FAD
        f=y
        return f
def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.
# model =new_F3Net()
# print(model)
