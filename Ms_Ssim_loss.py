#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2021/8/2 17:09
# @Author  : Qilin Zhejiang University
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class MS_SSIM_LOSS(nn.Module):
    """
    Have to use cuda, otherwise the speed is too slow.
    Both the group and shape of input image should be attention on.
    I set 255 and 1 for gray image as default.
    """

    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=255.0,
                 K=(0.01, 0.03),  # c1,c2
                 alpha=1.0,  # weight of ssim
                 compensation=1,  # final factor for total loss 200或1
                 cuda_dev=0,  # cuda device choice
                 channel=1):  # RGB image should set to 3 and Gray image should be set to 1
        super(MS_SSIM_LOSS, self).__init__()
        self.channel = channel
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])  #16
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1) # 33
        g_masks = torch.zeros(
            (self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))  # 创建了(3*5, 1, 33, 33)个masks
        for idx, sigma in enumerate(gaussian_sigmas):
            if self.channel == 1:
                # only gray layer
                g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            elif self.channel == 3:
                # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
                g_masks[self.channel * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size,
                                                                                   sigma)  # 每层mask对应不同的sigma
                g_masks[self.channel * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            else:
                raise ValueError
        self.g_masks = g_masks.cuda(cuda_dev)  # 转换为cuda数据类型

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)
        # Outer product of input and vec2. If input is a vector of size nn and vec2 is a vector of size mm,
        # then out must be a matrix of size (n \times m)(n×m).

    def forward(self, hr_img, pred_img, seg_img):
        hr_img = hr_img.float().cuda()
        pred_img = pred_img.float().cuda()
        b, c, h, w = hr_img.shape
        assert c == self.channel
        
        mux = F.conv2d(hr_img, self.g_masks, groups=c, padding=self.pad)  # 图像为96*96，和33*33卷积，出来的是64*64，加上pad=16,出来的是96*96
        muy = F.conv2d(pred_img, self.g_masks, groups=c, padding=self.pad)  # groups 是分组卷积，为了加快卷积的速度

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(hr_img * hr_img, self.g_masks, groups=c, padding=self.pad) - mux2
        sigmay2 = F.conv2d(pred_img * pred_img, self.g_masks, groups=c, padding=self.pad) - muy2
        sigmaxy = F.conv2d(hr_img * pred_img, self.g_masks, groups=c, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        if self.channel == 3:
            lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]  # 亮度对比因子
            PIcs = cs.prod(dim=1)
        elif self.channel == 1:
            lM = l[:, -1, :, :]
            PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]
        # loss_ms_ssim = lM * PIcs  # [B, H, W]  #不行，预测图为全白

        loss_mix = self.alpha * loss_ms_ssim
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()


if __name__ == '__main__':
    hr_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/src_img.png",0)
    hr_img = torch.from_numpy(hr_img)[None][None] # [n,c,h,w] ，[None]扩充两个维度N,C

    pred_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/pred_img.png",0)
    pred_img = torch.from_numpy(pred_img)[None][None]  # [n,c,h,w]

    seg_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/seg_img.png", 0)
    seg_img = torch.from_numpy(seg_img)[None][None]# [n,c,h,w]

    handle = MS_SSIM_LOSS()
    loss = handle.forward(hr_img=hr_img, pred_img=pred_img, seg_img=seg_img)
    print("loss： ",loss)
    print("finished! \n")
