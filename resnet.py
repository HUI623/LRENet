#! /usr/bin/env/python3
# -*- coding=utf-8 -*-
'''
======================模块功能描述=========================    
       @File     : resnet.py
       @IDE      : PyCharm
       @Author   : Wanghui-BIT
       @Date     : 2022/11/23 下午2:03
       @Desc     : the Res_UNet model
=========================================================   
'''


import torch
import torch.nn as nn
from torch.nn import functional as F  #插值法-最邻近nearest
import ptflops
from ptflops import get_model_complexity_info
from torchvision import models


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect', bias=False),# 图像大小不变
            nn.BatchNorm2d(out_channel),
            # 防止过拟合
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channel, out_channel,3,1,1,padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


# 残差模块, 包含两个卷积模块
class res_block(nn.Module):
    def __init__(self, in_channel, out_channel):  #需要做跳连，因此输入和输出通道数是一致的
        super(res_block, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,1,1,0,padding_mode='reflect',bias=False),  #图像尺寸不变
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3,1,1, padding_mode='reflect', bias=False),   #图像尺寸不变
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
#         """
#         前向传播.（只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现-利用Autograd）
#         :参数 input: 输入图像集，张量表示，大小为 (N, c, w, h)
#         :返回: 输出图像集，张量表示，大小为 (N, c, w, h)
#         """
        output = self.layer2(x)  # (N, C, w, h)
        residual = self.layer1(x)
        output += residual  # (N, C, w, h)
        return output


class ResPath(nn.Module):
    def __init__(self, channel, stage):
        super(ResPath, self).__init__()
        self.stage = stage
        self.block = res_block(channel,channel)

    def forward(self,x):
        out = self.block(x)
        for i in range(self.stage-1):
            out = self.block(out)
        return out


class Down(nn.Module):
    """Downscaling with COV3*3 then double conv"""
    # kernel=3*3,strides=2,pading=1  最大池化用卷积代替
    def __init__(self, channel):
        super(Down, self).__init__()
        self.layer = nn.Sequential(
            # 使用卷积进行2倍下采样，通道数不变
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),   # 图像大小缩小2倍
            nn.BatchNorm2d(channel), # 可用可不用
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    """Upscaling then double conv"""
    # 上采样两种，转置卷积（空洞卷积）和插值法（线性和最邻近）
    def __init__(self, channel):
        super(UpSample,self).__init__()
        # 通道减半，kernel=1*1不会特征提取，只为降通道，strides=1
        self.layer=nn.Conv2d(channel,channel//2,1,1)

        # if nearest, use the normal convolutions to reduce the number of channels
    def forward(self, x):
        up=F.interpolate(x,scale_factor=2,mode='nearest')  # 输入图X的尺寸，插值后变成原来的2倍
        out=self.layer(up)   # 上采样的特征图
        return out
        # return torch.cat((out,feature_map),dim=1)  #上下采样通道C拼接;NCHW,所以dim=1


class UNet(nn.Module):
    def __init__(self,n_channels):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.c1=DoubleConv(n_channels,64)  # 输入通道3，输出64
        self.p1=ResPath(64,4)
        self.d1=Down(64)
        self.c2=DoubleConv(64,128)
        self.p2=ResPath(128,3)
        self.d2=Down(128)
        self.c3=DoubleConv(128,256)
        self.p3=ResPath(256,2)
        self.d3=Down(256)
        self.c4=DoubleConv(256,512)
        self.p4=ResPath(512,1)
        self.d4=Down(512)
        self.c5=DoubleConv(512,1024)
        # 四次上采样
        self.u1=UpSample(1024)
        self.c6=DoubleConv(1024,512)
        self.u2=UpSample(512)
        self.c7=DoubleConv(512,256)
        self.u3=UpSample(256)
        self.c8=DoubleConv(256,128)
        self.u4=UpSample(128)
        self.c9=DoubleConv(128,64)

        self.out=nn.Conv2d(64,n_channels,3,1,1)  # 输出彩色图片：n_channels=3,cov3*3,s=1,p=1,灰度图n_channels=1
        # 激活函数，用tan（）也可以，多分类用softmax
        # self.Th = torch.nn.Sigmoid()


    def forward(self,x):
        # print("x",x.shape)  # [4, 1, 512, 512]
        # 下采样部分
        R1=self.c1(x)  # [4, 64, 512, 512]
        # print("R1", R1.shape)
        # x1=self.p1(R1)   # [4, 64, 512, 512]
        # print("x1", x1.shape)
        # b1=self.d1(R1)   # [4, 64, 256, 256]
        # # print("b1", b1.shape)
        # R2=self.c2(b1)   # [4, 128, 256, 256]
        R2 = self.c2(self.d1(R1))
        # print("R2", R2.shape)
        # x2=self.p2(R2)
        # print("x2", x2.shape)
        R3 = self.c3(self.d2(R2))
        # print("R3", R3.shape)
        # x3=self.p3(R3)
        # print("x3", x3.shape)
        R4 = self.c4(self.d3(R3))
        # x4=self.p4(R4)
        # print("x4", x4.shape)
        R5 = self.c5(self.d4(R4))
        # print("R5", R5.shape)

        #上采样部分，需要拼接
        # up1=self.u1(R5)
        cat1=torch.cat((self.p4(R4),self.u1(R5)), dim=1)
        O1 = self.c6(cat1)
        # up2=self.u2(O1)
        cat2=torch.cat((self.p3(R3),self.u2(O1)), dim=1)
        O2 = self.c7(cat2)
        # up3=self.u3(O2)
        cat3 = torch.cat((self.p2(R2),self.u3(O2)), dim=1)
        # print("cat3",cat3.shape)
        O3 = self.c8(cat3)
        # print("O3",O3.shape)
        # up4= self.u4(O3)
        cat4 = torch.cat((self.p1(R1),self.u4(O3)), dim=1)
        # O4 = self.c9(cat4)   # 64,

        # print("outc.shape: ", self.outc.shape)
        logits = self.out(self.c9(cat4))

        # print('logits',logits)
        return logits

        #输出预测，这里大小和输入一致
        #可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        # logits = self.Th(self.out(O4))
        # # print("logits.shape: ",logits.shape)
        # return logits
        # print(logits)
#测试一下看网络对不对，给一个x的值，看输出是不是一致的
if __name__ =='__main__':
    x=torch.randn(16,1,512,512)   #N,C,H,W
    net=UNet(n_channels=1)
    print(net(x).shape)   # 16,1,512,512
