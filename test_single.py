#! /usr/bin/env/python3
# -*- coding=utf-8 -*-
'''
======================模块功能描述=========================    
       @File     : test.py
       @IDE      : PyCharm
       @Author   : Wanghui-BIT
       @Date     : 2022/11/22 下午9:04
       @Desc     : 执行单张样本测试
=========================================================   
'''

import torch
from torchvision import transforms
from unet_model import UNet
# from resnet import UNet
import time
from PIL import Image
import numpy as np


# 测试图像 1827 5483 12338 14166 17822 18279 24220 38387 39758 42500
imgPath ='/media/wanghui/wh2021/code/Pytorch-UNet-master_augmentation/best_test/orig_images/images_1827.png'

# 模型参数
if __name__ == '__main__':
    # 预训练模型
    net_checkpoint = "./checkpoint_VSnetRL/checkpoint_epoch8.pth"

    # 加载模型
    net = UNet(n_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(net_checkpoint, map_location=device))

    net.eval()

    # 加载图像,原图
    preTransform = transforms.Compose([transforms.ToTensor()])
    img = Image.open(imgPath)
    #  img.convert('1')为二值图像，非黑即白。每个像素用8个bit表示，0表示黑，255表示白
    #  img.convert('L')为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度
    img = np.array(img.convert('L'))
    img1 = preTransform(img).unsqueeze(0)
    img1 = img1.cuda()

    # 记录时间
    start = time.time()

    # 模型推理
    with torch.no_grad():
        source = net(img1)[0, :, :, :]
        source = source.cpu().detach().numpy()  # 转为numpy
        if source.shape[0] == 3:   #三通道图
            source = np.transpose(source, (1, 2, 0))
        elif source.shape[0] == 1:  #单通道图
            source = source[0]
        # source = source.transpose((1, 2, 0))  # 切换形状
        source = np.clip(source, 0, 1)          # 修正图片

        img = Image.fromarray(np.uint8(source*255))   #PIL格式
        # 保存为图片 1827 5483 12338 14166 17822 18279 24220 38387 39758 42500
        img.save('/media/wanghui/wh2021/code/Pytorch-UNet-master_augmentation/best_test/net_VSRL/pred_1827.png')


    print('用时  {:.3f} 秒'.format(time.time() - start))


