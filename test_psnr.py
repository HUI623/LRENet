import os
import numpy as np
import math
from PIL import Image

import time

start = time.time()


def psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 10 * math.log10(255.0 * 255.0 / mse)


def mse(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    return mse

def mae(img1, img2):
    mae = np.mean(np.absolute(img1 / 1. - img2 / 1.))
    return mae

def ssim(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

# CLAHE POSHE Top-Hat Zero-DCE Retinex EnlightenGAN U-RLS790
path1 = '/media/wanghui/wh2021/code/Pytorch-UNet-master_augmentation/best_test/net_VSRL/'  # 指定输出结果文件夹
path2 = '/media/wanghui/wh2021/code/Pytorch-UNet-master_augmentation/best_test/orig_images/'  # 指定原图文件夹

f_nums = len(os.listdir(path2))

list_psnr = []
list_ssim = []
list_mse = []
list_mae = []

for i in range(f_nums):
    img1 = os.listdir(path1)
    # print(img1)
    img_a = Image.open(path1+"/"+img1[i])
    img2 = os.listdir(path2)
    # print(img2)
    img_b = Image.open(path2+"/"+img2[i])

    # img_a = np.array(img_a)             #单通道
    # img_b = np.array(img_b)
    img_a = np.array(img_a.convert('L'))   #三通道转灰度图
    img_b = np.array(img_b.convert('L'))

    psnr_num = psnr(img_a, img_b)
    ssim_num = ssim(img_a, img_b)
    mse_num = mse(img_a, img_b)
    mae_num = mae(img_a, img_b)

    list_ssim.append(ssim_num)
    list_psnr.append(psnr_num)
    list_mse.append(mse_num)
    list_mae.append(mae_num)

print("平均MAE:", np.mean(list_mae))    # ,list_mae)
print("平均PSNR:", np.mean(list_psnr))  # ,list_psnr)
print("平均SSIM:", np.mean(list_ssim))  # ,list_ssim)
# print("平均MSE:", np.mean(list_mse))  # ,list_mse)


elapsed = (time.time() - start)
# print("Time used:", elapsed)
