import matplotlib
matplotlib.use('TkAgg')
import nibabel as nib
from PIL import Image
import numpy as np
import os

"""
函数功能：读取liver数据集并存为png文件
输入参数：
    1) src_dir liver数据集（nii文件）文件夹路径，该文件夹包含多个nii文件
    2）dst_dir liver数据集（png文件）文件夹路径，该文件夹包含多个png文件
    3）src_type 存为png文件类型，即将原图image/标签图label进行转为png文件
输出参数：
"""


def read_liver_data_save_image(src_dir,dst_dir,src_type,dataset_type):
    src_path = os.path.join(os.getcwd(),src_dir)
    src_files = os.listdir(src_path)
    # sort
    if src_type == "image":
        src_files.sort(key=lambda x: int(x[7:-4]))
    elif src_type == "label":
        src_files.sort(key=lambda x: int(x[13:-4]))
    num = 0
    for src_file in src_files:
        img = nib.load(os.path.join(src_path,src_file))
        width, height, queue = img.dataobj.shape
        file_name = src_file.split(".")[0].replace("segmentation","volume")

        if dataset_type == 'mmseg_unet':
            os.mkdir(dst_dir + '/' + file_name)
            temp_save_path = dst_dir + '/' + file_name
        elif dataset_type == 'pytorch_unet':
            temp_save_path = dst_dir

        print("样本名：",src_file)
        print("切片数：",queue)
        for i in range(queue):
            print("第",i,"张切片")
            img_arr = img.dataobj[:, :, i]
            path_save = temp_save_path + "/" + "slice-" + '%04d' % i + ".png"
            print("path_save: ",path_save)
            if src_type == "label":
                Image.fromarray(np.uint8(img_arr)).save(path_save, "png")
            elif src_type == "image":
                Image.fromarray(np.uint8(img_arr*255)).save(path_save, "png")
        num += 1


if __name__ == '__main__':
    src_dir = "/media/wanghui/wh2021/liver/01_liver_data/2train_lable"
    dst_dir = "/media/wanghui/wh2021/DataSets/01_liver/Type_Cityscapes_liver_hu_valued/gtFine/train"
    src_type = "label"
    read_liver_data_save_image(src_dir,dst_dir,src_type,dataset_type='mmseg_unet')






