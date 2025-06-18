import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]  #读图.startswith以.开头，splitext分离文件名[0]与扩展名[1]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC) # 最邻近插值和双三次插值,512变成512
        img_ndarray = np.asarray(pil_img)   # 转numpy格式，不占内存  H,W,C

        if img_ndarray.ndim == 2 and not is_mask:  #.ndim维度
            img_ndarray = img_ndarray[np.newaxis, ...]  #在np.newaxis位置增加一维 1,H,W,C
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))  #1,C,H,W

        if not is_mask:
            img_ndarray = img_ndarray / 255  #原图、hu图都没有做归一化，需要/255
            # img_ndarray = img_ndarray     #归一化图
        if is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]



        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            # open读取文件：W,H,C
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        # modify the mask images name read style.
        # mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        mask_file = list(self.masks_dir.glob(name + '.*'))  #.glob该方法返回所有匹配的文件路径列表（list）
        img_file = list(self.images_dir.glob(name + '.*'))
        # print("*" * 100)
        # print("mask_file: ",mask_file)
        # print("img_file: ",img_file)
        # print("name: ",name)
        # print("*"*100)

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        hr_img=torch.as_tensor(img.copy()).float().contiguous()
        lr_img=torch.nn.MaxPool2d(4)(hr_img)
        mask=torch.as_tensor(mask.copy()).long().contiguous()

        return {
            'hr_img': hr_img,
            'lr_img': lr_img,
            'mask': mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
