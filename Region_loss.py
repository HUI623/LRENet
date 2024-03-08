import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage as ndi
from torch import Tensor


class Region_Loss(nn.Module):
    def __init__(self, side):
        super(Region_Loss, self).__init__()
        self.cuda = "cuda:0"
        self.side = side   # 19
        self.struct_elem = torch.ones((1, 1, self.side, self.side), dtype=torch.uint8).to(self.cuda)


    def forward(self, hr_img, pred_img, seg_img):
        lesion_mask = seg_img
        # 标签为0,1,2
        lesion_mask[seg_img == 1] = 0  # 将肝脏标签至为=0
        lesion_mask[seg_img == 2] = 1  # 病灶=1

        # 标签为0,1

        # 2.获取病灶周围区域mask
        lesion_mask_np = lesion_mask.cpu().numpy()  #tensor转numpy格式
        struct_elem_np = self.struct_elem.cpu().detach().numpy()
        outer_border_mask = ndi.binary_dilation(lesion_mask_np, struct_elem_np) - lesion_mask_np
        inner_border_mask = lesion_mask_np - ndi.binary_erosion(lesion_mask_np, struct_elem_np)
        # 3.分别计算S1和S3的像素均值Mean_S1和Mean_S3
        pred_img_np = pred_img.cpu().detach().numpy()
        outer_and = np.logical_and(outer_border_mask, pred_img_np)
        inner_and = np.logical_and(inner_border_mask, pred_img_np)
        outer_edge_mean = outer_and.mean()
        inner_edge_mean = inner_and.mean()
        # print(outer_edge_mean)  # 值大 0.011714935302734375
        # print(inner_edge_mean)   # 值小 0.00652313232421875

        # outer_edge_mean = cv2.mean(outer_border_mask.reshape(-1), pred_img_np.reshape(-1)) #背景，[0]取第一个通道
        # inner_edge_mean = cv2.mean(inner_border_mask, pred_img_np)[0]  #病灶
        # region_loss = np.absolute(outer_edge_mean - inner_edge_mean)  #不行，预测图为全白
        region_loss = 1 - np.absolute(outer_edge_mean - inner_edge_mean)
        return region_loss


if __name__ == '__main__':
    src_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/src_img.png",0)
    src_img = torch.from_numpy(src_img)[None][None]  # [n,c,h,w] ，[None]扩充两个维度N,C

    seg_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/seg_img.png", 0)
    seg_img = torch.from_numpy(seg_img)[None][None] # [n,c,h,w]

    pred_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/pred_img.png", 0)
    pred_img = torch.from_numpy(pred_img)[None][None]  # [n,c,h,w]


    handle = Region_Loss(side=19)
    loss = handle.forward(src_img=src_img, pred_img=pred_img, seg_img=seg_img)
    print("loss： ",loss)
    print("finished! \n")