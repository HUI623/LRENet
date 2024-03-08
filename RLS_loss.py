import cv2
import torch
import torch.nn as nn
import time
from Region_loss import Region_Loss
from L1_loss import L1_Loss
from Ms_Ssim_loss import MS_SSIM_LOSS


class RLS_LOSS(nn.Module):
    def __init__(self, weights):
        super(RLS_LOSS, self).__init__()
        self.cuda = "cuda:0"
        self.W1 = weights[0]
        self.W2 = weights[1]
        self.W3 = weights[2]

    def forward(self, hr_img, pred_img, seg_img):
        loss_region_handle = Region_Loss(side=19)
        loss_region = loss_region_handle.forward(hr_img, pred_img, seg_img)
        loss_l1_handle = L1_Loss(softw=0.08, c=0.1)
        loss_l1 = loss_l1_handle.forward(hr_img, pred_img, seg_img)
        loss_ms_ssim_handle = MS_SSIM_LOSS()
        loss_ms_ssim = loss_ms_ssim_handle.forward(hr_img, pred_img, seg_img)
        loss_total = self.W1 * loss_region + self.W2 * loss_l1 + self.W3 + loss_ms_ssim
        return loss_total


if __name__ == '__main__':
    hr_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/src_img.png", 0)
    hr_img = torch.from_numpy(hr_img)[None][None]  # tensor[n,c,h,w] ，[None]扩充两个维度N,C

    pred_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/pred_img.png", 0)
    pred_img = torch.from_numpy(pred_img)[None][None]  # [n,c,h,w]

    seg_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/seg_img.png", 0)
    seg_img = torch.from_numpy(seg_img)[None][None]  # [n,c,h,w]

    RLS_LOSS_handle = RLS_LOSS(weights=[0.7, 0.975, 0.025])
    t1 = time.time()
    loss = RLS_LOSS_handle.forward(hr_img=hr_img, pred_img=pred_img, seg_img=seg_img)
    t2 = time.time()
    print("cost time: ", t2-t1)
    print("loss： ", loss)
    print("finished! \n")
