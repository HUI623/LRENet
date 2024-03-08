import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1_Loss(nn.Module):
    def __init__(self,softw,c):
        super(L1_Loss, self).__init__()
        self.cuda = "cuda:0"
        self.softw = softw   # P=softw
        self.c = c

    def forward(self, hr_img, pred_img, seg_img):
        soft_relu = nn.ReLU()
        tmp = torch.abs(pred_img.cuda() - hr_img.cuda())/255.0
        # tmp1 = soft_relu(tmp - self.softw)
        tmp2 = torch.mean(soft_relu(tmp - self.softw))
        if tmp2==0:
            soft_l1_loss = self.c * tmp
        else:
            soft_l1_loss = tmp
        loss = soft_l1_loss.mean()
        return loss


if __name__ == '__main__':
    hr_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/src_img.png",0)
    hr_img = torch.from_numpy(hr_img)[None][None] # [n,c,h,w] ，[None]扩充两个维度N,C

    pred_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/pred_img.png",0)
    pred_img = torch.from_numpy(pred_img)[None][None]  # [n,c,h,w]

    seg_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/seg_img.png", 0)
    seg_img = torch.from_numpy(seg_img)[None][None]  # [n,c,h,w]

    handle = L1_Loss(softw=0.08,c=0.1)
    loss = handle.forward(hr_img=hr_img, pred_img=pred_img, seg_img=seg_img)
    print("loss： ",loss)
    print("finished! \n")