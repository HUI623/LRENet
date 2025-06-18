import torch
import torch.nn.functional as F
import numpy
import math

from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def bright_mae(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    eval_mae=0

    # iterate over the validation set 遍历验证集
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image = batch['image']
        # move images  to correct device and type
        image = image.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the image
            out_pred = net(image)

            # compute the mae
            tmp= torch.abs(out_pred.cuda() - image.cuda())   #tensor类型abs绝对值，tmp是否/ 255.0 ？
            eval_mae = torch.mean(tmp)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return eval_mae
    return eval_mae / num_val_batches

def bright_mse(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    eval_mse = 0

    # iterate over the validation set 遍历验证集
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image = batch['image']
        # move images  to correct device and type
        image = image.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the image
            out_pred = net(image)

            # compute the mae
            tmp = (out_pred.cuda() - image.cuda()) ** 2
            eval_mse = torch.mean(tmp)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return eval_mse
    return eval_mse / num_val_batches

def bright_AB(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    eval_AB = 0

    # iterate over the validation set 遍历验证集
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image = batch['image']
        # move images  to correct device and type
        image = image.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the image
            out_pred = net(image)

            # compute the mse
            tmp1 = torch.mean(image.cuda())
            tmp2 = torch.mean(out_pred.cuda())
            tmp = tmp1 - tmp2
            eval_AB = torch.abs(tmp)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return eval_AB
    return eval_AB / num_val_batches

def log10(x):
    numerator = K.log(x)
    denominator = K.log(K.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def bright_psnr(net, dataloader, device):
    mse = K.mean((K.abs(y_pred[:, :, :, :3] - y_true[:, :, :, :3])) ** 2)
    max_num = 1.0
    psnr = 10 * log10(max_num ** 2 / mse)
    return psnr


if __name__ == '__main__':
    src_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/src_img.png",0)
    src_img = torch.from_numpy(src_img)[None][None] # [n,c,h,w] ，[None]扩充两个维度N,C

    pred_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/pred_img.png",0)
    pred_img = torch.from_numpy(pred_img)[None][None]  # [n,c,h,w]

    seg_img = cv2.imread("/media/wanghui/wh2021/code/test/20221029/seg_img.png", 0)
    seg_img = torch.from_numpy(seg_img)[None][None]  # [n,c,h,w]

    handle = L1_Loss(softw=0.08,c=0.1)
    loss = handle.forward(src_img=src_img, pred_img=pred_img, seg_img=seg_img)
    print("loss： ",loss)
    print("finished! \n")




