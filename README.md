# LRENet
Official PyTorch implementation of the paper [LRENet: A Location-Related Enhancement Network for Liver Lesions in CT Images](https://iopscience.iop.org/article/10.1088/1361-6560/ad1d6b/meta).

# Architecture

![LRENet](https://github.com/HUI623/Loss_RLM/blob/main/Architecture.png)
## 1.Requirements
```bash
# Environment Setup  
matplotlib
numpy
Pillow
torch
torchvision
tqdm
wandb
```
## 2. Installation
```bash
git clone https://github.com/HUI623/LRENet 
cd LRENet 
```
## 3. Data Preprocessing

```bash
read_liver_data_save_image.py
```

## 4. Training
```bash
python train_res.py
```
## 5. Testing
```bash
python test_psnr.py
```
## 6. Visual
```bash
python test_single.py
```
## Acknowledgements
Thanks to [U-Net](https://github.com/milesial/Pytorch-UNet/tree/master) for their outstanding work.

