import numpy
import torch
from torchvision import transforms


std = [0.229,0.224,0.225]
mean = [0.485,0.456,0.406]
def encoder(img):
    img = img.cuda()
    for i in range(0,img.size(0)):
        for j in range(0,img.size(1)):
                    img[i][j] = (img[i][j]-112)/112
    return img

def decoder(img):
    img = img.cuda()
    for i in range(0, img.size(0)):
        for j in range(0, img.size(1)):
            img[i][j] = img[i][j]*112 + 112
    return img

def verencode(verify):
    transform_val_list = [
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ]
    trans_compose = transforms.Compose(transform_val_list)
    re = trans_compose(verify)
    return re