import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2 as cv
class merge(Dataset):
    def __init__(self,rawdata,ylabel):
        super(merge, self).__init__()
        self.xpro = rawdata
        self.ypro = ylabel

    def __getitem__(self, item):
        data = self.xpro[item]
        label = self.ypro[item]
        return data,label

    def __len__(self):
        return len(self.xpro)
