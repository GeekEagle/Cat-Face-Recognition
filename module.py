import torch
import torchvision
import torchvision.models

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

class AlexNet(nn.Module):
    def __init__(self,outclass:int):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(48,64,kernel_size=5,padding=2)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2304,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,outclass)
    def forward(self, data):
        data = self.conv1(data)
        data = self.relu(data)
        data = self.pool(data)
        data = self.conv2(data)
        data = self.relu(data)
        data = self.pool(data)
        data = self.conv3(data)
        data = self.relu(data)
        data = self.conv4(data)
        data = self.relu(data)
        data = self.conv5(data)
        data = self.relu(data)
        data = self.pool(data)
        data = self.flatten(data)
        data = self.dropout(data)
        data = self.fc1(data)
        data = self.relu(data)
        data = self.dropout(data)
        data = self.fc2(data)
        data = self.relu(data)
        data = self.fc3(data)
        return data