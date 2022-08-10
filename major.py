import numpy as np
import os
import cv2 as cv
import torch
from torchvision import transforms,datasets,utils
import torchvision.models
from torch import nn
from module import AlexNet
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
from bianmajiema import encoder,decoder,verencode
from mergedata import merge

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using gpu: %s'%torch.cuda.is_available())
data_transform = {
    "train": transforms.Compose([transforms.RandomSizedCrop(227),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize((227, 227)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                  }
check_intest_pred = []
check_intest = []
check_intest_label = []
save_check_intest_pred = './checkintest.npy'
save_check_intest_label = './checkintestlabel.npy'
save_check_intest = './testout.npy'
check_intrain = []
check_intrain_label = []
check_intrain_pred = []
save_check_intrain = './checkintrain.npy'
save_check_intrain_label = './checkintrainlabel.npy'
save_check_intrain_pred = './trainout.npy'
testxpath = './cats/test_x.npy'
testypath = './cats/test_y.npy'
valxpath = './cats/val_x.npy'
valypath = './cats/val_y.npy'
trainxpath = "./cats/train_x.npy"
trainypath = './cats/train_y.npy'

train_label = np.load(trainypath)
train_label = list(train_label)
for i in range(0,len(train_label)):
    train_label[i] = train_label[i][1:]

val_label = np.load(valypath)
val_label = list(val_label)
for i in range(0,len(val_label)):
    val_label[i] = val_label[i][1:]

test_label = np.load(testypath)
test_label = list(test_label)
for i in range(0,len(test_label)):
    test_label[i] = test_label[i][1:]

train_data = np.load(trainxpath)
test_data = np.load(testxpath)
val_data = np.load(valxpath)
train_data = merge(train_data,train_label)
test_data = merge(test_data,test_label)
val_data = merge(val_data,val_label)
traindata = DataLoader(dataset=train_data,batch_size=32, shuffle=True, num_workers=0)

train_loss_all = []  # 存放训练集损失的数组
val_loss_all = []
tran_acc_all = []
in_channels = 3
out_channels = 512
model = AlexNet(outclass=18)
loss = nn.CrossEntropyLoss()

def train():
    model.to(device)
    epoch = 21
    learning = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr = learning)
    train_num = 0
    # model.load_state_dict(torch.load("trainparam.pth"))
    for i in range(1,epoch):
        train_loss = 0
        model.train()
        traindata = DataLoader(dataset=train_data, batch_size=32,
                               shuffle=True, num_workers=0)
        for step,data in enumerate(traindata):
            img,target = data
            # print(type(img))
            img = img.transpose(dim0=3,dim1=1)
            img = img.transpose(dim0=3,dim1=2)
            img = img.float()
            optimizer.zero_grad()
            outputs = model(img.to(device))
            target = target.float().cuda()
            loss1 = loss(outputs,target)
            loss1.backward()
            optimizer.step()
            outputs = decoder(outputs)
            train_loss += abs(loss1.item()*img.size(0))
            train_num += img.size(0)
            if i == 20:
                img = img.transpose(dim0=3, dim1=1)
                img = img.transpose(dim0=2, dim1=1)
                img = img.int()
                check_intrain.append(img[0].data.tolist())
                check_intrain_label.append(target[0].data.tolist())
                check_intrain_pred.append(outputs[0].data.tolist())
            drawtrainout = outputs
            drawtrainlabel = target
        train_loss_all.append(train_loss / train_num)   #将训练的损失放到一个列表里 方便后续画图
        print("epoch：{} ， train-Loss：{}".format(i, train_loss/train_num))  # 输出训练情况
        torch.save(model.state_dict(), "trainparam.pth")
        if i % 5 == 0:
            test("validation")
        # np.save(save_check_intrain_pred,check_intrain_pred)
        # np.save(save_check_intrain,check_intrain)
        # np.save(save_check_intrain_label,check_intrain_label)
    return drawtrainout,drawtrainlabel

def test(str):
    if (str=="validation"):
        chosenx = DataLoader(dataset= val_data, batch_size= 32, shuffle= True , num_workers=0)
    elif (str=="test"):
        chosenx = DataLoader(dataset= test_data, batch_size= 32, shuffle= False , num_workers=0)

    test_num = 0
    test_loss = 0
    # model.load_state_dict(torch.load("trainparam.pth"))
    model.to(device)
    model.eval()
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for step, data in enumerate(chosenx):
            img,target = data
            img = img.transpose(dim0=3, dim1=1)
            img = img.transpose(dim0=3, dim1=2)
            img = img.float()
            outputs = model(img.to(device))
            target = target.float().cuda()
            loss1 = loss(outputs,target)
            outputs = decoder(outputs)
            test_loss += abs(loss1.item() * img.size(0))
            test_num += img.size(0)
            img = img.transpose(dim0=3, dim1=1)
            img = img.transpose(dim0=2, dim1=1)
            img = img.int()
            check_intest.append(img[0].data.tolist())
            check_intest_label.append(target[0].data.tolist())
            check_intest_pred.append(outputs[0].data.tolist())
        print("test-Loss：{}".format(test_loss / test_num))
        val_loss_all.append(test_loss / test_num)
        # np.save(save_check_intest_pred, check_intest_pred)
        # np.save(save_check_intest, check_intest)
        # np.save(save_check_intest_label, check_intest_label)

    if(str=="test"):
        drawtestlabel = np.load(save_check_intest_label)
        drawtestout = np.load(save_check_intest_pred)
        img = np.load(save_check_intest)
        for i in range(0, img.shape[0]):
            xout = drawtestout[i][::2]
            yout = drawtestout[i][1::2]
            xlabel = drawtestlabel[i][::2].data.tolist()
            ylabel = drawtestlabel[i][1::2].data.tolist()
            print(xlabel)
            print(ylabel)
            print(xout)
            print(yout)
            plt.scatter(xout, yout, c='g', marker=".")
            plt.scatter(xlabel, ylabel, c='r', marker=".")
            plt.imshow(img[i])
            plt.show()
            input("qwq")
    return outputs


if __name__ == "__main__":
    train()
    print("complete training")
    test("test")
    print("complete test")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    x = [x for x in range(1,21)]
    print(x)
    print(train_loss_all)
    plt.plot(x, train_loss_all, "r-", label="Train loss")
    plt.plot(x, val_loss_all, "b-", label="Validation loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.show()
    drawlabel = np.load(save_check_intrain_label)
    drawout = np.load(save_check_intrain_pred)
    img = np.load(save_check_intrain)
    for i in range(0,img.shape[0]):
        xout = drawout[i][::2]
        yout = drawout[i][1::2]
        xlabel = drawlabel[i][::2].data.tolist()
        ylabel = drawlabel[i][1::2].data.tolist()
        print(xlabel)
        print(ylabel)
        print(xout)
        print(yout)
        plt.scatter(xout, yout, c='g', marker=".")
        plt.scatter(xlabel, ylabel, c='r', marker=".")
        plt.imshow(img[i])
        plt.show()
        input("qwq")