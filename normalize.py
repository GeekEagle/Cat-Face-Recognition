import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import natsort

train_x = []
train_y = []
cv_x = []
cv_y = []
test_x = []
test_y = []
save_trainx = './cats/train_x.npy'
save_trainy = './cats/train_y.npy'
save_cvx = './cats/val_x.npy'
save_cvy = './cats/val_y.npy'
save_testx = './cats/test_x.npy'
save_testy = './cats/test_y.npy'
def resize_img(DATADIR, img_size):
    w = img_size[0]
    h = img_size[1]
    '''设置目标像素大小，此处设为300'''
    path = os.path.join(DATADIR)
    # 返回path路径下所有文件的名字，以及文件夹的名字，
    for root,dirs,files in os.walk(path):
        for i in dirs:
            direc = os.path.join(root, i)
            print(root)
            print(direc)
            for file in os.listdir(direc):
                if file.endswith(".jpg"):
                    img_array = cv2.imread((direc + '/' + file), cv2.IMREAD_COLOR)
                    new_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_CUBIC)
                    len = img_array.shape[0]
                    wid = img_array.shape[1]
                    # print(len, wid)
                    savefile(new_array, direc,file)
                elif file.endswith(".cat"):
                    temp = direc + '\\' + file
                    # print(temp)
                    data = np.loadtxt(temp, dtype=int)
                    for j in range(1, data.shape[0]):
                        if (j % 2 == 1):
                            data[j] = data[j] * w / wid
                        else:
                            data[j] = data[j] * h / len
                    savefile(data,direc,file)
        break
    np.save(save_trainx,train_x)
    np.save(save_trainy,train_y)
    np.save(save_cvx,cv_x)
    np.save(save_cvy,cv_y)
    np.save(save_testx,test_x)
    np.save(save_testy,test_y)

def savefile(new_array,direc,file):
    if direc.endswith("00") or direc.endswith("01") \
            or direc.endswith("02") or direc.endswith("03"):
        if len(np.array(new_array).shape) == 1:
            train_y.append(new_array)
        else:
            cv2.imwrite('./cats/train_x/'+file, new_array)
            train_x.append(new_array)
    elif direc.endswith("04"):
        if len(np.array(new_array).shape)==1:
            cv_y.append(new_array)
        else:
            cv2.imwrite('./cats/val_x/'+file, new_array)
            cv_x.append(new_array)
    elif direc.endswith("05") or direc.endswith("06"):
        if len(np.array(new_array).shape) == 1:
            test_y.append(new_array)
        else:
            cv2.imwrite('./cats/test_x/'+file, new_array)
            test_x.append(new_array)

if __name__ == '__main__':
    path = "F:\\Python code\\venv\\share\\mldesign\\cats\\"
    # 需要修改的新的尺寸
    img_size = [227, 227]
    resize_img(path, img_size)
    parameter = np.load(save_trainy)
    seecat = np.load(save_trainx)
    i=0
    files = os.listdir('./cats/train_x/sbdx/')
    print(len(files))
    print(len(seecat))
    for file in files:
        img = cv2.imread('./cats/train_x/sbdx/'+files[i], cv2.IMREAD_COLOR)
        xlabel = parameter[i][1::2]
        ylabel = parameter[i][2::2]
        plt.scatter(xlabel, ylabel, c='r', marker='.')
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()
        plt.scatter(xlabel, ylabel, c='b', marker='.')
        plt.imshow(seecat[i], cmap=plt.cm.binary)
        plt.show()
        print(i)
        i = i + 184
        if (i > 5888): break
        input('qwq')
    i=0
    parameter = np.load(save_cvy)
    seecat = np.load(save_cvx)
    for root,dirs,files in os.walk('./cats/val_x/sbdx'):
        print((len(files)))
        for file in files:
            img = cv2.imread(root + '/' + files[i], cv2.IMREAD_COLOR)
            xlabel = parameter[i][1::2]
            ylabel = parameter[i][2::2]
            plt.scatter(xlabel, ylabel, c='r', marker='.')
            plt.imshow(img, cmap=plt.cm.binary)
            plt.show()
            plt.scatter(xlabel, ylabel, c='b', marker='.')
            plt.imshow(seecat[i], cmap=plt.cm.binary)
            plt.show()
            print(i)
            i = i + 42
            if (i > 1376): break
            input('qwq')

    parameter = np.load(save_testy)
    seecat = np.load(save_testx)
    i=0
    for root, dirs, files in os.walk('./cats/test_x/sbdx'):
        print(len(files))
        for file in files:
            img = cv2.imread(root + '/' + files[i], cv2.IMREAD_COLOR)
            xlabel = parameter[i][1::2]
            ylabel = parameter[i][2::2]
            plt.scatter(xlabel, ylabel, c='r', marker='.')
            plt.imshow(img, cmap=plt.cm.binary)
            plt.show()
            plt.scatter(xlabel, ylabel, c='b', marker='.')
            plt.imshow(seecat[i], cmap=plt.cm.binary)
            plt.show()
            print(i)
            i = i + 83
            if (i > 2688): break
            input('qwq')