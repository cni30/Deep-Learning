import pandas as pd
import torch
import torchvision
import torchvision.models

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from PIL import ImageFile

#图像预处理（缩放裁剪等）
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(120),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((120, 120)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
# train_data = torchvision.datasets.CIFAR10(root = "./data" , train = True ,download = True,
#                                           transform = trans)


# 导入自己的数据，自己的数据放在跟代码相同的文件夹下新建一个data文件夹，data文件夹里的新建一个train文件夹用于放置训练集的图片。同理新建一个val文件夹用于放置测试集的图片。
def main():
    train_data = torchvision.datasets.ImageFolder(root = "./data/train" ,   transform = data_transform["train"])



    traindata = DataLoader(dataset= train_data , batch_size= 128 , shuffle= True , num_workers=0 )

    # test_data = torchvision.datasets.CIFAR10(root = "./data" , train = False ,download = False,
    #                                           transform = trans)
    test_data = torchvision.datasets.ImageFolder(root = "./data/val" , transform = data_transform["val"])

    train_size = len(train_data)
    test_size = len(test_data)
    print(train_size)
    print(test_size)
    testdata = DataLoader(dataset = test_data , batch_size= 128 , shuffle= True , num_workers=0 )

    # 设置调用GPU，如果有GPU就调用GPU，如果没有GPU则调用CPU训练模型
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    # Alexnet卷积神经网络
    class alexnet(nn.Module):
        def __init__(self):
            super(alexnet , self).__init__()
            self.model = nn.Sequential(

                nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 120, 120]  output[48, 55, 55]

                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
                nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]

                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
                nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]

                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]

                nn.ReLU(inplace=True),
                nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]

                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 7),

            )
        def forward(self , x):
            x = self.model(x)
            return x

    # 启动模型，将模型放入GPU
    alexnet1 = alexnet()
    print(alexnet1)
    alexnet1.to(device)
    test1 = torch.ones(64, 3, 120, 120)

    test1 = alexnet1(test1.to(device))
    print(test1.shape)

    # zhai = xiaozhai()
    # print(zhai)
    # test1 = torch.ones(64, 3, 120, 120)
    # test1 = zhai(test1)
    # print(test1.shape)

    # 设置训练需要的参数，epoch，学习率learning 优化器。损失函数。
    epoch  = 100
    learning = 0.001
    optimizer = torch.optim.Adam(alexnet1.parameters(), lr = learning)
    loss = nn.CrossEntropyLoss()

    # 设置四个空数组，用来存放训练集的loss和accuracy测试集的loss和accuracy
    train_loss_all = []
    train_accur_all = []
    test_loss_all = []
    test_accur_all = []

    # 开始训练
    for i in range(epoch):
        train_loss = 0
        train_num = 0.0
        train_accuracy = 0.0
        alexnet1.train()
        train_bar = tqdm(traindata)
        for step , data in enumerate(train_bar):
            img , target = data
            optimizer.zero_grad()
            outputs = alexnet1(img.to(device))

            loss1  = loss(outputs , target.to(device))
            outputs = torch.argmax(outputs, 1)
            loss1.backward()
            optimizer.step()
            train_loss += abs(loss1.item())*img.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            train_accuracy = train_accuracy + accuracy
            train_num += img.size(0)

        print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(i+1 , train_loss/train_num , train_accuracy/train_num))
        train_loss_all.append(train_loss/train_num)
        train_accur_all.append(train_accuracy.double().item()/train_num)
        test_loss = 0
        test_accuracy = 0.0
        test_num = 0
        alexnet1.eval()
        with torch.no_grad():
            test_bar = tqdm(testdata)
            for data in test_bar:
                img , target = data

                outputs = alexnet1(img.to(device))

                loss2 = loss(outputs, target.to(device))
                outputs = torch.argmax(outputs, 1)
                test_loss = test_loss + abs(loss2.item())*img.size(0)
                accuracy = torch.sum(outputs == target.to(device))
                test_accuracy = test_accuracy + accuracy
                test_num += img.size(0)

        print("test-Loss：{} , test-accuracy：{}".format(test_loss / test_num, test_accuracy / test_num))
        test_loss_all.append(test_loss/test_num)
        test_accur_all.append(test_accuracy.double().item()/test_num)
    plt.figure(figsize=(12,4))
    plt.subplot(1 , 2 , 1)
    plt.plot(range(epoch) , train_loss_all,
             "ro-",label = "Train loss")
    plt.plot(range(epoch), test_loss_all,
             "bs-",label = "test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(epoch) , train_accur_all,
             "ro-",label = "Train accur")
    plt.plot(range(epoch) , test_accur_all,
             "bs-",label = "test accur")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

    torch.save(alexnet1.state_dict(), "alexnet.pth")

    print("模型已保存")

if __name__ == '__main__':
    main()




















