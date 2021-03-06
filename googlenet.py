import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse
from torchvision import datasets

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

EPOCH = 4   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 4      #批处理尺寸(batch_size)
LR = 0.001        #学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/wang/data', train=True, download=True, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='/home/wang/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Inception(nn.Module):
    def __init__(self, inchannel, outchannel1, out1, outchannel2, out2, outchannel3, outchannel4):
        super(Inception, self).__init__()
        self.a1 = nn.Sequential(
        	nn.Conv2d(inchannel, outchannel1, kernel_size = 1),
        	nn.BatchNorm2d(outchannel1),
        	nn.ReLU(True)
        	)

        self.a2 = nn.Sequential(
        	nn.Conv2d(inchannel, out1, kernel_size = 1),
        	nn.BatchNorm2d(out1),
        	nn.ReLU(True),
        	nn.Conv2d(out1, outchannel2, kernel_size = 3, padding = 1),
        	nn.BatchNorm2d(outchannel2),
        	nn.ReLU(True)
        	)

        self.a3 = nn.Sequential(
        	nn.Conv2d(inchannel, out2, kernel_size = 1),
        	nn.BatchNorm2d(out2),
        	nn.ReLU(True),
        	nn.Conv2d(out2, outchannel3, kernel_size = 3, padding = 1),
        	nn.BatchNorm2d(outchannel3),
        	nn.ReLU(True),
        	nn.Conv2d(outchannel3, outchannel3, kernel_size = 3, padding = 1),
        	nn.BatchNorm2d(outchannel3),
        	nn.ReLU(True)
        	)

        self.a4 = nn.Sequential(
        	nn.MaxPool2d(3, stride = 1, padding = 1),
        	nn.Conv2d(inchannel, outchannel4, kernel_size = 1),
        	nn.BatchNorm2d(outchannel4),
        	nn.ReLU(True)
        	)

    def forward(self, x):
    	output1 = self.a1(x)
    	output2 = self.a2(x)
    	output3 = self.a3(x)
    	output4 = self.a4(x)
    	return torch.cat([output1, output2, output3, output4], 1)
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.layer2 = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.layer3 = nn.AvgPool2d(8, stride = 1)
        self.layer4 = nn.Linear(1024, 10)


    def forward(self, x):
        out = self.layer1(x)
        out = self.a3(out)
        out = self.a4(out)
        out = self.layer2(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.layer2(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        return out
net = GoogLeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
number = 1
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('epoch == %d current loss == %.4f , accuracy == %.2f'
                          % (number, sum_loss / (i + 1), 100. * correct / total))
                    number += 1
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

 


#test
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
for images, labels in testloader:
    images = Variable(images)
    labels = Variable(labels)
    outputs = net(images)
    loss = criterion(outputs, labels)
    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()
    accuracy = 100 * correct / total
print('loss = %.4f ,Accuracy = %.3f' % (loss, accuracy))




