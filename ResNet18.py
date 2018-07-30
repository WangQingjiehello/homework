import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse
from torchvision import datasets

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', # batch_size参数，如果想改，如改成128可这么写：python main.py -batch_size=128
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',# test_batch_size参数，
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, # GPU参数，默认为False
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', # 跑多少次batch进行一次日志记录
                    help='how many batches to wait before logging training status')
args = parser.parse_args()  # 这个是使用argparse模块时的必备行，将参数进行关联，详情用法请百度 argparse 即可
args.cuda = not args.no_cuda and torch.cuda.is_available()  # 这个是在确认是否使用gpu的参数,比如
 
torch.manual_seed(args.seed) # 设置一个随机数种子，相关理论请自行百度或google，并不是pytorch特有的什么设置
if args.cuda:
    torch.cuda.manual_seed(args.seed) # 这个是为GPU设置一个随机数种子
 
 
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# 超参数设置
num_epochs = 5   #遍历数据集次数
BATCH_SIZE = 10      #批处理尺寸(batch_size)
learning_rate = 0.001

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/wang/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   
testset = torchvision.datasets.CIFAR10(root='/home/wang/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(Block, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, Block, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(Block, 64,  2, stride=1)
        self.layer2 = self.make_layer(Block, 128, 2, stride=2)
        self.layer3 = self.make_layer(Block, 256, 2, stride=2)
        self.layer4 = self.make_layer(Block, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(Block)

model = ResNet18()
if args.cuda:
    model=model.cuda()

criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # 优化算法

# 训练
number = 1
for epoch in range(num_epochs):
    print('current epoch = %d' % epoch)
    loss_total = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(trainloader):   # 取出一个可迭代对象的内容
        if args.cuda:
            images=images.cuda()
            labels=labels.cuda()
        images = Variable(images)
        labels = Variable(labels)

        optimizer.zero_grad()  # 把模型的参数梯度置为0
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        _, predicts = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum()
        accuracy = 100 * correct / total
      
        print('[ %d ] current loss == %.4f , accuracy == %.2f' % (number, loss_total / (i + 1), accuracy))
        number += 1

