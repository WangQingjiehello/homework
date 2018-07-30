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
num_epochs = 6   #遍历数据集次数
BATCH_SIZE = 10      #批处理尺寸(batch_size)
learning_rate = 0.001

# 准备数据集并预处理
transform = transforms.Compose([
        transforms.ToTensor(), # range[0-255] -> [0.0-1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化到[-1,1]
                             ])

# downloads data
train_set = datasets.CIFAR10(
                    root='/home/wang/data/',
                    train=True,
                    download=True,
                    transform=transform)

train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=4,
                    shuffle=True,
                    num_workers=2)
test_set = datasets.CIFAR10(
                    '/home/wang/data/',
                    train=False,
                    download=True,
                    transform=transform)

test_loader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=4,
                    shuffle=False,
                    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Block(nn.Module):
    def __init__(self, inchannel, outchannel1,outchannel, stride=1):
        super(Block, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel1, outchannel1, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel1, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel)

        )
        self.shortcut = nn.Sequential()
        if  stride != 1 or inchannel != outchannel:
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
        self.layer1 = self.make_layer(Block, 64, 64,256, 1, stride=1)
        self.layer11 = self.make_layer(Block, 256, 64,256, 2, stride=1)
        self.layer2 = self.make_layer(Block, 256, 128, 512, 1, stride=2)
        self.layer22 = self.make_layer(Block, 512, 128, 512, 3, stride=1)
        self.layer3 = self.make_layer(Block, 512, 256, 1024, 1, stride=2)
        self.layer33 = self.make_layer(Block, 1024, 256, 1024, 5, stride=1)
        self.layer4 = self.make_layer(Block, 1024, 512, 2048, 1, stride=2)
        self.layer44 = self.make_layer(Block, 2048, 512, 2048, 2, stride=1)

        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, block, inchannel,outchannel1,outchannel, numbers, stride):
        
        layers = []
        for i in range(numbers):
            layers.append(block(inchannel, outchannel1, outchannel, stride))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer11(out)
        out = self.layer2(out)
        out = self.layer22(out)
        out = self.layer3(out)
        out = self.layer33(out)
        out = self.layer4(out)
        out = self.layer44(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet50():

    return ResNet(Block)

model = ResNet50()
if args.cuda:
    model=model.cuda()

criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # 优化算法

# 训练
correct = 0
total = 0
for epoch in range(num_epochs):
    print('current epoch = %d' % epoch)

    for i, (images, labels) in enumerate(train_loader):   # 取出一个可迭代对象的内容
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
        _, predicts = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum()
        accuracy = 100 * correct / total
        if i % 100 == 0:
            print('current loss == %.4f , accuracy == %.2f' % (loss.data[0],accuracy))


print('Training Finished')


#test
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
for images, labels in test_loader:
    images = Variable(images)
    labels = Variable(labels)
    outputs = model(images)
    loss = criterion(outputs, labels)
    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()
    accuracy = 100 * correct / total
print('loss = %.4f ,Accuracy = %.3f' % (loss, accuracy))