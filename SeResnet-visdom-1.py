import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse
from torchvision import datasets
import visdom
import numpy as np
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

viz = visdom.Visdom()
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.stride = stride
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out1 = self.left(x)
        fc1 = nn.MaxPool2d(kernel_size = out1.size(2))
        out = fc1(out1)
        out = out.view(out.size(0), -1)
        fc2 = nn.Linear(self.outchannel, self.outchannel // 16)
        out = fc2(out)
        fc3 = nn.ReLU(True)
        out = fc3(out)
        fc4 = nn.Linear(self.outchannel // 16, self.outchannel)
        out = fc4(out)
        out = out.view(out1.size(0), out1.size(1), 1, 1)
        out = out * out1
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
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

    return ResNet(ResidualBlock)

net = ResNet18().to(device)

def train(model, data, target, lossfc, optimizer):
	model.train()
	optimizer.zero_grad()
	output = model(data)
	loss = lossfc(output, target)
	loss.backward()
	optimizer.step()

	predictions = output.max(1, keepdim = True)[1]
	correct = predictions.eq(target.view_as(predictions)).sum().item()
	Acc = correct / len(target)
	return Acc, loss

def test(model, testloader, lossfc, use_cuda):
    model.eval()
    sum_loss = 0
    sum_Acc = 0
    number = 0
    with torch.no_grad():
    	for data, target in testloader:
    		number += 1
    		data, target = data.to(device), target.to(device)
    		if use_cuda:
    			data = data.cuda()
    			target = target.cuda()
    		output = model(data)
    		loss = lossfc(output, target)
    		predictions = output.max(1, keepdim = True)[1]
    		correct = predictions.eq(target.view_as(predictions)).sum().item()
    		Acc = correct / len(target)
    		sum_loss += loss
    		sum_Acc += Acc
    return sum_Acc / number, sum_loss / number

def main():
    EPOCH = 40   #遍历数据集次数
    BATCH_SIZE = 8
    LR = 0.001
    use_cuda = torch.cuda.is_available()
    transform_train = transforms.Compose([
    	transforms.RandomCrop(32, padding = 4),
    	transforms.RandomHorizontalFlip(),
    	transforms.ToTensor(),
    	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    	])
    transform_test = transforms.Compose([
    	transforms.ToTensor(),
    	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    	])
    trainset = torchvision.datasets.CIFAR100(root='/home/wang/data100', train=True, download=True, transform=transform_train) #训练数据集
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练
    testset = torchvision.datasets.CIFAR100(root='/home/wang/data100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=2)
    model = net
    if use_cuda:
    	model = model.cuda()		
    criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    step = 0
    test_number = 0
    x, train_Acc, test_Acc = 0, 0 ,0
    win = viz.line(
    	X = np.array([x]),
    	Y = np.column_stack((np.array([train_Acc]), np.array([test_Acc]))),
    	opts = dict(
    		legend = ["train_Acc", "test_Acc"]
    		)
    	)
    for i in range(EPOCH):
    	for data, target in trainloader:
    		step += 1
    		x = step
    		data, target = data.to(device), target.to(device)
    		if use_cuda:
    			data = data.cuda()
    			target = target.cuda()
    		Acc, loss = train(model, data, target, criterion, optimizer)
    		train_Acc = Acc
    		print ('train : step = %d, loss = %.4f, Acc = %.2f' %(step, loss, 100 * Acc))
    		if step % 6250 == 0:
    			test_number += 1
    			Acc, loss = test(model, testloader, criterion, use_cuda)
    			test_Acc = Acc
    			print('Test: test_number = %d, loss = %.4f, Acc = %.2f' %(test_step, loss, 100 * Acc))
    		if step % 100 == 0:
    			viz.line(
    				X = np.array([x]),
    				Y = np.column_stack((np.array([train_Acc]), np.array([test_Acc]))),
    				win = win,
    				update = "append"
    				)

if __name__ == '__main__':
	main()
	














