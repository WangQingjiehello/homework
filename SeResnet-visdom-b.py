import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import argparse
from torch.nn.utils import clip_grad_norm
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse
import os
import sys
import time
import torch.backends.cudnn as cudnn
from torchvision import datasets
#import visdom
import numpy as np
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description = 'PyTorch CIFAR100 Training')
parser.add_argument('--lf', default = 0.1, type = float, help='LR')
parser.add_argument('--resume', '-r', action = 'store_true', help="resume from checkpoint")
args = parser.parse_args()

#viz = visdom.Visdom()
best_Acc = 0
start_EPOCH = 0
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.Dropout(0.4)
        )
        self.stride = stride
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.shortcut = nn.Sequential()
        self.fc2 = nn.Linear(self.outchannel, self.outchannel // 16)
        self.fc3 = nn.ReLU(True)
        self.fc4 = nn.Linear(self.outchannel // 16, self.outchannel)
        self.fc = nn.Dropout(0.4)
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out1 = self.left(x)
        out = nn.functional.max_pool2d(out1, kernel_size = out1.size(2))
        out = out.view(out.size(0), -1)
        out = self.fc2(out)
        out = self.fc(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc(out)
        out = out.view(out1.size(0), out1.size(1), 1, 1)
        out = out * out1
        out += self.shortcut(x)
        out = self.fc(out)
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
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark == True
# load checkpoi 
args.resume = True
if args.resume:
    print('Resuming from checkpoint....')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_Acc = checkpoint['Acc']
    start_EPOCH = checkpoint['EPOCH']
net = torch.load('./checkpoint/ckpt.t7')
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

def test(model, testloader, lossfc, use_cuda, EPOCH):
    global best_Acc
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
    acc = 100. * sum_Acc / number
    if acc > best_Acc:
        print('Saving model..')
        state = {
        'net': model.state_dict(),
        'Acc': acc,
        'EPOCH': EPOCH,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_Acc = acc
    return best_Acc, acc, sum_loss / number

def main():
    global LR
    BATCH_SIZE = 128
    LR = 0.001
    use_cuda = torch.cuda.is_available()
    transform_train = transforms.Compose([
    	transforms.RandomCrop(32, padding = 4),
    	transforms.RandomHorizontalFlip(),
    	transforms.ToTensor(),
    	transforms.Normalize((0.4385, 0.4181, 0.3776), (0.2571, 0.2489, 0.2413)),
    	])
    transform_test = transforms.Compose([
    	transforms.ToTensor(),
    	transforms.Normalize((0.5008, 0.4874, 0.4419), (0.2019, 0.2000, 0.2036)),
    	])
    trainset = torchvision.datasets.CIFAR100(root='/input/cifar-100-python', train=True, download=True, transform=transform_train) #训练数据集
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练
    testset = torchvision.datasets.CIFAR100(root='/input/cifar-100-python', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    model = net
    if use_cuda:
    	model = model.cuda()		
    criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    step = 0
    test_number = 0
    x, train_Acc, train_loss, test_Acc, test_loss = 0, 0 ,0, 0, 0
    # win = viz.line(
    	# X = np.array([x]),
    	# Y = np.column_stack((np.array([train_Acc]), np.array([test_Acc]))),
    	# opts = dict(
    	#	legend = ["train_Acc", "test_Acc"]
    	#	)
    	#)
    for i in range(start_EPOCH, 180):     
        if i > 80 == 0:
            LR = 0.0001
        print('EPOCH = %d' %i)
        for data, target in trainloader:
            step += 1
            x = step
            data, target = data.to(device), target.to(device)
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            Acc, loss = train(model, data, target, criterion, optimizer)
            train_Acc = Acc
            train_loss = loss
            print ('train : step = %d, loss = %.4f, Acc = %.2f' %(step, loss, 100 * Acc))
            if step % 390 == 0:
                test_number += 1
                best_acc, Acc, loss = test(model, testloader, criterion, use_cuda, i)
                test_Acc = Acc
                test_loss = loss
                print('Test: test_number = %d, loss = %.4f, current_acc = %.2f, best_Acc = %.2f' %(test_number, loss, Acc, best_acc))
            #if step % 100 == 0:
                #viz.line(
                    #X = np.array([x]),
                    #Y = np.column_stack((np.array([train_Acc]), np.array([test_Acc]))),
                    #win = win,
                    #update = "append"
                    #)
    print('Test: loss = %.4f, best_Acc = %.2f' %(loss, best_acc))
if __name__ == '__main__':
	main()




