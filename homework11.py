import torchvision as tv
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()

transform = transforms.Compose([
        transforms.ToTensor(), # range[0-255] -> [0.0-1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化到[-1,1]
                             ])
train_set = tv.datasets.CIFAR10(
                    root='/home/wang/data/',
                    train=True,
                    download=True,
                    transform=transform)

train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=4,
                    shuffle=True,
                    num_workers=2)
test_set = tv.datasets.CIFAR10(
                    '/home/wang/data/',
                    train=False,
                    download=True,
                    transform=transform)

test_loader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=4,
                    shuffle=False,
                    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        # 两个卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 三个全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = RNN()
print(net)

criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # 优化算法

for epoch in range(3):   # 训练三次

    loss_total = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)   # 获取数据

        optimizer.zero_grad()    # 梯度清零

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()  # 更新参数

        loss_total += loss.data[0]
        if i % 1000 == 0:
            print('loss: %.4f' % (loss_total / 1000))
            loss_total = 0.0
print('训练结束')

# 测试
correct = 0
total = 0
for data in test_loader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
accuracy = 100 * correct / total
print('accuracy is %.2f' % accuracy)
