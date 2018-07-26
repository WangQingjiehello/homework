import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms, datasets

# 超参数
batch_size = 16
input_size = 32*32*3  # 输入数据的size
num_classes = 10
num_epochs = 3
learning_rate = 0.001
hidden_size = 128

transform = transforms.Compose([
        transforms.ToTensor(), # range[0-255] -> [0.0-1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化到[-1,1]
                             ])
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

# 建立神经网络模型
class RNN(nn.Module):
    def __init__(self, input_num, hidden_size, output_num):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_size)  # 输入层到隐藏层
        self.relu = nn.ReLU()   # 激活函数
        self.fc2 = nn.Linear(hidden_size, output_num)  # 隐藏层到输出层

    def forward(self, x):
        out_1 = self.fc1(x)
        out_2 = self.relu(out_1)
        out = self.fc2(out_2)
        return out

mode1 = RNN(input_size, hidden_size, num_classes)
print(mode1)
# 优化模型
criterion = nn.CrossEntropyLoss()  # 多分类用的交叉熵损失函数
optimizer = torch.optim.Adam(mode1.parameters(), lr=learning_rate)   # 优化算法
for epoch in range(num_epochs):
    print('current epoch = %d' % epoch)
    for i, (images, labels) in enumerate(train_loader):   # 取出一个可迭代对象的内容
        images = Variable(images.view(-1, input_size))
        labels = Variable(labels)

        optimizer.zero_grad()  # 把模型的参数梯度置为0
        outputs = mode1(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print('current loss == %.4f' % loss.data[0])


correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
for images, labels in test_loader:
    images = Variable(images.view(-1, 32*32*3))
    labels = Variable(labels)
    outputs = mode1(images)
    loss = criterion(outputs, labels)
    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()
    accuracy = 100 * correct / total
print('loss = %.4f ,Accuracy = %.3f' % (loss, accuracy))