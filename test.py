import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn.functional as fun
# 超参数
batch_size = 64
input_size = 28*28  # 输入数据的size
num_classes = 10
num_epochs = 10
learning_rate = 0.001
hidden_size = 128
train_dataset = datasets.MNIST(root='./home/wang/data_mnist',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./home/wang/data_mnist',
                              train=False,
                              transform=transforms.ToTensor())

# 加载数据/打乱数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

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

        if i % 200 == 0:
            print('current loss == %.4f' % loss.data[0])


correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    labels = Variable(labels)
    outputs = mode1(images)
    loss = criterion(outputs, labels)
    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).sum()
    accuracy = 100 * correct / total
print('loss = %.4f ,Accuracy = %.3f' % (loss, accuracy))