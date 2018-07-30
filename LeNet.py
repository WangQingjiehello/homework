import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms, datasets


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, 
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', 
                    help='how many batches to wait before logging training status')
args = parser.parse_args()  
args.cuda = not args.no_cuda and torch.cuda.is_available()  
 
torch.manual_seed(args.seed) 
if args.cuda:
    torch.cuda.manual_seed(args.seed)
 
 
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

input_size = 32*32*3
learning_rate = 0.0004
num_classes = 10
num_epochs =  10
batch_size = 16

# deal with data
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

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
# net
class LeNet(nn.Module):
	def __init__(self, class_num = num_classes):
	    super(LeNet, self).__init__()
	    self.conv1 = nn.Conv2d(3,6,5, stride = 1)
	    self.conv2 = nn.Conv2d(6,16,5, stride = 1)
	    self.fc1 = nn.Linear(16*5*5, 120)
	    self.fc2 = nn.Linear(120, 84)
	    self.fc3 = nn.Linear(84, class_num)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size = (2,2), stride = 2)
		x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size = (2,2), stride = 2)
		x = x.view(x.size()[0], -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		output = self.fc3(x)
		return output

model = LeNet(num_classes)
if args.cuda:
    model=model.cuda()

# train
criterion = nn.CrossEntropyLoss()  # 多分类用的交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # 优化算法
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

        if i % 1000 == 0:
            print('current loss == %.4f' % loss.data[0])

print('Training Finished')


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

