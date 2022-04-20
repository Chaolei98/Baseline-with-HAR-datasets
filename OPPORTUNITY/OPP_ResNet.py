import torch
import torch.nn as nn
import torch.utils.data as Data

import os
import numpy as np
from collections import Counter
import torch.nn.functional as F
import argparse


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=256, help='Batch size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--epoch', type=int, default=200, help='Epoch')
args = parser.parse_args()

#load data
train_x = torch.from_numpy(np.load('/Datasets/opportunity/x_train.npy')).float()
train_y = torch.from_numpy(np.load('/Datasets/opportunity/y_train.npy')).long()
test_x = torch.from_numpy(np.load('/Datasets/opportunity/x_test.npy')).float()
test_y = torch.from_numpy(np.load('/Datasets/opportunity/y_test.npy')).long()

train_x = torch.unsqueeze(train_x, 1)
test_x = torch.unsqueeze(test_x, 1)
# train_x = train_x.reshape(train_x.size(0), 1, train_x.size(1), train_x.size(2))
# test_x = test_x.reshape(test_x.size(0), 1, test_x.size(1),test_x.size(2))
num_classes = len(Counter(train_y.tolist()))
len_train, len_test = len(train_y),  len(test_y)

train_dataset = Data.TensorDataset(train_x, train_y)
test_dataset = Data.TensorDataset(test_x, test_y)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=True)


class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, 3, 1, 1),
            nn.BatchNorm2d(output_channel),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
        )
    def forward(self, x):
        identity = self.shortcut(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = x + identity
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(ResNet, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, 3, 2, 1)
        self.layer2 = self._make_layers(64, 128, 3, 2, 1)
        self.layer3 = self._make_layers(128, 256, 3, 2, 1)
        self.fc = nn.Linear(256*2*4, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return BasicBlock(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('aa',x.shape)
        x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 50))
    optimizer.param_groups[0]['lr'] = lr

# model = ResNet(BasicBlock, [2,2,2,2], num_classes)
model = ResNet(input_channel=1, num_classes=num_classes)
model.cuda()
print(model)

loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    # print('LR:',optimizer.param_groups[0]['lr'])
    train_loss = 0
    train_num = 0
    model.train()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        output = model(x)
        loss = loss_f(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred = torch.max(output, 1)[1].cpu().numpy()
        label = y.cpu().numpy()
        train_num += (pred==label).sum()
    train_acc = train_num / len_train
    print('Train Epoch:{} Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch, train_loss/len(train_loader), train_acc),end='||')

def test():
    test_loss = 0
    test_num = 0
    model.eval()

    for step, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        output = model(x)
        loss = loss_f(output, y)

        test_loss += loss.item()

        pred = torch.max(output, 1)[1].cpu().numpy()
        label = y.cpu().numpy()
        test_num += (pred==label).sum()
    test_acc = test_num / len_test
    print('Test Loss:{:.4f} Test Acc:{:.4f}'.format(test_loss/len(test_loader), test_acc))

for epoch in range(args.epoch):
    train(epoch)
    test()
