import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """Some Information about Net"""
    def __init__(self, n_class):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #insize:32  outsize: 32-5+1=28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  #14
        x = self.pool(F.relu(self.conv2(x))) # 14-5+1=10 /2 =5
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x