import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import copy

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)
dataloaders = {'train': trainloader, 'val': testloader}
dataset_sizes = {'train': len(trainset), 'val': len(testset)}
print(dataset_sizes)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------展示样例图片--------------
def imshow(img):
    img = img / 2 + 0.5  # 将正则化后图片恢复
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,out_channels=16,kernel_size=5), #input:3x32x32 -> size:28
            nn.ReLU(inplace=True),
            nn.Conv2d(16,out_channels=32,kernel_size=5), # 24
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels=64, kernel_size=4,stride=2),  # 24-4+1 /2 = 11
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=3,stride=2),  # 11-3+1 /2 = 5
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=32, kernel_size=5),  # 1
        ) #32x1x1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,out_channels=32,kernel_size=5), # 5
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels=32, kernel_size=3,stride=2),  # 11
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels=32, kernel_size=4, stride=2),  # 24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels=32, kernel_size=5),  # 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels=3, kernel_size=5),  # 32
        )
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)
    imshow(torchvision.utils.make_grid(images))
    epochs = 1
    model = AutoEncoder()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        batch = 0
        for inputs, _ in trainloader:
            inputs = inputs.to(device)
            x_hat = model(inputs)
            loss = criterion(x_hat,inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 20 == 0:
                print(f'Epoch: {epoch},Batch: {batch},Loss:{loss.item()}')
    imshow(torchvision.utils.make_grid(inputs.cpu()))
    imshow(torchvision.utils.make_grid(x_hat.cpu()))
