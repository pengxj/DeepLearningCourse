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

# ====================================利用已有模型进行前向推理=================================================
# # ----------PIL读取并用预训练模型进行推理-----------
# img = Image.open('data/cat.jpg')
# img.show()
# trans = transforms.Compose([transforms.ToTensor(),
#                             transforms.Resize(224),
#                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# # -----------使用Opencv-----------------
# img = cv2.imread('data/cat.jpg') #BGR
# cv2.imshow('img',img)
# cv2.waitKey(0)
# img = img[:, :, ::-1].copy()  # BGR to RGB
# trans = transforms.Compose([transforms.ToPILImage(),
#                             transforms.ToTensor(),
#                             transforms.Resize(224),
#                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# # -----------模型推理--------------
# model = torchvision.models.resnet18(pretrained=True)
#
# img = trans(img)
# img = img.unsqueeze(0) # torch要求是4D tensor  NCHW
# model.eval() # 一定要设置为测试模型，默认是训练模型得不到相应的结果
# with torch.no_grad():  # 无梯度计算能够加速推理
#     result = model(img)[0]
# classes = [l.rstrip() for l in open('data/imagenet-class.txt', 'r')]
# result = softmax(result, dim=0)
# ind = torch.argmax(result)
# print(f'index: {ind},score:{result[ind]} {classes[ind]}')

# =========================================在CIFAR10对原模型进行finetune=================================================
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
dataloaders = {'train':trainloader, 'val':testloader}
dataset_sizes = {'train': len(trainset), 'val': len(testset)}
print(dataset_sizes)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# --------展示样例图片--------------
def imshow(img):
    img = img / 2 + 0.5     # 将正则化后图片恢复
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)
    # imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(16)))
    model = torchvision.models.resnet18(pretrained=True)
    # print(model)
    inchannel = model.fc.in_features
    model.fc = nn.Linear(inchannel, 10)
    model.cuda()
    # -----定义优化器和Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # -----训练--验证-----
    num_epochs = 20
    device = 'cuda:0'# 'cpu'
    best_acc = 0
    for epoch in range(num_epochs):
        message = 'Epoch {}/{} '.format(epoch+1, num_epochs)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # 当训练时候才使能梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch += 1
                if batch%20==0:
                    print(f'Epoch: {epoch},Batch: {batch},Loss:{loss.item()}')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            message += ' {} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc)
            print(message)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts,'bestmodel.pt')
        print(message)
