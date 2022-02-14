import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import copy

from model import LeNet

# --------展示样例图片--------------
def imshow(img):
    img = img / 2 + 0.5     # 将正则化后图片恢复
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__=='__main__':
    mytransform = transforms.Compose(
        [transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.ImageFolder(root='../data/smokedata/train',transform=mytransform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=0)
    testset = torchvision.datasets.ImageFolder(root='../data/smokedata/val',transform=mytransform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)
    dataloaders = {'train': trainloader, 'val': testloader}
    dataset_sizes = {'train': len(trainset), 'val': len(testset)}
    classes = trainset.classes
    print(classes)
    print(dataset_sizes)
    # -----------查看样例-------
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    # -----------模型定义----------
    model = LeNet(n_class=2)
    model.cuda()
    writer = SummaryWriter()
    # -----定义优化器和Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # -----训练--验证-----
    num_epochs = 100
    device = 'cuda:0'  # 'cpu'
    best_acc = 0
    for epoch in range(num_epochs):
        message = 'Epoch {}/{} '.format(epoch + 1, num_epochs)
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
                if batch % 20 == 0:
                    print(f'Epoch: {epoch},Batch: {batch},Loss:{loss.item()}')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                writer.add_scalars('Acc', {'Train':epoch_acc}, global_step=epoch)
                writer.add_scalars('Loss', {'Train': epoch_loss},global_step=epoch)
            else:
                writer.add_scalars('Acc', {'Val': epoch_acc}, global_step=epoch)
                writer.add_scalars('Loss', {'Val': epoch_loss}, global_step=epoch)
            message += ' {} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc)
            print(message)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts,'bestmodel.pt')
        print(message)
