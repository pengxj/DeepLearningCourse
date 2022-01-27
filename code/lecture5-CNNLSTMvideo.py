import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import time
from datetime import datetime
import copy
from lecture5dataset import VideoSmoke

class cnnlstm(nn.Module):
    def __init__(self, pretrained_=False):
        super(cnnlstm, self).__init__()
        self.res18 = models.resnet18(pretrained=pretrained_)
        self.res18.fc = nn.Linear(512,256)
        self.lstm = nn.LSTM(256, 256)
        self.classifier = nn.Linear(256,2)

    def forward(self, x):
        x = self.res18(x)
        # N C
        x = x.view(3,-1,256) # T N C
        y, (hx, cx) = self.lstm(x)
        x = y[-1,:,:]
        x = x.squeeze(axis=0)
        x = self.classifier(x)
        return x

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            freq = 20
            for cnt, (inputs, labels) in enumerate(dataloaders[phase]):
                print(inputs.shape)
                n = inputs.shape[0]
                inputs = inputs.view(n*3, 3, 112, 112)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
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
                cnt += 1
                if cnt % freq == 0:
                    print('{}--{}, Epoch: {}  Iter: {} Loss: {:.4f} Acc: {:.4f}'.format(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        phase, epoch, cnt, loss.item(), torch.sum(preds == labels.data) / preds.size(0)))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            torch.save(model.module.state_dict(), f'epoch_{epoch}.pt')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    inputs = torch.randn(9, 3, 112, 112)
    model = cnnlstm()
    outputs = model.forward(inputs)
    print(outputs.size())
    batchsize = 4
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(128),
            transforms.RandomCrop(112),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }
    vsmoke = {x: VideoSmoke(os.path.join(sys.argv[1], x),
                            spatial_transform=data_transforms[x])
              for x in ['train', 'val']}
    dataset_sizes = {x: len(vsmoke[x]) for x in ['train', 'val']}
    print(dataset_sizes)
    dataloaders = {x: torch.utils.data.DataLoader(vsmoke[x], batch_size=batchsize,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,8], gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1, eta_min=0.00001)
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=10)
    torch.save(model.state_dict(), "best.pt")

