from PIL import Image
import torch
import os
import os.path


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = Image.open(img_path).convert('RGB')  # 读取该图片
        label = img_path.split('\\')[-1].split('.')[0]
        if self.transform:
            img = self.transform(img)  # 对样本进行变换

        return img, label  # 返回该样本
