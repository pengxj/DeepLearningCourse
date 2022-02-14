import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from model import LeNet


img = Image.open('../data/smokedata/nn401.jpg')
img.show()
mytransform = transforms.Compose(
        [transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# -----------模型推理--------------
model = LeNet(n_class=2)
model.load_state_dict(torch.load('bestmodel.pt'))
img = mytransform(img)
img = img.unsqueeze(0) # torch要求是4D tensor  NCHW
model.eval() # 一定要设置为测试模型，默认是训练模型得不到相应的结果
with torch.no_grad():  # 无梯度计算能够加速推理
    result = model(img)[0]
classes = ['non', 'smoke']
ind = torch.argmax(result)
print(f'index: {ind},score:{result[ind]} {classes[ind]}')