from torchvision import transforms
import torchvision
import torch
import argparse
import cv2
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

coco_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# create a color pallette, selecting a color for each class
label_map = [
               (0, 0, 0),  # background
               (128, 0, 0), # aeroplane
               (0, 128, 0), # bicycle
               (128, 128, 0), # bird
               (0, 0, 128), # boat
               (128, 0, 128), # bottle
               (0, 128, 128), # bus
               (128, 128, 128), # car
               (64, 0, 0), # cat
               (192, 0, 0), # chair
               (64, 128, 0), # cow
               (192, 128, 0), # dining table
               (64, 0, 128), # dog
               (192, 0, 128), # horse
               (64, 128, 128), # motorbike
               (192, 128, 128), # person
               (0, 64, 0), # potted plant
               (128, 64, 0), # sheep
               (0, 192, 0), # sofa
               (128, 192, 0), # train
               (0, 64, 128) # tv/monitor
]

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]

    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image

def image_overlay(image, segmented_image):
    alpha = 0.7 # how much transparency to apply
    beta = 1 - alpha # alpha + beta should equal 1
    gamma = 0 # scalar added to each sum

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='input/image_1.jpg',
                        help='path to input input image')

    args = vars(parser.parse_args())

    # define the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # load the model
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    # load the model on to the computation device
    model.eval().to(device)
    #
    tensor_a = torch.randn(1,3,300,300).to(device)
    res = model(tensor_a)
    # read the image
    image = Image.open(args['input']).convert('RGB')#.resize([480,480])
    # image.show()
    transform = transforms.Compose([
        # transforms.Resize([300,300]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    t_image = transform(image).to(device)
    t_image = t_image.unsqueeze(0)
    s = time.time()
    with torch.no_grad():
        outputs = model(t_image)['out']
    print(f'inference time: {time.time()-s} s')
    print(outputs.shape)
    # # 官方源码情况如下
    # result = OrderedDict()
    # x = features["out"]
    # x = self.classifier(x)
    # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
    # result["out"] = x
    segmented_image = draw_segmentation_map(outputs)
    final_image = image_overlay(image, segmented_image)
    save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
    # show the segmented image and save to disk
    cv2.imshow('Segmented image', final_image)
    cv2.waitKey(0)
    cv2.imwrite(f"outputs/{save_name}.jpg", final_image)
