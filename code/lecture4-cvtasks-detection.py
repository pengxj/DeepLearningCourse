from torchvision import transforms
import torchvision
import torch
import argparse
import cv2
from PIL import Image
import numpy as np
import time

coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def draw_boxes(boxes, classes, labels, scores, image):
    """
    Draws the bounding box around a detected object.
    """
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i]+f'{scores[i]:.2f}', (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='input/image_1.jpg',
                        help='path to input input image')
    parser.add_argument('-t', '--threshold', default=0.5, type=float,
                        help='detection threshold')
    args = vars(parser.parse_args())

    # define the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # load the model
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    # load the model on to the computation device
    model.eval().to(device)

    tensor_a = torch.randn(1,3,300,300).to(device)
    res = model(tensor_a)
    # read the image
    image = Image.open(args['input'])
    # image.show()
    transform = transforms.Compose([
        # transforms.Resize([300,300]),
        transforms.ToTensor(),
    ])
    detection_threshold = args['threshold']
    t_image = transform(image).to(device)
    t_image = t_image.unsqueeze(0)
    s = time.time()
    with torch.no_grad():
        outputs = model(t_image)
    print(f'inference time: {time.time()-s} s')
    # 官方源码情况如下
    # detections.append(
    #     {
    #         "boxes": image_boxes[keep],
    #         "scores": image_scores[keep],
    #         "labels": image_labels[keep],
    #     }
    # get all the predicited class names
    labels =outputs[0]['labels'].cpu().numpy()
    classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    print(pred_scores.shape)
    scores = pred_scores[pred_scores >= detection_threshold]
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)

    # draw bounding boxes
    # image = image.resize((300,300))
    image = draw_boxes(boxes, classes, labels, scores, image)
    save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{''.join(str(args['threshold']).split('.'))}"
    cv2.imshow('Image', image)
    cv2.imwrite(f"{save_name}.jpg", image)
    cv2.waitKey(0)
