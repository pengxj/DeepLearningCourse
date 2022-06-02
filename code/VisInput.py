import torch
import torchvision
import PIL
import torch.nn.functional as F
import numpy
from matplotlib import cm
#CAM
def hook_store_A(module, input, output):
    module.A = output[0]
def hook_store_dydA(module, grad_input, grad_output):
    module.dydA = grad_output[0]

if __name__ == "__main__":
    model = torchvision.models.vgg19(pretrained=True)

    to_tensor = torchvision.transforms.ToTensor()
    img = PIL.Image.open('elephant_hippo.jpeg')
    input = to_tensor(img).unsqueeze(0)
    layer = model.features[35]
    layer.register_forward_hook(hook_store_A)
    layer.register_backward_hook(hook_store_dydA)

    output = model(input)
    c = 386  # African elephant
    output[0, c].backward()
    alpha = layer.dydA.mean((2, 3), keepdim=True)
    L = torch.relu((alpha * layer.A).sum(1, keepdim=True))
    L = L / L.max()
    L = F.interpolate(L, size=(input.size(2), input.size(3)),
                      mode='bilinear', align_corners=False)
    l = L.view(L.size(2), L.size(3)).detach().numpy()
    PIL.Image.fromarray(numpy.uint8(cm.gist_earth(l) * 255)).save('result.png')

    res = PIL.Image.open('result.png')
    img=img.convert('RGBA')
    merge_res = PIL.Image.blend(img, res, 0.8)
    merge_res.save('result-merge.png')