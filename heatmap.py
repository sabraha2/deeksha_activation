import os
from os.path import join
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from backbones import get_model

# Global variables for hook outputs
activations = None
gradients = None

def inference(network, patch_size, stride, model_name, image_path, destination):
    dir_path = '/afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/results/vit_t_dp005_mask0_p28_s28_original/'
    model_path = join(dir_path, model_name, 'model.pt')
    
    net = get_model(network, patch_size=int(patch_size), stride=int(stride),
                    dropout=0.0, fp16=True, num_features=512).cuda()
    
    net.load_state_dict(torch.load(model_path))
    net.eval().cuda()
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).cuda()
    output = net(image.unsqueeze(0))
    return net, output

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach()

if __name__ == '__main__':

    activations = None
    gradients = None

    network = "vit_t_dp005_mask0"
    patch_size = 28
    stride = 28
    model_name = "vit_t_dp005_mask0_p28_s28_original_4"
    image_path = "/store01/flynn/darun/AWE-Ex_New_images_lr/220_R/10.png"
    destination = "./"
    
    net, output = inference(network, patch_size, stride, model_name, image_path, destination)
    

    handle_forward = net.blocks[-1].attn.out_proj.register_forward_hook(forward_hook)
    handle_backward = net.blocks[-1].attn.out_proj.register_full_backward_hook(backward_hook)
    
    pred_class = output.argmax(dim=1).item()
    net.zero_grad()
    score = output[0, pred_class]
    score.backward()
    
    handle_forward.remove()
    handle_backward.remove()
    
    if gradients is None or activations is None:
        raise ValueError("Failed to capture gradients or activations via hooks.")
    
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()
    
    img = np.array(Image.open(image_path).convert('RGB').resize((112, 112)))
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Grad-CAM for class {pred_class}')
    plt.savefig("test_cam.png")
