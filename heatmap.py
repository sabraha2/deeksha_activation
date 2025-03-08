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

def inference(network, patch_size, stride, model_name, image_path, destination):
    dir_path = '/afs/crc.nd.edu/user/d/darun/if-copy/recognition/arcface_torch/results/'
    model_path = join(dir_path, model_name, 'model.pt')
    
    net = get_model(network, patch_size=(patch_size, patch_size), stride=(stride, stride), 
                    dropout=0.0, fp16=True, num_features=512).cuda()
    
    state_dict = torch.load(model_path, map_location="cuda")
    net.load_state_dict(state_dict)
    net.eval()
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image = Image.open(image_path).convert('RGB')
    image_cv = cv2.cvtColor(np.array(image.resize((112, 112))), cv2.COLOR_RGB2BGR)
    image_tensor = transform(image).unsqueeze(0).cuda()
    
    output = net(image_tensor)
    return net, output, image_cv, image_tensor

def generate_gradcam(net, image_tensor, pred_class=None):
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    target_layer = net.blocks[-1].attn
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)
    
    output = net(image_tensor)
    if pred_class is None:
        pred_class = output.argmax(dim=1).item()
    
    net.zero_grad()
    score = output[0, pred_class]
    score.backward()
    
    handle_forward.remove()
    handle_backward.remove()
    
    if activations.size(1) > 1:
        activations = activations[:, 1:, :]
        gradients = gradients[:, 1:, :]
    
    num_tokens = activations.size(1)
    grid_size = int(num_tokens ** 0.5)
    if grid_size * grid_size != num_tokens:
        raise ValueError("The number of tokens does not form a perfect square grid.")
    
    activations = activations.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
    gradients = gradients.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
    
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()[0]
    
    cam = cv2.resize(cam, (112, 112))
    return cam, pred_class

if __name__ == "__main__":
    network = "vit_t" 
    patch_size = 28    
    stride = 28      
    model_name = "vit_t_dp005_mask0_p28_s28_original_4"
    image_path = "/store01/flynn/darun/AWE-Ex_New_images_lr/220_R/10.png"  
    destination = "./"  

    net, output, image_cv, image_tensor = inference(network, patch_size, stride, model_name, image_path, destination)
    cam, pred_class = generate_gradcam(net, image_tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_cv, 0.5, heatmap, 0.5, 0)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM for class {pred_class}")
    plt.axis("off")
    plt.savefig("test.png")
