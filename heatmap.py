import os
from os.path import join
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from backbones import get_model

# Global variables to store hook outputs.
activations = {}
gradients = {}

def inference(network, patch_size, stride, model_name, image_path):
    dir_path = '/afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/results/vit_t_dp005_mask0_p28_s28_original/'
    model_path = join(dir_path, model_name, 'model.pt')
    net = get_model(network, patch_size=int(patch_size), stride=int(stride),
                    dropout=0.0, fp16=True, num_features=512).cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).cuda()
    return net, image_tensor

def forward_hook(layer_name):
    def hook(module, input, output):
        if output.ndim == 3:
            B, N, C = output.shape
            grid_size = int(np.sqrt(N))
            output = output.transpose(1,2).reshape(B, C, grid_size, grid_size)
        activations[layer_name] = output.detach()
    return hook

def backward_hook(layer_name):
    def hook(module, grad_in, grad_out):
        out = grad_out[0]
        if out.ndim == 3:
            B, N, C = out.shape
            grid_size = int(np.sqrt(N))
            out = out.transpose(1,2).reshape(B, C, grid_size, grid_size)
        gradients[layer_name] = out.detach()
    return hook

def register_hooks(net, target_layers):
    handles = []
    for name, layer in target_layers.items():
        handle_forward = layer.register_forward_hook(forward_hook(name))
        handle_backward = layer.register_full_backward_hook(backward_hook(name))
        handles.append((handle_forward, handle_backward))
    return handles

def remove_hooks(handles):
    for handle_forward, handle_backward in handles:
        handle_forward.remove()
        handle_backward.remove()

if __name__ == '__main__':
    network = "vit_t_dp005_mask0"
    patch_size = 28
    stride = 28
    model_name = "vit_t_dp005_mask0_p28_s28_original_4"
    image_path = "/store01/flynn/darun/AWE-Ex_New_images_lr/220_R/10.png"
    
    net, img_tensor = inference(network, patch_size, stride, model_name, image_path)
    
    target_layers = {
        "patch_embed": net.patch_embed.linear,
        "block_4_attn": net.blocks[4].attn.proj,
        "block_8_mlp": net.blocks[8].mlp.fc2
    }
    
    hooks = register_hooks(net, target_layers)
    
    output = net(img_tensor)
    pred_class = output.argmax(dim=1).item()
    net.zero_grad()
    score = output[0, pred_class]
    score.backward()
    
    remove_hooks(hooks)
    
    for layer_name, grad in gradients.items():
        if grad is None or activations[layer_name] is None:
            raise ValueError(f"Failed to capture gradients or activations for {layer_name}.")
        
        weights = torch.mean(grad, dim=[2,3], keepdim=True)
        cam = torch.sum(weights * activations[layer_name], dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        
        img = np.array(Image.open(image_path).convert("RGB").resize((112,112)))
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)
        
        plt.figure(figsize=(8,6))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Grad-CAM for {layer_name} - class {pred_class}")
        plt.savefig(f"gradcam_{layer_name}.png")
        plt.show()
