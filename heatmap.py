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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    return net, image_tensor, image

def forward_hook(layer_name):
    def hook(module, input, output):
        activations[layer_name] = output.detach()
    return hook

def register_hooks(net):
    handles = []
    for name, module in net.named_modules():
        handle = module.register_forward_hook(forward_hook(name))
        handles.append(handle)
    return handles

def remove_hooks(handles):
    for handle in handles:
        handle.remove()

if __name__ == '__main__':
    network = "vit_t_dp005_mask0"
    patch_size = 28
    stride = 28
    model_name = "vit_t_dp005_mask0_p28_s28_original_4"
    image_path = "/store01/flynn/darun/AWE-Ex_New_images_lr/220_R/10.png"
    
    net, img_tensor, orig_img = inference(network, patch_size, stride, model_name, image_path)
    
    hooks = register_hooks(net)
    
    with torch.no_grad():
        output = net(img_tensor)
    
    remove_hooks(hooks)
    
    averaged_activation = None
    valid_layers = 0
    
    for layer_name, activation in activations.items():
        if "attn" not in layer_name:
            logging.info(f"Skipping {layer_name} as it is not an attention layer.")
            continue
        
        if activation is None or activation.numel() == 0:
            logging.error(f"No activations captured for {layer_name}, skipping.")
            continue
        
        activation = activation.squeeze().cpu().numpy()
        
        logging.info(f"Layer: {layer_name}, Activation shape: {activation.shape}")
        
        if activation.ndim == 1 or activation.ndim > 3:
            logging.warning(f"Skipping visualization for {layer_name} due to incompatible shape {activation.shape}.")
            continue
        
        # Compute mean activation over all channels
        if activation.ndim == 3:
            activation = activation.mean(axis=0)
        
        # Normalize activation map
        activation = activation - activation.min()
        if activation.max() > 0:
            activation = activation / activation.max()
        else:
            logging.warning(f"Skipping {layer_name} due to zero max activation.")
            continue
        
        # Ensure activation can be resized
        if activation.shape[0] < 2 or activation.shape[1] < 2:
            logging.warning(f"Skipping {layer_name} due to small activation shape {activation.shape}.")
            continue
        
        try:
            # Resize activation map to match image dimensions
            activation = cv2.resize(activation, (orig_img.size[0], orig_img.size[1]))
        except cv2.error as e:
            logging.error(f"OpenCV error resizing activation for {layer_name}: {e}")
            continue
        
        if averaged_activation is None:
            averaged_activation = activation
        else:
            averaged_activation += activation
        valid_layers += 1
    
    if averaged_activation is not None and valid_layers > 0:
        averaged_activation /= valid_layers
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * averaged_activation), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)
        
        # Save and show the overlayed image
        plt.figure(figsize=(8,6))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Averaged Activation Overlay for Attention Layers")
        plt.savefig("activation_overlay_attention_layers.png")
