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

def resize_pos_embed(state_dict, model):
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']  # shape: [1, old_num, C]
        pos_embed_model = model.pos_embed                # shape: [1, new_num, C]
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            cls_token = pos_embed_checkpoint[:, :1, :]
            pos_tokens = pos_embed_checkpoint[:, 1:, :]
            old_num = pos_tokens.shape[1]
            new_num = pos_embed_model.shape[1] - 1
            old_size = int(np.sqrt(old_num))
            new_size = int(np.sqrt(new_num))
            pos_tokens = pos_tokens.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
            state_dict['pos_embed'] = torch.cat([cls_token, pos_tokens], dim=1)
    return state_dict

def resize_patch_embed_weight(state_dict, model):
    if 'patch_embed.linear.weight' in state_dict:
        weight_checkpoint = state_dict['patch_embed.linear.weight']  # shape: [out_dim, old_patch_dim*3]
        weight_model = model.patch_embed.linear.weight                # shape: [out_dim, new_patch_dim*3]
        if weight_checkpoint.shape != weight_model.shape:
            out_dim = weight_checkpoint.shape[0]
            old_patch_dim = weight_checkpoint.shape[1] // 3
            new_patch_dim = weight_model.shape[1] // 3
            old_size = int(np.sqrt(old_patch_dim))
            new_size = int(np.sqrt(new_patch_dim))
            weight_checkpoint = weight_checkpoint.view(out_dim, 3, old_size, old_size)
            weight_resized = torch.nn.functional.interpolate(
                weight_checkpoint, size=(new_size, new_size), mode='bicubic', align_corners=False
            )
            weight_resized = weight_resized.view(out_dim, 3 * new_size * new_size)
            state_dict['patch_embed.linear.weight'] = weight_resized
    return state_dict

def resize_feature_weight(state_dict, model):
    if 'feature.0.weight' in state_dict:
        weight_checkpoint = state_dict['feature.0.weight']  # shape: [out_dim, old_dim]
        weight_model = model.feature[0].weight                # shape: [out_dim, new_dim]
        if weight_checkpoint.shape != weight_model.shape:
            out_dim = weight_checkpoint.shape[0]
            old_dim = weight_checkpoint.shape[1]
            new_dim = weight_model.shape[1]
            old_size = int(np.sqrt(old_dim))
            new_size = int(np.sqrt(new_dim))
            weight_checkpoint = weight_checkpoint.view(out_dim, 1, old_size, old_size)
            weight_resized = torch.nn.functional.interpolate(
                weight_checkpoint, size=(new_size, new_size), mode='bicubic', align_corners=False
            )
            weight_resized = weight_resized.view(out_dim, new_size * new_size)
            state_dict['feature.0.weight'] = weight_resized
    return state_dict

def inference(network, patch_size, stride, model_name, image_path, destination):
    dir_path = '/afs/crc.nd.edu/user/d/darun/if-copy/recognition/arcface_torch/results/'
    model_path = join(dir_path, model_name, 'model.pt')
    
    # Build model with desired patch configuration. 
    # (If get_model ignores your input, consider modifying it so that the values are used.)
    net = get_model(network, patch_size=(patch_size, patch_size), stride=(stride, stride), 
                    dropout=0.0, fp16=True, num_features=512).cuda()
    
    state_dict = torch.load(model_path, map_location="cuda")
    state_dict = resize_pos_embed(state_dict, net)
    state_dict = resize_patch_embed_weight(state_dict, net)
    state_dict = resize_feature_weight(state_dict, net)
    
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
    patch_size = 28    # training used patch_size=28
    stride = 28        # training used stride=28
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
