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
        pos_embed_checkpoint = state_dict['pos_embed']  # [1, N, C]
        pos_embed_model = model.pos_embed                # [1, N_new, C]
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            num_tokens_checkpoint = pos_embed_checkpoint.shape[1]
            num_tokens_model = pos_embed_model.shape[1]
            # If there is a class token and the rest form a perfect square:
            if num_tokens_checkpoint > 1 and int(np.sqrt(num_tokens_checkpoint - 1))**2 == (num_tokens_checkpoint - 1):
                cls_token = pos_embed_checkpoint[:, :1, :]
                pos_tokens = pos_embed_checkpoint[:, 1:, :]
                old_size = int(np.sqrt(pos_tokens.shape[1]))
                new_size = int(np.sqrt(num_tokens_model - 1))
                pos_tokens = pos_tokens.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
                state_dict['pos_embed'] = torch.cat([cls_token, pos_tokens], dim=1)
            else:
                old_size = int(np.sqrt(num_tokens_checkpoint))
                new_size = int(np.sqrt(num_tokens_model))
                pos_embed_checkpoint = pos_embed_checkpoint.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
                pos_embed_checkpoint = torch.nn.functional.interpolate(pos_embed_checkpoint, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
                state_dict['pos_embed'] = pos_embed_checkpoint
    return state_dict

def resize_patch_embed_weight(state_dict, model):
    if 'patch_embed.linear.weight' in state_dict:
        weight_checkpoint = state_dict['patch_embed.linear.weight']  # [out_dim, old_patch_dim*3]
        weight_model = model.patch_embed.linear.weight                # [out_dim, new_patch_dim*3]
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
        weight_checkpoint = state_dict['feature.0.weight']  # [out_dim, old_dim]
        weight_model = model.feature[0].weight                # [out_dim, new_dim]
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
    
    # Attempt to force the desired configuration.
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
    image_cv = cv2.cvtColor(np.array(image.resize((112,112))), cv2.COLOR_RGB2BGR)
    image_tensor = transform(image).unsqueeze(0).cuda()
    
    output = net(image_tensor)
    return net, output, image_cv, image_tensor

def generate_gradcam(net, image_tensor, pred_class=None):
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    # Use full backward hook to capture all grad outputs.
    def full_backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    target_layer = net.blocks[-1].attn
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(full_backward_hook)
    
    output = net(image_tensor)
    if pred_class is None:
        pred_class = output.argmax(dim=1).item()
    
    net.zero_grad()
    score = output[0, pred_class]
    score.backward()
    
    handle_forward.remove()
    handle_backward.remove()
    
    num_tokens = activations.size(1)
    # If there's a CLS token and removing it forms a perfect square, do so.
    if num_tokens > 1 and int(np.sqrt(num_tokens - 1))**2 == (num_tokens - 1):
        activations = activations[:, 1:, :]
        gradients = gradients[:, 1:, :]
        num_tokens = activations.size(1)
    
    # If tokens don't form a perfect square, pad them.
    grid_side = int(np.ceil(np.sqrt(num_tokens)))
    expected_tokens = grid_side * grid_side
    if expected_tokens != num_tokens:
        pad_tokens = expected_tokens - num_tokens
        activations = torch.cat([activations, torch.zeros(activations.size(0), pad_tokens, activations.size(2), device=activations.device)], dim=1)
        gradients = torch.cat([gradients, torch.zeros(gradients.size(0), pad_tokens, gradients.size(2), device=gradients.device)], dim=1)
        num_tokens = activations.size(1)
    
    grid_size = int(np.sqrt(num_tokens))
    activations = activations.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
    gradients = gradients.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
    
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()[0]
    
    if cam.size == 0:
        print("Activations shape:", activations.shape)
        print("Gradients shape:", gradients.shape)
        raise ValueError("Grad-CAM computation failed: cam is empty.")
    
    cam = cv2.resize(cam, (112, 112))
    return cam, pred_class

if __name__ == "__main__":
    network = "vit_t" 
    patch_size = 28    # Training configuration: patch_size=28
    stride = 28        # Training configuration: stride=28
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
