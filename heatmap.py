import os
from os.path import join
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from backbones import get_model 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def inference(network, patch_size, stride, model_name, image_path, destination):
    dir_path = '/afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/results/vit_t_dp005_mask0_p28_s28_original/'
    model_path = join(dir_path, model_name, 'model.pt')
    
    # Build model exactly as during training.
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
    image_tensor = transform(image).unsqueeze(0).cuda()
    return net, image_tensor

if __name__ == '__main__':
    network = "vit_t_dp005_mask0"
    patch_size = 28
    stride = 28
    model_name = "vit_t_dp005_mask0_p28_s28_original_4"
    image_path = "/store01/flynn/darun/AWE-Ex_New_images_lr/220_R/10.png"
    destination = "./"
    
    net, img_tensor = inference(network, patch_size, stride, model_name, image_path, destination)
    
    # Patch the patch embedding forward function to output a 4D tensor.
    # The original output is [B, num_patches, C] (here, [B, 16, 256]),
    # so we reshape it to [B, C, grid, grid] (i.e. [B, 256, 4, 4]).
    old_forward = net.patch_embed.forward
    def new_forward(x):
        out = old_forward(x)  # shape: [B, N, C]
        B, N, C = out.shape
        grid_size = int(np.sqrt(N))
        out = out.transpose(1, 2).reshape(B, C, grid_size, grid_size)
        return out
    net.patch_embed.forward = new_forward
    
    # Prepare the original image for visualization.
    img = Image.open(image_path).convert("RGB").resize((112,112))
    img_np = np.array(img) / 255.0
    
    # Use the (patched) patch embedding layer as the target layer.
    target_layer = net.patch_embed
    
    # Create the GradCAM object (without use_cuda since it’s not supported).
    cam_extractor = GradCAM(model=net, target_layers=[target_layer])
    
    # Compute the CAM. The output is a 2D heatmap.
    grayscale_cam = cam_extractor(input_tensor=img_tensor)[0, :]
    
    # Overlay the CAM on the image.
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    plt.imshow(visualization)
    plt.axis("off")
    plt.title("Grad-CAM")
    plt.show()
