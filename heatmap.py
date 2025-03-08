import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

model_path = "/afs/crc.nd.edu/user/d/darun/if-copy/recognition/arcface_torch/results/vit_t_dp005_mask0_p28_s28_original_4/model.pt"
model = torch.load(model_path)
model.eval()

print(model)
target_layer = model.blocks[-1].attn

activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach()

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

img = cv2.imread("/store01/flynn/darun/AWE-Ex_New_images_lr/220_R/10.png")

img = cv2.resize(img, (224, 224))  
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
img_tensor.requires_grad = True

output = model(img_tensor)
pred_class = output.argmax(dim=1).item()

model.zero_grad()
score = output[0, pred_class]
score.backward()

weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
cam = torch.sum(weights * activations, dim=1).squeeze()


cam = F.relu(cam)

cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)
cam = cam.cpu().numpy()


cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f'Grad-CAM for class {pred_class}')
plt.savefig("test_cam.png")
