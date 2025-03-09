import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from backbones import get_model 

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


net = get_model("vit_t_dp005_mask0", 
                patch_size=(28, 28), 
                stride=(28, 28), 
                dropout=0.0, 
                fp16=True, 
                num_features=512).cuda()
state_dict = torch.load("/afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/results/vit_t_dp005_mask0_p28_s28_original/model.pt", map_location="cuda")
net.load_state_dict(state_dict)
net.eval()


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
image_path = "/store01/flynn/darun/AWE-Ex_New_images_lr/220_R/10.png"
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).cuda()

img_np = np.array(img.resize((112, 112))) / 255.0

target_layer = net.patch_embed


cam_extractor = GradCAM(model=net, target_layers=[target_layer], use_cuda=True)
grayscale_cam = cam_extractor(input_tensor=img_tensor)[0, :]


visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.axis("off")
plt.title("Grad-CAM")
plt.show()
