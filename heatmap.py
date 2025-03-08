import os
from os.path import join
import torch
from torchvision import transforms
from PIL import Image
from backbones import get_model  

def inference(network, patch_size, stride, model_name, image_path, destination):
    dir_path = '/afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/results/'
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
    image = transform(image).cuda()
    output = net(image.unsqueeze(0))
    return net, output

if __name__ == '__main__':
    network = "vit_t" 
    patch_size = 28    
    stride = 28      
    model_name = "vit_t_dp005_mask0_p28_s28_original_4"
    image_path = "/store01/flynn/darun/AWE-Ex_New_images_lr/220_R/10.png"  # update with your actual path
    destination = "./"
    
    net, output = inference(network, patch_size, stride, model_name, image_path, destination)
    print("Output shape:", output.shape)
