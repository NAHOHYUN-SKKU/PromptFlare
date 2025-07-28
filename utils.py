from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
    
def prepare_image(image):
    image = image.resize((512, 512), resample=Image.BICUBIC)
    image = np.array(image)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).to(dtype=torch.half) / 127.5 - 1.0
    
    return image

def prepare_mask(mask):
    mask = mask.convert("L").resize((512, 512), resample=Image.BICUBIC)
    mask = np.array(mask)
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask != 1] = 0
    mask[mask == 1] = 1
    mask = torch.from_numpy(mask).to(dtype=torch.half)
    
    return mask

def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor * 255).byte()
    tensor = T.ToPILImage()(tensor)
    return tensor

def recover_image(image, init_image, mask):
    image = prepare_image(image)
    init_image = prepare_image(init_image)
    mask = prepare_mask(mask)
    
    result = image * (mask) + init_image * (1 - mask)
    
    return tensor_to_pil(result)

def overlay_image_mask(image, mask):
    image = prepare_image(image)
    mask = prepare_mask(mask)
    
    image = image * (mask)
    
    return tensor_to_pil(image)
