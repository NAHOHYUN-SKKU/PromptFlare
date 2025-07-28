import os
import torch
from PIL import Image, ImageOps
import argparse
from diffusers import StableDiffusionInpaintPipeline

# from our_method import method
from promptflare import method
    
def main(args):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        variant="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    image_folder_path = args.image_folder_path
    mask_folder_path = args.mask_folder_path
    # output_folder_path = os.path.join(args.output_folder_path, f"eps={args.eps}")
    output_folder_path = args.output_folder_path
    os.makedirs(output_folder_path, exist_ok=True)
    
    args.eps /= 255.0
    args.step_size /= 255.0
    
    for i, image_file in enumerate(os.listdir(image_folder_path)):
        if not image_file.endswith('.png'):
            continue
        image_path = os.path.join(image_folder_path, image_file)
        mask_path = os.path.join(mask_folder_path, image_file)
        output_path = os.path.join(output_folder_path, image_file)
        
        # print(image_path, mask_path, image_file)
        assert os.path.exists(mask_path), f"There must be a mask with the same name as the image."
        if os.path.exists(output_path):
            continue
        
        init_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")
        mask_image = ImageOps.invert(mask_image)
        
        adv = method(pipe, init_image, mask_image, args)
        
        adv.save(output_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", dest="image_folder_path", action="store", default="./sample/original_images")
    parser.add_argument("--mask", dest="mask_folder_path", action="store", default="./sample/masks")
    parser.add_argument("--output", dest="output_folder_path", action="store", default="./sample/protected_images")
    
    parser.add_argument("--eps", dest="eps", action="store", default=12, type=int)
    parser.add_argument("--step", dest="step_size", action="store", default=2, type=float)
    parser.add_argument("--epochs", dest="epochs", action="store", default=400, type=int)
    
    args = parser.parse_args()
    main(args)