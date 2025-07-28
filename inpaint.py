import os
import argparse
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline

from utils import *

def inpaint(args):
    annotations = pd.read_csv(args.annotation_path)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_name, safety_checker=None, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True
    ).to("cuda")
            
    os.makedirs(args.output_folder_path, exist_ok=True)
    
    for i in range((len(annotations) + args.batch_size - 1) // args.batch_size):
        file_names = annotations["image_file_name"].loc[i * args.batch_size : i * args.batch_size + args.batch_size].to_list()
        
        image_files = [Image.open(os.path.join(args.protected_folder_path, file_name)).convert("RGB") for file_name in file_names]
        mask_files = [Image.open(os.path.join(args.mask_folder_path, file_name)).convert("RGB") for file_name in file_names]
        mask_files = [tensor_to_pil(prepare_mask(ImageOps.invert(mask_file))) for mask_file in mask_files]
        prompts = annotations["prompt"].loc[i * args.batch_size : i * args.batch_size + args.batch_size].to_list()
        
        inpaintings = pipe(prompt=prompts, image=image_files, mask_image=mask_files, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, strength=args.strength).images
        
        for idx in range(len(file_names)):
            original_image = Image.open(os.path.join(args.original_image_folder_path, file_names[idx])).convert("RGB")
            inpainting = recover_image(inpaintings[idx], original_image, mask_files[idx])
            inpainting.save(os.path.join(args.output_folder_path, file_names[idx]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", dest="annotation_path", action="store", default="./sample/annotations.csv")
    
    parser.add_argument("--model", dest="model_name", action="store", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--guidance_scale", dest="guidance_scale", action="store", default=7.5, type=float)
    parser.add_argument("--inference", dest="num_inference_steps", action="store", default=50, type=int)
    parser.add_argument("--strength", dest="strength", action="store", default=1.0, type=float)
    
    parser.add_argument("--original", dest="original_image_folder_path", action="store", default="./sample/original_images")
    parser.add_argument("--mask", dest="mask_folder_path", action="store", default="./sample/masks")
    parser.add_argument("--protected", dest="protected_folder_path", action="store", default="./sample/protected_images")
    parser.add_argument("--output", dest="output_folder_path", action="store", default="./sample/inpainted_images")
    
    parser.add_argument("--batch", dest="batch_size", action="store", default=10, type=int)
    
    args = parser.parse_args()

    inpaint(args)
