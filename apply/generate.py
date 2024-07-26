from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionImg2ImgPipeline,
)
import torch
from PIL import Image
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model", 
        type=str, 
        default="./pretrained-models/stable-diffusion-v1-4", 
        help="The pretrained model."
    )
    parser.add_argument(
        "--model_used", 
        type=str, 
        default=None,
        required=True,
        help="The model used to generate images according to specific prompt."
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default='', 
        help="The prompt to guide the generation."
    )
    parser.add_argument(
        "--img_num", 
        type=int, 
        default=10, 
        help="How many images to generate."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default='cuda:0', 
        help="Device used."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='generated_img', 
        help="Device used."
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=100, 
        help="How many steps taken when model generate each image."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(args.model_used, 'unet')
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model, unet=unet, safety_checker=None
    ).to(args.device)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for i in range(args.img_num):
        image = pipe(prompt=args.prompt, 
                     num_inference_steps=args.num_inference_steps).images[0]
        image.save(os.path.join(args.output_dir, f"{i}.png"))


if __name__ == "__main__":
    main()
