"""
SatDiff: A Stable Diffusion Framework for Inpainting Very High-Resolution Satellite Imagery
Author: Teerapong Panboonyuen
License: MIT
Reference: https://ieeexplore.ieee.org/document/10929005
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load trained model
pipe = StableDiffusionPipeline.from_pretrained(f"{config['training']['output_dir']}/model_epoch_{config['training']['epochs']}")
pipe.to("cuda")

# Inpainting function
def inpaint_image(image_path, mask_path, output_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    inpainted = pipe(image, mask)
    inpainted.save(output_path)

# Example call
inpaint_image("data/test/image.png", "data/test/mask.png", "outputs/inpainted_output.png")