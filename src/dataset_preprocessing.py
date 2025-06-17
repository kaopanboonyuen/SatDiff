"""
SatDiff: A Stable Diffusion Framework for Inpainting Very High-Resolution Satellite Imagery
Author: Teerapong Panboonyuen
License: MIT
Reference: https://ieeexplore.ieee.org/document/10929005
"""

import os
from PIL import Image

def preprocess_images(image_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            image.save(os.path.join(output_dir, f"image_{filename}"))
            mask.save(os.path.join(output_dir, f"mask_{filename}"))

# Example usage
preprocess_images("data/raw/images", "data/raw/masks", "data/train")