"""
SatDiff: A Stable Diffusion Framework for Inpainting Very High-Resolution Satellite Imagery
Author: Teerapong Panboonyuen
License: MIT
Reference: https://ieeexplore.ieee.org/document/10929005
"""

import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load validation dataset
val_dataset = load_dataset("imagefolder", data_dir=config['dataset']['val_images'])
val_loader = DataLoader(val_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)

# Load trained model
pipe = StableDiffusionPipeline.from_pretrained(f"{config['training']['output_dir']}/model_epoch_{config['training']['epochs']}")
pipe.to("cuda")

# Evaluation loop
pipe.eval()
for batch in val_loader:
    images = batch['image'].to("cuda")
    masks = batch['mask'].to("cuda")
    inpainted_images = pipe(images, masks)
    # Evaluation metrics would go here (PSNR, SSIM, etc.)
    # For now, just save or log output
    print("Batch processed.")

print("Evaluation complete.")