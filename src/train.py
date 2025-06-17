"""
SatDiff: A Stable Diffusion Framework for Inpainting Very High-Resolution Satellite Imagery
Author: Teerapong Panboonyuen
License: MIT
Reference: https://ieeexplore.ieee.org/document/10929005
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import yaml
import os

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Prepare dataset
train_dataset = load_dataset("imagefolder", data_dir=config['dataset']['train_images'])
train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True)

# Load model
pipe = StableDiffusionPipeline.from_pretrained(config['model']['pretrained_model_name_or_path'])
pipe.to("cuda")

# Training loop
for epoch in range(config['training']['epochs']):
    pipe.train()
    for batch in train_loader:
        images = batch['image'].to("cuda")
        masks = batch['mask'].to("cuda")
        loss = pipe(images, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % config['training']['save_model_every'] == 0:
        pipe.save_pretrained(f"{config['training']['output_dir']}/model_epoch_{epoch}")

    print(f"Epoch {epoch}/{config['training']['epochs']} - Loss: {loss.item()}")
print("Training complete!")