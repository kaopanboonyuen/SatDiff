"""
SatDiff: A Stable Diffusion Framework for Inpainting Very High-Resolution Satellite Imagery
Author: Teerapong Panboonyuen
License: MIT
Reference: https://ieeexplore.ieee.org/document/10929005
"""

import os
import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from data.dataset_loader import load_satellite_dataset
from models import get_model
from utils.logger import get_logger
from utils.tiling import tile_image, stitch_tiles

def load_config(config_path: str = "config/default.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, dataloader, optimizer, scheduler, accelerator, epoch, config, logger):
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(accelerator.device)
        masks = batch.get("mask", torch.zeros_like(images)).to(accelerator.device)

        loss = model(images, masks)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if accelerator.is_main_process:
            logger.info(f"Epoch [{epoch}] Iter [{batch_idx}] - Loss: {loss.item():.4f}")

    scheduler.step()
    return loss.item()

def main():
    config = load_config()
    accelerator = Accelerator(mixed_precision=config['training'].get('precision', 'fp16'))

    logger = get_logger("train_log.txt", accelerator)
    logger.info("🚀 Starting training...")

    dataset = load_satellite_dataset(config)
    dataloader = DataLoader(dataset, batch_size=config['dataset']['batch_size'], shuffle=True)

    model = get_model(config).to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(config['training']['epochs']):
        loss = train_one_epoch(model, dataloader, optimizer, scheduler, accelerator, epoch, config, logger)

        if accelerator.is_main_process and (epoch + 1) % config['training']['save_freq'] == 0:
            model.save_pretrained(os.path.join(config['training']['output_dir'], f"epoch_{epoch+1}"))
            logger.info(f"✅ Model saved at epoch {epoch + 1}")

    logger.info("🎯 Training Complete.")


if __name__ == "__main__":
    main()