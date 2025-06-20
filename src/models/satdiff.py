"""
SatDiff: A Stable Diffusion Framework for Inpainting Very High-Resolution Satellite Imagery
Author: Teerapong Panboonyuen
License: MIT
Reference: https://ieeexplore.ieee.org/document/10929005
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDPMScheduler


class SatDiffModel(nn.Module):
    """
    SatDiff: Satellite Inpainting using Stable Diffusion backbone.
    Adapts U-Net latent diffusion to masked satellite images.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Load pretrained autoencoder for VAE encoding/decoding
        self.vae = AutoencoderKL.from_pretrained(
            config["model"]["vae_path"], subfolder="vae"
        )
        self.vae.requires_grad_(False)

        # 2. UNet-based diffusion model (pretrained weights or random init)
        self.unet = UNet2DConditionModel.from_pretrained(
            config["model"]["unet_path"], subfolder="unet"
        )

        # 3. Noise scheduler (DDPM forward diffusion)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config["model"]["scheduler_path"]
        )

        # 4. Loss
        self.loss_fn = nn.MSELoss()

        # 5. Conditioning (e.g., masks or prompts)
        self.condition_with_mask = config["model"].get("use_mask_conditioning", True)

    def forward(self, images, masks):
        """
        Forward pass during training.
        Args:
            images: Tensor of shape (B, C, H, W) - RGB satellite image
            masks: Tensor of shape (B, 1, H, W) - binary inpainting masks
        Returns:
            loss: training loss
        """
        # Step 1: VAE encode
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Step 2: Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Step 3: Concatenate mask as channel or add as condition
        if self.condition_with_mask:
            mask = F.interpolate(masks, size=latents.shape[-2:], mode='nearest')
            noise_input = torch.cat([noisy_latents, mask], dim=1)
        else:
            noise_input = noisy_latents

        # Step 4: UNet predicts noise
        noise_pred = self.unet(
            sample=noise_input,
            timestep=timesteps,
            encoder_hidden_states=None  # no text prompts here
        ).sample

        # Step 5: Compute loss
        loss = self.loss_fn(noise_pred, noise)
        return loss

    def save_pretrained(self, output_dir):
        """
        Save U-Net and config.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.unet.save_pretrained(os.path.join(output_dir, "unet"))
        # VAE and scheduler are assumed fixed and loaded from HF, so not saved.
        config_path = os.path.join(output_dir, "satdiff_config.yaml")
        with open(config_path, "w") as f:
            import yaml
            yaml.dump(self.config, f)