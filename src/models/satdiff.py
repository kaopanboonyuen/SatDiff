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
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionInpaintPipeline,
    LatentDiffusion,
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
)
from diffusers.utils import logging

logger = logging.get_logger(__name__)

class SatDiffModel(nn.Module):
    """
    SatDiff: Satellite Inpainting framework with extended SOTA diffusion backbones.
    
    Supports multiple cutting-edge diffusion models for satellite image inpainting,
    including Stable Diffusion variants, Latent Diffusion, and others for maximal
    research flexibility and baseline comparison.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config["model"]

        # 1. Load fixed pretrained VAE autoencoder
        self.vae = AutoencoderKL.from_pretrained(model_cfg["vae_path"], subfolder="vae")
        self.vae.requires_grad_(False)

        # 2. Flexible backbone loader for multiple SOTA diffusion models
        backbone_type = model_cfg.get("backbone", "unet").lower()

        if backbone_type == "unet":
            self.unet = UNet2DConditionModel.from_pretrained(model_cfg["unet_path"], subfolder="unet")

        elif backbone_type == "stable_diffusion_inpaint":
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_cfg["pipeline_path"])
            self.unet = pipeline.unet

        elif backbone_type == "latent_diffusion":
            # Hypothetical Latent Diffusion backbone loading
            self.unet = LatentDiffusion.from_pretrained(model_cfg["latent_diffusion_path"])

        elif backbone_type == "ldm_text2im":
            # Text-to-image latent diffusion model, can adapt for mask conditioning
            self.unet = LatentDiffusion.from_pretrained(model_cfg["ldm_text2im_path"])

        elif backbone_type == "latent_diffusion_bert":
            # Hypothetical alternative latent diffusion with BERT conditioning
            self.unet = LatentDiffusion.from_pretrained(model_cfg["latent_diffusion_bert_path"])

        elif backbone_type == "improved_diffusion":
            # Improved diffusion model from OpenAI (placeholder example)
            self.unet = UNet2DConditionModel.from_pretrained(model_cfg["improved_diffusion_path"])

        else:
            logger.error(f"Unsupported backbone type: {backbone_type}")
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        # 3. Scheduler selection with support for popular samplers
        scheduler_name = model_cfg.get("scheduler", "ddpm").lower()
        if scheduler_name == "ddpm":
            self.noise_scheduler = DDPMScheduler.from_pretrained(model_cfg["scheduler_path"])
        elif scheduler_name == "pndm":
            self.noise_scheduler = PNDMScheduler.from_pretrained(model_cfg["scheduler_path"])
        elif scheduler_name == "lms":
            self.noise_scheduler = LMSDiscreteScheduler.from_pretrained(model_cfg["scheduler_path"])
        elif scheduler_name == "ddim":
            self.noise_scheduler = DDIMScheduler.from_pretrained(model_cfg["scheduler_path"])
        else:
            logger.warning(f"Unknown scheduler '{scheduler_name}', defaulting to DDPM.")
            self.noise_scheduler = DDPMScheduler.from_pretrained(model_cfg["scheduler_path"])

        # 4. Loss function
        self.loss_fn = nn.MSELoss()

        # 5. Mask conditioning flag
        self.condition_with_mask = model_cfg.get("use_mask_conditioning", True)

    def forward(self, images, masks):
        # Encode to latent space
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise and timestep for diffusion forward process
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device
        ).long()

        # Add noise according to scheduler
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Prepare U-Net input, concatenating mask if enabled
        if self.condition_with_mask:
            mask_resized = F.interpolate(masks, size=latents.shape[-2:], mode='nearest')
            unet_input = torch.cat([noisy_latents, mask_resized], dim=1)
        else:
            unet_input = noisy_latents

        # Predict noise with U-Net conditioned on timestep
        noise_pred = self.unet(
            sample=unet_input,
            timestep=timesteps,
            encoder_hidden_states=None,
        ).sample

        # Compute MSE loss against true noise
        loss = self.loss_fn(noise_pred, noise)
        return loss

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.unet.save_pretrained(os.path.join(output_dir, "unet"))
        config_path = os.path.join(output_dir, "satdiff_config.yaml")
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(self.config, f)