from diffusers import StableDiffusionInpaintPipeline
import torch.nn as nn

class StableDiffusionInpaintModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config['model']['pretrained'],
            torch_dtype=torch.float16
        )
        self.pipe.to("cuda")

    def forward(self, images, masks):
        # Masked inpainting
        output = self.pipe(prompt=[""] * images.shape[0],
                           image=images,
                           mask_image=masks,
                           num_inference_steps=self.pipe.scheduler.config.num_train_timesteps,
                           guidance_scale=7.5)
        return output.loss  # or custom loss for training

    def save_pretrained(self, output_dir):
        self.pipe.save_pretrained(output_dir)