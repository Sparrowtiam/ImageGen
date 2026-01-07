"""
Image Generation Module

This module provides functions to generate images from text prompts
using Stable Diffusion through the Hugging Face diffusers library.
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os


def initialize_generator(model_id="runwayml/stable-diffusion-v1-5"):
    """
    Initialize the Stable Diffusion pipeline for image generation.
    
    Args:
        model_id (str): The Hugging Face model ID to use for image generation.
                       Default is Stable Diffusion v1.5
    
    Returns:
        StableDiffusionPipeline: Initialized pipeline ready for inference
    """
    # Check if CUDA is available for faster processing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the pre-trained model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Disable for faster processing
    )
    
    # Move pipeline to the appropriate device
    pipe = pipe.to(device)
    
    # Enable memory optimization
    if device == "cuda":
        pipe.enable_attention_slicing()
    
    return pipe


def generate_image(prompt, pipe, num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
    """
    Generate an image from a text prompt using Stable Diffusion.
    
    Args:
        prompt (str): Text description of the image to generate
        pipe (StableDiffusionPipeline): The initialized pipeline
        num_inference_steps (int): Number of denoising steps (higher = better quality but slower)
        guidance_scale (float): How strongly the model follows the prompt (higher = more adherent)
        height (int): Height of generated image in pixels
        width (int): Width of generated image in pixels
    
    Returns:
        PIL.Image: Generated image
    """
    # Generate the image
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
    
    return image


def save_image(image, filepath):
    """
    Save the generated image to disk.
    
    Args:
        image (PIL.Image): Image to save
        filepath (str): Path where to save the image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the image
        image.save(filepath)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False
