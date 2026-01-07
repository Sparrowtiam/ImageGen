"""
Image Generation Module

This module provides utilities for image generation.

For local development: Use Stable Diffusion via Hugging Face (requires PyTorch)
For Streamlit Cloud: Use API-based services

Recommended API services:
- Replicate.com: Great free tier for Stable Diffusion
- Hugging Face Inference API: Easy integration
- OpenAI DALL-E: Premium quality
"""

import os
import requests
from PIL import Image
from io import BytesIO

# Try to import local models (only available with requirements-local.txt)
try:
    import torch
    from diffusers import StableDiffusionPipeline
    HAS_LOCAL_MODELS = True
except (ImportError, OSError):
    HAS_LOCAL_MODELS = False


def initialize_local_generator(model_id="runwayml/stable-diffusion-v1-5"):
    """
    Initialize the Stable Diffusion pipeline for local image generation.
    
    Only works with requirements-local.txt installed (includes PyTorch).
    
    Args:
        model_id (str): The Hugging Face model ID to use
    
    Returns:
        StableDiffusionPipeline: Initialized pipeline or None if not available
    """
    if not HAS_LOCAL_MODELS:
        return None
    
    try:
        # Check if CUDA is available for faster processing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the pre-trained model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        # Move pipeline to appropriate device
        pipe = pipe.to(device)
        
        # Enable memory optimization
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        return pipe
    except Exception as e:
        print(f"Error initializing local generator: {e}")
        return None


def generate_image_local(prompt, pipe, num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
    """
    Generate an image from a text prompt using local Stable Diffusion.
    
    Args:
        prompt (str): Text description of the image to generate
        pipe: The initialized Stable Diffusion pipeline
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): How strongly the model follows the prompt
        height (int): Height of generated image in pixels
        width (int): Width of generated image in pixels
    
    Returns:
        PIL.Image: Generated image
    """
    if pipe is None:
        return None
    
    try:
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None
    """
    Generate image using Replicate.com API (recommended for Streamlit Cloud).
    
    Args:
        prompt (str): Text description for image generation
        api_token (str): Replicate API token (from environment or parameter)
        num_inference_steps (int): Number of inference steps
    
    Returns:
        PIL.Image: Generated image or None
    """
    try:
        import replicate
        
        # Get API token from parameter or environment
        token = api_token or os.getenv("REPLICATE_API_TOKEN")
        
        if not token:
            return None
        
        # Set API token
        replicate.Client(api_token=token)
        
        # Run the model
        output = replicate.run(
            "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041f529f2d4ee4cb280350045e9c63713",
            input={"prompt": prompt, "num_inference_steps": num_inference_steps}
        )
        
        # Download and open the image
        if output:
            response = requests.get(output[0])
            image = Image.open(BytesIO(response.content))
            return image
            
    except Exception as e:
        print(f"Error generating image with Replicate: {e}")
    
    return None


def generate_with_huggingface(prompt, api_token=None):
    """
    Generate image using Hugging Face Inference API.
    
    Args:
        prompt (str): Text description for image generation
        api_token (str): Hugging Face API token
    
    Returns:
        PIL.Image: Generated image or None
    """
    try:
        # Get API token from parameter or environment
        token = api_token or os.getenv("HUGGINGFACE_API_KEY")
        
        if not token:
            return None
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Use Hugging Face Inference API
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
        
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.content
        
        image_bytes = query({"inputs": prompt})
        image = Image.open(BytesIO(image_bytes))
        
        return image
        
    except Exception as e:
        print(f"Error generating image with Hugging Face: {e}")
    
    return None


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
