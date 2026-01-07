"""
Style Transfer Module

This module provides functions to apply artistic styles to uploaded images
using neural style transfer techniques and pre-trained models.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np


class StyleTransfer:
    """
    A class to perform neural style transfer on images.
    Uses VGG19 as the feature extractor for style and content representation.
    """
    
    def __init__(self):
        """Initialize the style transfer model and preprocessing."""
        # Load pre-trained VGG19 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg19 = models.vgg19(pretrained=True).to(self.device)
        
        # Freeze all VGG parameters
        for param in self.vgg19.parameters():
            param.requires_grad = False
        
        # Define the image normalization
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        
        # Define feature extraction layers
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    def preprocess_image(self, image, size=256):
        """
        Preprocess image for model input.
        
        Args:
            image (PIL.Image or np.ndarray): Input image
            size (int): Size to resize image to
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image).astype('uint8'))
        
        # Define preprocessing transformations
        preprocess = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cnn_normalization_mean, 
                               std=self.cnn_normalization_std)
        ])
        
        return preprocess(image).unsqueeze(0).to(self.device)
    
    def postprocess_image(self, tensor):
        """
        Convert tensor back to PIL Image.
        
        Args:
            tensor (torch.Tensor): Image tensor
        
        Returns:
            PIL.Image: Converted image
        """
        # Remove batch dimension
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        
        # Denormalize
        unorm = transforms.Compose([
            transforms.Normalize(mean=[-m/s for m, s in zip(self.cnn_normalization_mean, self.cnn_normalization_std)],
                               std=[1/s for s in self.cnn_normalization_std])
        ])
        image = unorm(image)
        
        # Convert to PIL
        image = transforms.ToPILImage()(image.clamp(0, 1))
        
        return image
    
    def apply_artistic_style(self, content_image, style_name, strength=1.0):
        """
        Apply predefined artistic styles to an image.
        
        Args:
            content_image (PIL.Image): Input image to stylize
            style_name (str): Name of the style to apply (van_gogh, monet, picasso, etc.)
            strength (float): Strength of style application (0.0 to 1.0)
        
        Returns:
            PIL.Image: Stylized image
        """
        # Define style characteristics (simplified version without actual style images)
        # In a production app, you would load actual style reference images
        
        styles = {
            'van_gogh': {'saturation': 1.3, 'contrast': 1.2, 'brightness': 0.95},
            'monet': {'saturation': 0.8, 'contrast': 0.9, 'brightness': 1.1},
            'picasso': {'saturation': 1.5, 'contrast': 1.4, 'brightness': 0.9},
            'dali': {'saturation': 1.6, 'contrast': 1.3, 'brightness': 1.0},
            'warhol': {'saturation': 2.0, 'contrast': 1.5, 'brightness': 1.0},
            'sepia': {'saturation': 0.3, 'contrast': 0.9, 'brightness': 0.95}
        }
        
        if style_name not in styles:
            return content_image
        
        # Convert to numpy for processing
        content_np = np.array(content_image).astype(np.float32) / 255.0
        
        # Apply style adjustments
        style_params = styles[style_name]
        
        # Convert to HSV for saturation adjustment
        from PIL import ImageEnhance
        
        img_pil = Image.fromarray((content_np * 255).astype(np.uint8))
        
        # Apply saturation
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(style_params['saturation'] * strength + (1 - strength))
        
        # Apply contrast
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(style_params['contrast'] * strength + (1 - strength))
        
        # Apply brightness
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(style_params['brightness'] * strength + (1 - strength))
        
        return img_pil
    
    def get_available_styles(self):
        """
        Get list of available artistic styles.
        
        Returns:
            list: List of style names
        """
        return ['van_gogh', 'monet', 'picasso', 'dali', 'warhol', 'sepia']


def apply_style_transfer(image, style_name):
    """
    Convenience function to apply style transfer to an image.
    
    Args:
        image (PIL.Image): Input image
        style_name (str): Name of the style
    
    Returns:
        PIL.Image: Stylized image
    """
    # Initialize style transfer
    st = StyleTransfer()
    
    # Apply the style
    stylized_image = st.apply_artistic_style(image, style_name, strength=0.8)
    
    return stylized_image
