"""
Style Transfer Module

This module provides functions to apply artistic styles to uploaded images
using image enhancement techniques (no deep learning required).
Works great on Streamlit Cloud!
"""

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


class StyleTransfer:
    """
    A lightweight class to apply artistic styles to images.
    Uses PIL image processing instead of neural networks for cloud compatibility.
    """
    
    def __init__(self):
        """Initialize the style transfer system."""
        pass
    
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
        # Define style characteristics
        styles = {
            'van_gogh': {'saturation': 1.3, 'contrast': 1.2, 'brightness': 0.95, 'blur': 1},
            'monet': {'saturation': 0.8, 'contrast': 0.9, 'brightness': 1.1, 'blur': 2},
            'picasso': {'saturation': 1.5, 'contrast': 1.4, 'brightness': 0.9, 'blur': 0},
            'dali': {'saturation': 1.6, 'contrast': 1.3, 'brightness': 1.0, 'blur': 1},
            'warhol': {'saturation': 2.0, 'contrast': 1.5, 'brightness': 1.0, 'blur': 0},
            'sepia': {'saturation': 0.3, 'contrast': 0.9, 'brightness': 0.95, 'blur': 1}
        }
        
        if style_name not in styles:
            return content_image
        
        # Get style parameters
        style_params = styles[style_name]
        
        # Start with the input image
        img_pil = content_image.copy()
        
        # Apply saturation
        enhancer = ImageEnhance.Color(img_pil)
        blend_factor = style_params['saturation'] * strength + (1 - strength)
        img_pil = enhancer.enhance(blend_factor)
        
        # Apply contrast
        enhancer = ImageEnhance.Contrast(img_pil)
        blend_factor = style_params['contrast'] * strength + (1 - strength)
        img_pil = enhancer.enhance(blend_factor)
        
        # Apply brightness
        enhancer = ImageEnhance.Brightness(img_pil)
        blend_factor = style_params['brightness'] * strength + (1 - strength)
        img_pil = enhancer.enhance(blend_factor)
        
        # Apply blur if specified
        if style_params.get('blur', 0) > 0:
            blur_amount = int(style_params['blur'] * strength)
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_amount))
        
        return img_pil
    
    def get_available_styles(self):
        """
        Get list of available artistic styles.
        
        Returns:
            list: List of style names
        """
        return ['van_gogh', 'monet', 'picasso', 'dali', 'warhol', 'sepia']



def apply_style_transfer(image, style_name, strength=0.8):
    """
    Convenience function to apply style transfer to an image.
    
    Args:
        image (PIL.Image): Input image
        style_name (str): Name of the style
        strength (float): Strength of style application
    
    Returns:
        PIL.Image: Stylized image
    """
    # Initialize style transfer
    st = StyleTransfer()
    
    # Apply the style
    stylized_image = st.apply_artistic_style(image, style_name, strength=strength)
    
    return stylized_image
