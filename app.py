"""
AI Image Generation and Style Transfer Application

A Streamlit-based GUI application that allows users to:
1. Generate images from text prompts using Stable Diffusion
2. Apply artistic styles to uploaded images
3. Preview, save, and download results

Author: AI Assistant
Date: 2026
"""

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
from pathlib import Path
import numpy as np

# Import custom modules
from style_transfer import apply_style_transfer


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="AI Image Generator & Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS Styling
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# Initialize Session State
# ============================================================================

if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

if 'stylized_image' not in st.session_state:
    st.session_state.stylized_image = None


# ============================================================================
# Utility Functions
# ============================================================================

def generate_image_demo(prompt):
    """
    Generate a demo image using a lightweight approach.
    For production, integrate with Hugging Face API or replicate.com
    
    Args:
        prompt (str): Text prompt for image generation
    
    Returns:
        PIL.Image: Generated image or placeholder
    """
    try:
        # For Streamlit Cloud: Use Replicate API or Hugging Face Inference API
        import requests
        
        # Placeholder implementation - shows user how to integrate API
        # Replace with actual API calls to Replicate.com or Hugging Face
        
        st.info("💡 **Note**: To generate images, you need to:")
        st.info("1. Get an API key from [Replicate.com](https://replicate.com) or [Hugging Face](https://huggingface.co)")
        st.info("2. Add it to Streamlit secrets")
        st.info("3. Use the API to generate images")
        
        # Create a placeholder image for demo
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        return img
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def download_button(image, filename, label="Download Image"):
    """
    Create a download button for an image.
    
    Args:
        image (PIL.Image): Image to download
        filename (str): Name of the file
        label (str): Button label text
    """
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Create download button
    st.download_button(
        label=label,
        data=img_byte_arr.getvalue(),
        file_name=filename,
        mime="image/png"
    )


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.title("🎨 AI Image Generator & Style Transfer")
    st.markdown("---")
    
    # Create tabs for different features
    tab1, tab2 = st.tabs(["Generate Image", "Style Transfer"])
    
    # ========================================================================
    # TAB 1: Image Generation
    # ========================================================================
    
    with tab1:
        st.header("✨ Generate Images from Text Prompts")
        st.markdown("Describe what you want to create, and the AI will generate an image for you.")
        
        # Create two columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text prompt input
            prompt = st.text_area(
                "Enter your prompt:",
                placeholder="e.g., A serene landscape with mountains, sunset, and a calm lake",
                height=100
            )
        
        with col2:
            st.markdown("### Settings")
            
            # Inference steps slider
            num_steps = st.slider(
                "Quality (inference steps)",
                min_value=20,
                max_value=100,
                value=50,
                step=10,
                help="Higher values produce better quality but take longer"
            )
            
            # Guidance scale slider
            guidance = st.slider(
                "Prompt guidance",
                min_value=1.0,
                max_value=15.0,
                value=7.5,
                step=0.5,
                help="How closely the image follows your prompt"
            )
            
            # Image dimensions
            size_option = st.selectbox(
                "Image size",
                ["512x512", "576x576", "640x640"],
                help="Larger images take longer to generate"
            )
            height, width = map(int, size_option.split('x'))
        
        # Generate button
        if st.button("🚀 Generate Image", use_container_width=True, key="gen_button"):
            if not prompt.strip():
                st.error("Please enter a prompt!")
            else:
                with st.spinner("🎨 Generating your image..."):
                    try:
                        # Generate the image
                        image = generate_image_demo(prompt)
                        
                        # Store in session state
                        st.session_state.generated_image = image
                        st.success("Image generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating image: {e}")
        
        # Display generated image
        st.markdown("---")
        
        if st.session_state.generated_image is not None:
            col_img, col_action = st.columns([3, 1])
            
            with col_img:
                st.subheader("Generated Image")
                st.image(st.session_state.generated_image, use_column_width=True)
            
            with col_action:
                st.subheader("Actions")
                
                # Download button
                download_button(
                    st.session_state.generated_image,
                    f"generated_{st.session_state.generated_image.width}x{st.session_state.generated_image.height}.png",
                    "⬇️ Download Image"
                )
                
                # Save to local folder button
                if st.button("💾 Save Locally", use_container_width=True):
                    output_dir = Path("generated_images")
                    output_dir.mkdir(exist_ok=True)
                    
                    # Generate filename
                    import time
                    timestamp = int(time.time())
                    filename = f"image_{timestamp}.png"
                    filepath = output_dir / filename
                    
                    # Save the image
                    if save_image(st.session_state.generated_image, str(filepath)):
                        st.success(f"✅ Saved to `{filepath}`")
                    else:
                        st.error("Failed to save image")
    
    # ========================================================================
    # TAB 2: Style Transfer
    # ========================================================================
    
    with tab2:
        st.header("🖼️ Apply Artistic Styles to Your Images")
        st.markdown("Upload an image and choose an artistic style to apply.")
        
        # Create columns for upload and settings
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "bmp", "gif"],
                help="Select an image file to apply style transfer"
            )
            
            # Display uploaded image
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file)
                st.image(input_image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("### Settings")
            
            # Style selector
            style_options = [
                'van_gogh',
                'monet',
                'picasso',
                'dali',
                'warhol',
                'sepia'
            ]
            
            selected_style = st.selectbox(
                "Select artistic style:",
                style_options,
                help="Choose a famous artist's style to apply"
            )
            
            # Style strength slider
            style_strength = st.slider(
                "Style intensity",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="How strongly the style is applied (0.1 = subtle, 1.0 = strong)"
            )
            
            st.markdown(f"**Selected Style:** {selected_style.replace('_', ' ').title()}")
        
        # Apply style button
        if uploaded_file is not None:
            if st.button("✨ Apply Style", use_container_width=True, key="style_button"):
                with st.spinner(f"Applying {selected_style} style..."):
                    try:
                        # Apply style transfer
                        stylized = apply_style_transfer(input_image, selected_style, strength=style_strength)
                        
                        # Store in session state
                        st.session_state.stylized_image = stylized
                        st.success("Style applied successfully!")
                        
                    except Exception as e:
                        st.error(f"Error applying style: {e}")
        
        # Display stylized image
        st.markdown("---")
        
        if st.session_state.stylized_image is not None:
            col_before, col_after = st.columns(2)
            
            with col_before:
                st.subheader("Original Image")
                st.image(input_image, use_column_width=True)
            
            with col_after:
                st.subheader("Stylized Image")
                st.image(st.session_state.stylized_image, use_column_width=True)
            
            # Download and save options
            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                download_button(
                    st.session_state.stylized_image,
                    f"stylized_{selected_style}.png",
                    "⬇️ Download Stylized Image"
                )
            
            with col_dl2:
                if st.button("💾 Save Locally", use_container_width=True, key="save_stylized"):
                    output_dir = Path("stylized_images")
                    output_dir.mkdir(exist_ok=True)
                    
                    import time
                    timestamp = int(time.time())
                    filename = f"stylized_{selected_style}_{timestamp}.png"
                    filepath = output_dir / filename
                    
                    if save_image(st.session_state.stylized_image, str(filepath)):
                        st.success(f"✅ Saved to `{filepath}`")
                    else:
                        st.error("Failed to save image")
    
    # ========================================================================
    # Sidebar Information
    # ========================================================================
    
    with st.sidebar:
        st.markdown("## 📋 Information")
        
        st.markdown("""
        ### Features
        - 🎨 **Image Generation**: Create images from text prompts
        - 🖼️ **Style Transfer**: Apply artistic styles to your images
        - ⬇️ **Download**: Save results to your computer
        - 💾 **Local Storage**: Save images to the app folders
        
        ### How to Use
        
        **Image Generation:**
        1. Enter a detailed text prompt
        2. Adjust quality and guidance settings
        3. Click "Generate Image"
        4. Download or save the result
        
        **Style Transfer:**
        1. Upload an image
        2. Select an artistic style
        3. Adjust style intensity
        4. Click "Apply Style"
        5. Download or save the stylized result
        
        ### Tips
        - Be specific in your prompts for better results
        - Use higher inference steps for better quality (slower)
        - Adjust guidance scale to control how much the model follows your prompt
        - Try different styles to see which works best with your images
        
        ### System Info
        """)
        
        # Display GPU availability
        if torch.cuda.is_available():
            st.success(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            st.info("⚠️ Using CPU (GPU would be faster)")
        
        st.markdown("---")
        st.markdown("### System Info")
        
        st.markdown("""
        **Current Setup**: Lightweight version for Streamlit Cloud
        
        **To Enable Full Features:**
        1. Use API-based image generation:
           - [Replicate.com](https://replicate.com)
           - [Hugging Face Inference API](https://huggingface.co/inference-api)
           - [OpenAI DALL-E](https://openai.com/dall-e-3/)
        
        2. Style transfer works locally with uploaded images
        """)


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
