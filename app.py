"""
AI Image Generation and Style Transfer Application
Simple, working version for Streamlit Cloud
"""

import streamlit as st
from PIL import Image, ImageEnhance
import io
from pathlib import Path

# ============================================================================
# Page Configuration (MUST be first)
# ============================================================================

st.set_page_config(
    page_title="AI Image Generator & Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Import custom modules with error handling
# ============================================================================

try:
    from image_generator import initialize_local_generator, generate_image_local, HAS_LOCAL_MODELS
except Exception:
    HAS_LOCAL_MODELS = False

try:
    from style_transfer import apply_style_transfer
except Exception:
    st.error("Error loading style transfer module")
    st.stop()

# Try to import PyTorch
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ============================================================================
# Session State
# ============================================================================

if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

if 'stylized_image' not in st.session_state:
    st.session_state.stylized_image = None

if 'input_image' not in st.session_state:
    st.session_state.input_image = None


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def load_generator():
    """Load image generation pipeline if available"""
    if not HAS_LOCAL_MODELS:
        return None
    try:
        return initialize_local_generator()
    except Exception:
        return None


def save_image_local(image, folder, prefix):
    """Save image to local folder"""
    try:
        output_dir = Path(folder)
        output_dir.mkdir(exist_ok=True)
        
        import time
        timestamp = int(time.time())
        filename = f"{prefix}_{timestamp}.png"
        filepath = output_dir / filename
        
        image.save(str(filepath))
        return str(filepath)
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None


def download_button(image, filename):
    """Create download button for image"""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    st.download_button(
        label="⬇️ Download Image",
        data=img_bytes.getvalue(),
        file_name=filename,
        mime="image/png",
        use_container_width=True
    )


# ============================================================================
# Main App
# ============================================================================

# Title
st.title("🎨 AI Image Generator & Style Transfer")
st.markdown("Create images from text or apply artistic styles to photos")
st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["🎨 Generate Image", "🖼️ Style Transfer"])

# ========================================================================
# TAB 1: Image Generation
# ========================================================================

with tab1:
    st.header("Generate Images from Text")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            placeholder="e.g., A serene landscape with mountains and sunset",
            height=100
        )
    
    with col2:
        st.markdown("### Settings")
        num_steps = st.slider("Quality (steps)", 20, 100, 50, 10)
        guidance = st.slider("Prompt strength", 1.0, 15.0, 7.5, 0.5)
        size = st.selectbox("Size", ["512x512", "576x576", "640x640"])
    
    if st.button("🚀 Generate", use_container_width=True, key="gen_btn"):
        if not prompt.strip():
            st.error("Please enter a prompt!")
        else:
            with st.spinner("Generating image..."):
                try:
                    h, w = map(int, size.split('x'))
                    
                    if HAS_LOCAL_MODELS:
                        pipeline = load_generator()
                        if pipeline:
                            image = generate_image_local(
                                prompt=prompt,
                                pipe=pipeline,
                                num_inference_steps=num_steps,
                                guidance_scale=guidance,
                                height=h,
                                width=w
                            )
                            st.session_state.generated_image = image
                            st.success("✅ Generated!")
                        else:
                            st.info("💡 Install PyTorch locally for image generation")
                            st.session_state.generated_image = Image.new('RGB', (h, w), color=(73, 109, 137))
                    else:
                        st.info("💡 Image generation requires PyTorch. Style transfer works great!")
                        st.session_state.generated_image = Image.new('RGB', (h, w), color=(73, 109, 137))
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    if st.session_state.generated_image:
        col_img, col_btn = st.columns([3, 1])
        
        with col_img:
            st.image(st.session_state.generated_image, use_column_width=True)
        
        with col_btn:
            download_button(
                st.session_state.generated_image,
                f"generated_{st.session_state.generated_image.width}x{st.session_state.generated_image.height}.png"
            )
            
            if st.button("💾 Save", use_container_width=True):
                path = save_image_local(st.session_state.generated_image, "generated_images", "img")
                if path:
                    st.success(f"✅ Saved to `{path}`")

# ========================================================================
# TAB 2: Style Transfer
# ========================================================================

with tab2:
    st.header("Apply Artistic Styles")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "gif"]
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.input_image = image
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.markdown("### Settings")
        
        styles = ['van_gogh', 'monet', 'picasso', 'dali', 'warhol', 'sepia']
        style = st.selectbox("Style:", styles)
        
        strength = st.slider("Intensity", 0.1, 1.0, 0.8, 0.1)
    
    if st.session_state.input_image and st.button("✨ Apply Style", use_container_width=True, key="style_btn"):
        with st.spinner(f"Applying {style}..."):
            try:
                stylized = apply_style_transfer(
                    st.session_state.input_image,
                    style,
                    strength=strength
                )
                st.session_state.stylized_image = stylized
                st.success("✅ Done!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    if st.session_state.stylized_image:
        col_before, col_after = st.columns(2)
        
        with col_before:
            st.subheader("Original")
            st.image(st.session_state.input_image, use_column_width=True)
        
        with col_after:
            st.subheader("Stylized")
            st.image(st.session_state.stylized_image, use_column_width=True)
        
        st.markdown("---")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            download_button(st.session_state.stylized_image, f"stylized_{style}.png")
        
        with col_d2:
            if st.button("💾 Save", use_container_width=True, key="save_style"):
                path = save_image_local(st.session_state.stylized_image, "stylized_images", f"stylized_{style}")
                if path:
                    st.success(f"✅ Saved to `{path}`")

# ========================================================================
# Sidebar
# ========================================================================

with st.sidebar:
    st.markdown("## ℹ️ Info")
    
    st.markdown("""
    ### Features
    - 🎨 **Image Generation**: Create from text
    - 🖼️ **Style Transfer**: 6 artistic styles
    - ⬇️ **Download**: Save to computer
    - 💾 **Local Save**: Store in folders
    
    ### How to Use
    **Generate:**
    1. Enter a text prompt
    2. Adjust settings
    3. Click "Generate"
    
    **Style Transfer:**
    1. Upload an image
    2. Pick a style
    3. Click "Apply Style"
    
    ### Available Styles
    - Van Gogh
    - Monet
    - Picasso
    - Dali
    - Warhol
    - Sepia
    """)
    
    st.markdown("---")
    st.markdown("### Status")
    
    if HAS_LOCAL_MODELS:
        st.success("✅ PyTorch ready - Full generation enabled")
    else:
        st.warning("⚠️ PyTorch not available")
        st.info("For image generation locally:\n```\npip install -r requirements-local.txt\n```")
    
    if HAS_TORCH:
        try:
            if torch.cuda.is_available():
                st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.info("ℹ️ CPU mode")
        except:
            st.info("ℹ️ CPU mode")
    
    st.markdown("---")
    st.markdown("**Repository**: [GitHub](https://github.com/Sparrowtiam/ImageGen)")
