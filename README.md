# AI Image Generator & Style Transfer Tool

A powerful, user-friendly Streamlit application that combines AI image generation and artistic style transfer in one tool.

## Features

✨ **Image Generation**
- Generate images from text prompts using Stable Diffusion
- Customize quality settings (inference steps)
- Control prompt adherence (guidance scale)
- Multiple image size options
- Download or save generated images

🖼️ **Style Transfer**
- Apply famous artistic styles to your uploaded images
- Available styles: Van Gogh, Monet, Picasso, Dali, Warhol, Sepia
- Adjustable style intensity for fine-tuned results
- Side-by-side comparison of original and stylized images
- Download or save stylized results

💾 **File Management**
- Download images directly from the browser
- Save images to local folders (`generated_images/` and `stylized_images/`)
- Works with common image formats (JPG, PNG, BMP, GIF)

## Project Structure

```
Bot-712/
├── app.py                    # Main Streamlit application
├── image_generator.py        # Image generation utilities (local + API)
├── style_transfer.py         # Style transfer functionality
├── requirements.txt          # Lightweight dependencies (for cloud)
├── requirements-local.txt    # Full dependencies (for local development with PyTorch)
├── .env.example             # Environment configuration template
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── README.md                # This file
├── generated_images/        # Output folder for generated images
└── stylized_images/         # Output folder for stylized images
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- RAM requirements:
  - **Lightweight mode**: 4GB RAM minimum
  - **Full local mode (with PyTorch)**: 8GB+ RAM recommended
- GPU support optional but recommended for local PyTorch setup

### Two Installation Options

#### Option 1: Local Development (Full Features with PyTorch) ⚡

Best for: Full image generation on your computer

```bash
# Create Python 3.11 environment
python3.11 -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install full dependencies
pip install -r requirements-local.txt

# Install PyTorch (CPU version, change cu121 to cpu if no GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Run the app
streamlit run app.py
```

**GPU Support (NVIDIA):**
If you have an NVIDIA GPU and CUDA installed:
```bash
# Already included in the PyTorch command above with cu121 (CUDA 12.1)
# For other CUDA versions:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```

#### Option 2: Lightweight/Cloud Deployment 🌐

Best for: Streamlit Cloud or minimal dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Install lightweight dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**Note:** Style transfer works great locally. For image generation on cloud, you'll need to add API credentials to Streamlit Secrets.

## Usage

### Running the Application

#### Local Setup (with PyTorch):
```bash
streamlit run app.py
```

Full image generation from text prompts is enabled! The app will open at `http://localhost:8501`

#### Cloud Setup (lightweight):
```bash
streamlit run app.py
```

Style transfer works instantly. Image generation requires API credentials in Streamlit Secrets.

### Image Generation Tab

1. **Enter a prompt**: Describe what you want to create
   - Example: "A futuristic city with flying cars at sunset"
   
2. **Adjust settings** (optional):
   - **Quality**: Higher values = better quality but slower (20-100 steps)
   - **Prompt guidance**: How strictly the model follows your prompt (1.0-15.0)
   - **Image size**: 512x512, 576x576, or 640x640 pixels

3. **Click "Generate Image"**: Wait for the model to process

4. **Download or Save**:
   - Click "Download Image" to save to your Downloads folder
   - Click "Save Locally" to store in `generated_images/` folder

### Style Transfer Tab

1. **Upload an image**: Click the upload area and select an image file

2. **Select a style**: Choose from available artistic styles:
   - Van Gogh: Vibrant, expressive brush strokes
   - Monet: Soft, impressionistic water lilies style
   - Picasso: Cubist, fragmented perspective
   - Dali: Surreal, dreamlike distortions
   - Warhol: High contrast, bold colors
   - Sepia: Classic vintage look

3. **Adjust style intensity** (0.1-1.0):
   - 0.1 = subtle effect
   - 1.0 = strong, dramatic effect

4. **Click "Apply Style"**: Process your image

5. **View and save**:
   - See side-by-side comparison
   - Download or save to `stylized_images/` folder

## System Requirements

### Minimum Requirements
- **RAM**: 8GB
- **Storage**: 10GB (for models)
- **Processor**: Multi-core CPU
- **Python**: 3.8+

### Recommended Requirements
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (2GB+ VRAM)
- **Storage**: 20GB+ for model caching
- **Internet**: For downloading pre-trained models

## GPU Setup (Optional but Recommended)

### NVIDIA GPU with CUDA

1. Install NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Install cuDNN: https://developer.nvidia.com/cudnn
3. Verify installation:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should return `True`

The app will automatically detect and use your GPU if available.

## Model Information

### Stable Diffusion v1.5
- **Size**: ~4.2 GB
- **Source**: Runway ML (via Hugging Face)
- **Model ID**: `runwayml/stable-diffusion-v1-5`
- **License**: Stable Diffusion License Agreement

### VGG19 (for style transfer)
- **Size**: ~574 MB
- **Source**: PyTorch pretrained models
- **Usage**: Feature extraction for style analysis

## Troubleshooting

### Issue: "CUDA out of memory" error
**Solution**: 
- Reduce image size (use 512x512)
- Reduce inference steps
- Use CPU instead (slower but works with less VRAM)
- Close other GPU-intensive applications

### Issue: Model downloads slowly
**Solution**:
- Check internet connection
- Models are cached after first download
- First run will take longer

### Issue: Port 8501 already in use
**Solution**:
```bash
streamlit run app.py --server.port 8502
```

### Issue: Low quality generated images
**Solution**:
- Increase inference steps (50-100)
- Improve your prompt (be more specific)
- Increase guidance scale (7.5-12.0)
- Try again (generation has randomness)

### Issue: Application crashes on startup
**Solution**:
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Clear Streamlit cache: `streamlit cache clear`

## Code Overview

### app.py
Main Streamlit application with:
- Two-tab interface for image generation and style transfer
- Session state management for caching images
- Download functionality
- User-friendly UI with sidebar information

### image_generator.py
Image generation utilities:
- `initialize_generator()`: Loads the Stable Diffusion pipeline
- `generate_image()`: Creates images from text prompts
- `save_image()`: Saves images to disk

### style_transfer.py
Style transfer functionality:
- `StyleTransfer` class: Handles neural style transfer
- `apply_artistic_style()`: Applies predefined artistic styles
- Multiple preset styles with customizable parameters

## Performance Tips

1. **Faster Generation**:
   - Reduce inference steps (30-50 for speed vs 100 for quality)
   - Use smaller image sizes (512x512)
   - Use GPU if available

2. **Better Quality**:
   - Increase inference steps (100+)
   - Be specific and detailed in prompts
   - Adjust guidance scale based on style

3. **Memory Efficiency**:
   - Process one image at a time
   - Close other applications
   - Clear browser cache periodically

## Advanced Configuration

### Changing the Base Model
Edit `app.py` and modify the model ID:
```python
pipeline = initialize_generator(model_id="your-model-id")
```

Available models:
- `runwayml/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-2`
- `stabilityai/stable-diffusion-2-1`

### Adding Custom Styles
Edit `style_transfer.py` and add to the `styles` dictionary:
```python
'custom_style': {
    'saturation': 1.2,
    'contrast': 1.1,
    'brightness': 0.95
}
```

## Future Enhancements

Potential features for expansion:
- DALL-E 3 integration
- More artistic styles
- Batch processing
- Image inpainting
- Real-time generation progress
- Model fine-tuning interface
- Collaborative features
- API endpoints
- Mobile app version

## License

This project uses:
- Stable Diffusion (Stable Diffusion License Agreement)
- PyTorch (BSD License)
- Streamlit (Apache 2.0)
- Hugging Face transformers (Apache 2.0)

## Support & Contact

For issues, questions, or suggestions:
1. Check the Troubleshooting section above
2. Review error messages in the terminal
3. Check Streamlit documentation: https://docs.streamlit.io/

## Credits

Built with:
- **Streamlit**: https://streamlit.io/
- **Stable Diffusion**: https://huggingface.co/runwayml/stable-diffusion-v1-5
- **PyTorch**: https://pytorch.org/
- **Hugging Face**: https://huggingface.co/

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Created**: AI Assistant
