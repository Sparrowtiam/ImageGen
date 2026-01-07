# Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### Step 1: Go to Streamlit Cloud
1. Visit: https://share.streamlit.io/
2. Sign in with your GitHub account (Sparrowtiam)

### Step 2: Deploy New App
1. Click **"New app"** button
2. Fill in the deployment details:
   - **Repository**: `Sparrowtiam/ImageGen`
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. Click **"Deploy"**

### Step 3: Wait for Deployment
The app will:
1. Install dependencies from `requirements.txt` (lightweight version)
2. Build the container
3. Start your app
4. Give you a public URL like: `https://imagegen-sparrowtiam.streamlit.app/`

⏱️ **First deployment**: 5-10 minutes
⏱️ **Subsequent deployments**: 2-3 minutes

### What's Included

✅ **Style Transfer**: Works instantly with lightweight image processing
✅ **Image Upload/Download**: Full functionality
✅ **Dark Theme**: Pre-configured Streamlit settings
✅ **Mobile Responsive**: Works on all devices

### Features Available on Streamlit Cloud

The deployed app uses `requirements.txt` (lightweight dependencies):
- ✨ **Style Transfer**: 6 artistic styles (Van Gogh, Monet, Picasso, Dali, Warhol, Sepia)
- 🖼️ **Image Processing**: Upload, process, download images
- 💾 **Session Storage**: Images persist during your session
- 📱 **Responsive UI**: Works on desktop and mobile

### Optional: Add Image Generation (Requires API)

To enable text-to-image generation on Streamlit Cloud:

1. Get an API key from one of these services:
   - **Replicate.com**: https://replicate.com/signin
   - **Hugging Face Inference API**: https://huggingface.co/inference-api
   - **OpenAI DALL-E**: https://platform.openai.com/api-keys

2. In Streamlit Cloud dashboard:
   - Go to your app settings
   - Click "Secrets"
   - Add your API key:
     ```toml
     REPLICATE_API_TOKEN = "your-token-here"
     ```
     Or for Hugging Face:
     ```toml
     HUGGINGFACE_API_KEY = "your-key-here"
     ```

3. The app will automatically detect your credentials and enable image generation

### Monitoring Your App

After deployment:
1. View app logs: Click "Manage" → "View logs"
2. Check performance: Monitor in real-time
3. Redeploy: Push to GitHub and it auto-deploys
4. Settings: Adjust in "Manage app" section

### Troubleshooting

**App won't load?**
- Check logs for errors
- Ensure `requirements.txt` has no syntax errors
- All files must be in the GitHub repository

**Style transfer not working?**
- Clear browser cache
- Restart the app from "Manage" → "Reboot"

**Want to use local PyTorch?**
- Not recommended for Streamlit Cloud (too large)
- Use local development instead with `requirements-local.txt`

### Local vs Cloud Setup

| Feature | Local (`requirements-local.txt`) | Cloud (`requirements.txt`) |
|---------|----------------------------------|--------------------------|
| Image Generation | ✅ Full Stable Diffusion | ❌ Needs API |
| Style Transfer | ✅ Fast | ✅ Fast |
| Model Size | ~4GB | ~50MB |
| Speed | GPU accelerated | CPU (slower) |
| Cost | Free (local) | Free tier available |

### Repository Structure

Your GitHub repo (`Sparrowtiam/ImageGen`) contains:
- `app.py` - Main application
- `image_generator.py` - Image generation utilities
- `style_transfer.py` - Style transfer functionality
- `requirements.txt` - Cloud dependencies (lightweight)
- `requirements-local.txt` - Local development (with PyTorch)
- `.streamlit/config.toml` - Streamlit configuration
- `README.md` - Full documentation

### Support

**For Streamlit Cloud issues**: https://discuss.streamlit.io/
**For app bugs**: Check GitHub issues or create one

---

**Your deployed app will be available at:**
🌐 https://imagegen-sparrowtiam.streamlit.app/

Share this link to show off your AI image tools! 🎨
