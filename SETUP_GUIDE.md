# Installation Guide for Requirements-Local

## Windows Setup with PyTorch

### Issue: Visual C++ Redistributable Missing
If you get an error about missing DLL, install the Visual C++ Redistributable:

1. Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Run the installer
3. Restart your terminal

### Alternative: Use CPU-only PyTorch (No CUDA)

```bash
# This is a lightweight CPU version without CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### With CUDA 12.1 (GPU Support)

```bash
# Already installed with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Works after installing Visual C++ Redistributable
```

### Once Dependencies are Fixed

```bash
streamlit run app.py
```

The app will now have FULL IMAGE GENERATION SUPPORT with Stable Diffusion!

## Troubleshooting

### "c10.dll not found"
→ Install Visual C++ Redistributable from link above

### "No module named 'diffusers'"
→ Run: `pip install -r requirements-local.txt`

### PyTorch too large?
→ Use CPU version: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

### Want minimal setup?
→ Use: `pip install -r requirements.txt` (lightweight, no PyTorch)
