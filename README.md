# SuperResizePyTorch

🚀 **Advanced AI-powered image super-resolution tool with CUDA acceleration**

SuperResizePyTorch is a high-performance image upscaling application that leverages PyTorch and NVIDIA CUDA for fast, high-quality image enhancement using Real-ESRGAN models.

## ✨ Features

- 🎯 **CUDA Acceleration** - Utilizes NVIDIA GPU for fast processing
- 🧠 **Real-ESRGAN Models** - State-of-the-art neural network models
- 👤 **Face Mode** - Specialized models for portraits and faces  
- 📁 **Batch Processing** - Process multiple images with wildcards
- ⚡ **Multiple Scale Factors** - 2x, 3x, 4x, 8x upscaling
- 🔄 **Auto Model Download** - Automatic model management with caching
- 🛡️ **Robust Error Handling** - Comprehensive error management

## 🔧 Requirements

- Python 3.7+
- NVIDIA GPU with CUDA support
- Updated NVIDIA drivers

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SuperResizePyTorch.git
cd SuperResizePyTorch

# Install dependencies
pip install torch torchvision opencv-python numpy
```

## 🎯 Usage

### Basic Usage
```bash
# Single image with default 2x scaling
python SuperResizePyTorch.py image.jpg

# Custom scale factor
python SuperResizePyTorch.py image.png --x 4

# Face/portrait mode
python SuperResizePyTorch.py portrait.jpg --x 2 --face
```

### Batch Processing
```bash
# Process all JPG files
python SuperResizePyTorch.py "*.jpg" --x 4

# Process specific pattern
python SuperResizePyTorch.py "img*.png" --x 2 --face

# Process folder
python SuperResizePyTorch.py "./photos/*.jpg" --x 8
```

## 📊 Models

**General Mode (Default):**
- RealESRGAN_x2plus.pth - For 2x scaling (22MB)
- RealESRGAN_x4plus.pth - For 4x scaling (64MB)

**Face Mode (--face):**
- RealESRGAN_x2plus_face.pth - For portraits 2x (22MB)  
- RealESRGAN_x4plus_face.pth - For portraits 4x (64MB)

*Models are automatically downloaded and cached in `./models/` folder.*

## 🏎️ Performance

- **NVIDIA GTX 1050 Ti (4GB)**: ~2-5 seconds per 1920x1080 image
- Higher scale factors require more GPU memory
- Batch processing is optimized for multiple images

## 📝 Examples

```bash
# Basic 2x upscaling
python SuperResizePyTorch.py photo.jpg
# Output: photo_PyTorch_x2.jpg

# 4x face mode
python SuperResizePyTorch.py portrait.png --x 4 --face  
# Output: portrait_PyTorch_x4_face.png

# Batch process all images
python SuperResizePyTorch.py "*.jpg" --x 2
```

## 🚨 Troubleshooting

**"CUDA not available"**: Check NVIDIA drivers and PyTorch CUDA installation  
**"Out of memory"**: Reduce image size or use lower scale factor  
**"Module not found"**: Install dependencies with pip  

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Credits

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Neural network models
- [PyTorch](https://pytorch.org) - Deep learning framework
- [OpenCV](https://opencv.org) - Computer vision library

## 📈 Version

SuperResizePyTorch v1.0 - July 2025
