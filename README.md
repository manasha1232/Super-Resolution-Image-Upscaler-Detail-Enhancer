# 🔍 Super-Resolution Image Upscaler & Detail Enhancer

[![Day](https://img.shields.io/badge/Day-25--30-blue?style=for-the-badge&logo=python)](https://github.com/manasha1232/30-Day-Computer-Vision-Challenge)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

A Deep Learning & Computer Vision engine for **Spatial Super-Resolution Upscaling ($2\times$ and $4\times$) and Sub-Pixel Detail Enhancement**. Combines PyTorch Deep Residual Convolutional Neural Networks (EDSR architecture) with Laplacian high-frequency gradient enhancement to reconstruct sharp images from low-resolution degraded inputs.

---

## 🌟 Key Features

- ⚡ **Deep Neural Super-Resolution**: PyTorch Deep Residual Network utilizing PixelShuffle sub-pixel convolutional upsampling.
- 📐 **High-Frequency Laplacian Enhancer**: Recovers sub-pixel edge sharpness ($\mathbf{I}_{\text{sr\_enhanced}} = \mathbf{I}_{\text{sr}} + \lambda \cdot \mathbf{\nabla}^2 \mathbf{I}_{\text{sr}}$).
- 📊 **Quantitative Benchmark Suite**:
  - **PSNR**: Peak Signal-to-Noise Ratio ($\text{dB}$)
  - **SSIM**: Structural Similarity Index
  - **Edge Sharpness**: Variance of Laplacian gradient magnitude $\mathbb{E}[|\nabla^2 I|]$.
- 🖼️ **4-Panel Split Comparison View**:
  1. Low-Resolution Degraded Input ($200\times200$)
  2. Baseline Bicubic Interpolation ($800\times800$)
  3. Deep Neural Super-Resolution Enhanced ($800\times800$)
  4. High-Frequency Detail Gain Heatmap

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/manasha1232/super_resolution_image_upscaler.git
cd super_resolution_image_upscaler

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1️⃣ Generate Synthetic Test Images & Run Pipeline
```bash
python generate_demo_lowres.py
python super_resolution.py
```

### 2️⃣ Run with Custom Low-Res Image
```bash
python super_resolution.py --lr path/to/low_res.jpg --scale 4
```

---

## 📊 Sample Output Telemetry JSON

```json
{
    "project": "Super-Resolution Image Upscaler & Detail Enhancer",
    "day": 25,
    "status": "SUCCESS",
    "magnification_scale": "4x",
    "resolution": {
        "input_low_res": "200x200",
        "output_super_res": "800x800"
    },
    "performance": {
        "execution_time_sec": 0.814
    },
    "quality_metrics": {
        "psnr_db": {
            "bicubic_baseline": 30.12,
            "deep_neural_sr": 34.85,
            "improvement_db": 4.73
        },
        "ssim": {
            "bicubic_baseline": 0.8912,
            "deep_neural_sr": 0.9421,
            "improvement": 0.0509
        },
        "edge_sharpness_variance": {
            "low_res_input": 142.3,
            "bicubic_upscaled": 210.5,
            "deep_neural_sr_enhanced": 485.6,
            "sharpness_gain_percent": 130.69
        }
    }
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
