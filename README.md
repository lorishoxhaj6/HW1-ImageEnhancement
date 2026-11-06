# Homework 1: Image Enhancement
Computer Vision, 2026/2027

Classical (no deep learning) pipeline to enhance degraded images: synthetic
degradation, four methods (baseline, linear contrast stretching, histogram
equalization, Non-Local Means denoising), figures and metrics (PSNR, SSIM, MSE).

Images are CC0 1.0 or public domain; sources and licenses are listed in the report.

## Requirements
- Python >= 3.10.12
- numpy, opencv-python, scikit-image

## How to run

**Linux / macOS**
```bash
git clone <URL of this repository>
cd <repository folder>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Windows (PowerShell)**
```powershell
git clone <URL of this repository>
cd <repository folder>
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The input PNG images must be in `images/original/`. All other folders are
created automatically.

## Parameters
- Resize factor: 0.12
- Degradation: contrast gain 0.7 (around the mean), Gaussian noise mean 0,
  sigma 10 (same on the three channels), seed 42
- Contrast stretching: L channel of CIE-Lab, 2nd-98th percentiles
- Histogram equalization: L channel of CIE-Lab
- NLM: h = 3, hColor = 0, patch 7x7, search window 21x21

## File structure
```
.
├── main.py
├── requirements.txt
├── README.md
├── report.pdf
├── images/
│   ├── original/              input images (PNG)
│   ├── resized/               resized originals (references for the metrics)
│   └── degraded/              degraded images
└── results/
    ├── baseline/
    ├── contrast_stretching/
    ├── histogram_equalization/
    ├── nlm_denoising/
    ├── figures/               <image>_<method>.png (original | degraded | result)
    └── metrics.csv            PSNR, SSIM, MSE for every image and method
```