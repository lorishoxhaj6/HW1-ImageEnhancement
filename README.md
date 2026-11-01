# Homework 1: Image Enhancement
Computer Vision, 2026/2027

A classical (no deep learning) pipeline to enhance images affected by low
contrast and non-uniform illumination, compared visually. The original images
are used as references for the synthetic degradation; in real cases a clean
reference is usually not available.

## Image licenses
All images are released under CC0 1.0 or are in the public domain.
They were converted to PNG to standardize the formats. They were then
resized by a factor of 0.12 (this is done by `main.py`).

| ID | Author | Source | License | Original format |
|---|---|---|---|---|
| Crayfish | Biem (talk) | <link> | Public Domain (PD-self) | jpeg |
| CanalTourBoat | Fons Heijnsbroek | <link> | CC0 1.0 | tif |
| AmsterdamCity | Fons Heijnsbroek | <link> | CC0 1.0 | tif |
| FoggyMotorway | Markus Spiske | <link> | CC0 1.0 | jpg |
| FoggyBohemiaSunrise | Ted Moravec | <link> | CC0 1.0 | jpg |

Accessed on 31/10/2026.

## Requirements
- Python (version >= 3.10.12) 
- numpy
- opencv-python

Install with `pip install -r requirements.txt`.

## How to run
```bash
git clone <URL of this repository>
cd <repository folder>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
The five original PNG files must be in `images/original/`.
The script resizes the images, degrades them, applies the methods,
and saves the processed images and the comparison figures.
All output folders are created automatically.

## Degradation parameters
Applied in this order, with the same values for every image:
- Contrast reduction: `cv2.convertScaleAbs`, gain = 0.7, bias = 0
  (out = 0.7 * in; this also makes the image darker)
- Gaussian noise: mean = 0, sigma = 10, added to the three channels
  (same noise on B, G and R)
- Random seed: 42 (`cv2.setRNGSeed`)

## Methods
All methods are applied to the degraded images.
1. **Baseline**: degraded image, no processing.
2. **Linear contrast stretching**: on the L channel of CIE-Lab, using the
   minimum and maximum of L, mapped to the range 0-255. The a and b channels
   are not changed.
3. **Histogram equalization**: on the L channel of CIE-Lab
   (`cv2.equalizeHist`). The a and b channels are not changed.
4. **Denoising + gamma correction + sharpening**:
   - Gaussian blur, kernel 3x3
   - gamma correction (gamma = 0.8) on the L channel of CIE-Lab
   - unsharp masking (blur 3x3, sigma = 1.0, amount = 1.5)

## File structure
```
.
├── main.py
├── requirements.txt
├── README.md
├── images/
│   ├── original/              input images (PNG)
│   ├── resized/               originals resized by 0.12 (references)
│   └── degraded/              degraded images
├── results/
│   ├── baseline/
│   ├── contrast_stretching/
│   ├── histogram_equalization/
│   ├── denoise_gamma_sharpen/
│   └── figures/               one comparison figure per image
└── report.pdf
```
Inside every folder the files have the same name as the image
(for example `Crayfish.png`).
