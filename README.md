# Homework 1: Image Enhancement
Computer Vision, 2026/2027

Classical (no deep learning) pipeline to enhance images affected by low
contrast and non-uniform illumination, compared quantitatively and
qualitatively.

## Image licenses
All images are released under CC0 1.0 or are in the public domain
(see the table below). They were converted to PNG to
standardize the formats; no other changes were applied.

## Requirements
- Python <version, e.g. 3.11>
- numpy, opencv-python, scikit-image, pandas, matplotlib
  (install with `pip install -r requirements.txt`)

## How to run
```linux
git clone URL of this repository
python -m venv venv
source venv/bin/activate
pip install -r requirments.txt
python main.py
```
This runs the full pipeline: degradation, enhancement, metrics and figures.

## Degradation parameters
- Contrast reduction: <formula and factor, e.g. alpha = 0.5>
- Gaussian noise: mean 0, sigma = <value>
- Random seed: <e.g. 42>

## Methods
1. Baseline (degraded image, no processing)
2. Linear contrast stretching (<per channel / luminance>)
3. Histogram equalization (<L in Lab / V in HSV>)
4. fourth method and its parameters

## File structure
```
.
├── main.py
├── requirements.txt
├── README.md
├── images/
│   ├── original/
│   └── degraded/
├── results/
│   ├── <method folders with processed images>
│   ├── figures/
│   └── metrics.csv
└── report.pdf
```
