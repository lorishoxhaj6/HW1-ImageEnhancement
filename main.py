# Computer Vision class 2026/27, HW1-ImageEnhancement

import cv2 
import numpy as np

def gaussian_noise(img, mean, stddev, gamma = 1,seed = 42):
    #Create Gaussian Noise
    gauss_noise = np.zeros(img.shape[:2])
    cv2.randn(gauss_noise, mean, stddev, seed)
    gauss_noise = (gauss_noise*gamma).astype(np.uint8)
    
    if len(img.shape) == 2:
        output = cv2.add(img, gauss_noise)
    elif len(img.shape) == 3:
        merged = cv2.merge([gauss_noise, gauss_noise, gauss_noise])
        output = cv2.add(img, merged)
    return gauss_noise, output

# Loading original images

AmsterdamCity = cv2.imread("images/original/AmsterdamCity.png")
CanalTourBoat = cv2.imread("images/original/CanalTourBoat.png")
Crayfish = cv2.imread("images/original/Crayfish.png")
FoggyBohemiaSunrise = cv2.imread("images/original/FoggyBohemiaSunrise.png")
FoggyMotorway = cv2.imread("images/original/FoggyMotorway.png")

# degradetion process

degreded_AmsterdamCity = 
degreded_CanalTourBoat = 
degreded_Crayfish = 
degreded_FoggyBohemiaSunrise
degreded_FoggyMotorway =
