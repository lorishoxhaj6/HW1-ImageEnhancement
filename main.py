# Computer Vision class 2026/27, HW1-ImageEnhancement

import cv2 
import numpy as np

def gaussian_noise(img, mean, stddev, gamma = 1,seed = 42):
    #Set the seed for reproducibility reasons
    cv2.setRNGSeed(seed)
    #Create Gaussian Noise
    gauss_noise = np.zeros(img.shape[:2],dtype=np.float32)
    cv2.randn(gauss_noise, mean, stddev)
    gauss_noise *= gamma

    img = img.astype(np.float32)
    if len(img.shape) == 2:
        output = cv2.add(img, gauss_noise)
    elif len(img.shape) == 3:
        merged = cv2.merge([gauss_noise, gauss_noise, gauss_noise])
        output = cv2.add(img, merged)

    output = np.clip(output, 0, 255).astype(np.uint8)
    return gauss_noise, output




# Loading original images

AmsterdamCity = cv2.imread("images/original/AmsterdamCity.png")
CanalTourBoat = cv2.imread("images/original/CanalTourBoat.png")
Crayfish = cv2.imread("images/original/Crayfish.png")
FoggyBohemiaSunrise = cv2.imread("images/original/FoggyBohemiaSunrise.png")
FoggyMotorway = cv2.imread("images/original/FoggyMotorway.png")



images = {
    "AmsterdamCity": AmsterdamCity,
    "CanalTourBoat": CanalTourBoat,
    "Crayfish": Crayfish,
    "FoggyBohemiaSunrise": FoggyBohemiaSunrise,
    "FoggyMotorway": FoggyMotorway
}

# check for correctness of the loading process
for name, img in images.items():
    if img is None:
        raise FileNotFoundError(f"Could not load image: {name}")

# resizing images
for name, img in images.items():
    images[name] = cv2.resize(img,None,fx=0.15,fy=0.15)

# gaussian parameters
mean = 0
stddev = 20
gamma = 1
seed = 42

# contrast parameters
gain = 0.7
bias = 0

# computing the degraded images without processing. 

degraded_images ={}

for name,img in images.items():
    # applying gaussian noise
    _, degraded = gaussian_noise(img,mean,stddev,gamma,seed)
    # reduce contrast
    degraded = cv2.convertScaleAbs(degraded, alpha=gain, beta=bias)
    degraded_images[name] = degraded


# save the results of the degraded images, they compose the baseline
for name,degraded_img in degraded_images.items():
    cv2.imwrite("results/baselines/degraded_"+name+".png",degraded_img)
    cv2.imwrite("images/degraded/degraded_"+name+".png",degraded_img)

# visualization of the comparison between original img versus degraded one
""" for name,img in images.items():
    window_name = "Original vs degraded "+name
    cv2.imshow(window_name,np.hstack([img, degraded_images[name]]))
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
 """

# computing Linear Contrast Streching

# computing Histogram equalization
for name,img in images.items():
    cv2




cv2.destroyAllWindows()
