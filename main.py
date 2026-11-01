# Computer Vision class 2026/27, HW1-ImageEnhancement

import os
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

def histogram_equalization(images, result):
    for name,img in images.items():
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L,a,b = cv2.split(lab)
        L_eq = cv2.equalizeHist(L)
        lab_eq = cv2.merge([L_eq, a, b])
        result[name] = cv2.cvtColor(lab_eq,cv2.COLOR_LAB2BGR)
    return result

def linearContrastStreching(images, result, out_max,out_min):
    for name,img in images.items():
        img_cvt = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        L,a,b = cv2.split(img_cvt)
        I_min = np.min(L)
        I_max = np.max(L)
        G = ((L-I_min)/(I_max-I_min))*(out_max-out_min)+out_min
        G = np.clip(G,0,255).astype(np.uint8)
        img_streched = cv2.merge([G, a, b])
        result[name] = cv2.cvtColor(img_streched,cv2.COLOR_LAB2BGR)

    return result


def methodOfChoice(images, result, kernel_size, gamma):
    for name, img in images.items():

        # STEP 1: Denoising using Gaussian Blur
        img_smooth = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # STEP 2: Gamma Correction using Lookup Table
        table = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ]).astype("uint8")

        img_smooth_cvt = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2LAB)

        L, a, b = cv2.split(img_smooth_cvt)

        L = cv2.LUT(L, table)

        img_corrected_cvt = cv2.merge([L, a, b])

        img_corrected = cv2.cvtColor(img_corrected_cvt, cv2.COLOR_LAB2BGR)

        # STEP 3: Sharpening using Unsharp Masking
        alpha = 1.5

        img_blurred = cv2.GaussianBlur(img_corrected,(3, 3),1.0)

        img_sharpened = cv2.addWeighted(img_corrected, 1 + alpha,img_blurred, -alpha,0)

        result[name] = img_sharpened

    return result


def visualize_comparison(images,degrade_images,processed_images, window_name):
    reset = window_name
    for name,img in images.items():
        window_name += " " + name
        cv2.imshow(window_name,np.hstack([img, degrade_images[name], processed_images[name]]))
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        window_name = reset


def save_images(folder, images_dict):
    """Save every image of the dictionary as <folder>/<name>.png (same name in every folder)."""
    os.makedirs(folder, exist_ok=True)   # create missing folders
    for name, img in images_dict.items():
        path = os.path.join(folder, name + ".png")
        if not cv2.imwrite(path, img):   # check the result
            raise IOError(f"Could not write {path}")


def save_figures(folder, originals, degraded, methods):
    """For every image save one side-by-side figure:
    original | degraded | method 1 | method 2 | ...
    `methods` is a dict {method label: {image name: processed image}}."""
    os.makedirs(folder, exist_ok=True)
    for name in originals:
        panels = [("Original", originals[name]), ("Degraded", degraded[name])]
        panels += [(label, imgs[name]) for label, imgs in methods.items()]

        labelled = []
        for title, im in panels:
            # white strip with the title above each panel
            header = np.full((30, im.shape[1], 3), 255, dtype=np.uint8)
            cv2.putText(header, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
            labelled.append(np.vstack([header, im]))

        path = os.path.join(folder, name + "_comparison.png")
        if not cv2.imwrite(path, np.hstack(labelled)):
            raise IOError(f"Could not write {path}")


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
    images[name] = cv2.resize(img,None,fx=0.12,fy=0.12)

# save the resized originals: they are the references for the metrics.
# NOTE: saved in a separate folder, NOT in images/original, otherwise a second
# run would resize the already resized images again.
save_images("images/resized", images)

# gaussian parameters
mean = 0
stddev = 10
gamma = 1
seed = 42

# contrast parameters
gain = 0.7
bias = 0

# computing the degraded images without processing. 

degraded_images ={}

for name,img in images.items():
    # reduce contrast
    degraded = cv2.convertScaleAbs(img,alpha=gain,beta=bias)
    # applying gaussian noise
    _, degraded = gaussian_noise(degraded,mean,stddev,gamma,seed)
    degraded_images[name] = degraded


# save the results of the degraded images, they compose the baseline
save_images("images/degraded", degraded_images)
save_images("results/baseline", degraded_images)

# visualization of the comparison between original img versus degraded one
# visualize_comparison(images,degraded_images,"Original vs Degraded ")

# computing Linear Contrast Streching
out_max = 255 # Streching  paramaters
out_min = 0 

lcs_images = {}
linearContrastStreching(degraded_images,lcs_images,out_max,out_min)
save_images("results/contrast_stretching", lcs_images)

# visualizing comparison between original image, degraded and histogram equalized one
visualize_comparison(images,degraded_images,lcs_images,"Original vs Degraded vs Linear Contrast Streched image")

# computing Histogram equalization

histeq_images = {}
histogram_equalization(degraded_images,histeq_images)
save_images("results/histogram_equalization", histeq_images)

# visualizing comparison between original image, degraded and histogram equalized one
visualize_comparison(images,degraded_images,histeq_images,"Original vs Degraded vs Histogram equalized image")

# 4th method: Denoising + gamma correction (+ sharpening)
gamma = 0.8
kernel = 3
methodChoice_images = {}
methodOfChoice(degraded_images,methodChoice_images,kernel,gamma)
save_images("results/denoise_gamma_sharpen", methodChoice_images)


# visualizing comparison between original image, degraded and the one denoised + gamma corrected
visualize_comparison(images,degraded_images,methodChoice_images,"Original vs Degraded vs denoised + gamma corrected image")

# side-by-side figures (original | degraded | all methods), one file per image
save_figures("results/figures", images, degraded_images, {
    "Contrast stretching": lcs_images,
    "Hist. equalization": histeq_images,
    "Denoise+gamma+sharpen": methodChoice_images,
})

cv2.destroyAllWindows()