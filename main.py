# Computer Vision class 2026/27, HW1-ImageEnhancement
import os
import cv2 
import csv
import numpy as np
from skimage.metrics import structural_similarity 


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


def linearContrastStreching(images, result, out_max, out_min):
    for name, img in images.items():
        # Convert BGR to LAB
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(img_lab)

        # Calculate robust intensity limits using percentiles
        p_low = np.percentile(L, 2)
        p_high = np.percentile(L, 98)

        # Avoid division by zero
        if p_high <= p_low:
            result[name] = img.copy()
            continue

        # Apply percentile-based linear contrast stretching
        L_float = L.astype(np.float32)

        G = ((L_float - p_low) /
             (p_high - p_low)) * (out_max - out_min) + out_min

        # Clip values outside the selected percentile range
        G = np.clip(G, out_min, out_max).astype(np.uint8)

        # Reconstruct LAB image
        img_stretched = cv2.merge([G, a, b])

        # Convert back to BGR
        result[name] = cv2.cvtColor(
            img_stretched, cv2.COLOR_LAB2BGR
        )

    return result



def methodOfChoice(images, result, h, h_color, template_size, search_size):
    """4th method: Non-Local Means denoising.
    Every pixel is replaced by the weighted average of pixels with similar patches,
    so edges are preserved better than with a Gaussian blur."""
    for name, img in images.items():
        result[name] = cv2.fastNlMeansDenoisingColored(
            img, None, h, h_color, template_size, search_size)

    return result



def save_images(folder, images_dict):
    """Save every image of the dictionary as <folder>/<name>.png (same name in every folder)."""
    os.makedirs(folder, exist_ok=True)   # create missing folders
    for name, img in images_dict.items():
        path = os.path.join(folder, name + ".png")
        if not cv2.imwrite(path, img):   # check the result
            raise IOError(f"Could not write {path}")


def save_figures(folder, originals, degraded, methods):
    """For every image AND every method save one side-by-side figure:
    original | degraded | processed by that method.
    `methods` is a dict {method name (used in the file name): (title, {image name: processed image})}.
    Files are saved as <folder>/<image name>_<method name>.png"""
    os.makedirs(folder, exist_ok=True)
    for name in originals:
        for method_name, (title, imgs) in methods.items():
            panels = [("Original", originals[name]),
                      ("Degraded", degraded[name]),
                      (title, imgs[name])]

            labelled = []
            for panel_title, im in panels:
                # white strip with the title above each panel
                header = np.full((30, im.shape[1], 3), 255, dtype=np.uint8)
                cv2.putText(header, panel_title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)
                labelled.append(np.vstack([header, im]))

            path = os.path.join(folder, name + "_" + method_name + ".png")
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
save_images("images/resized", images)

# gaussian parameters
mean_gaus = 0
stddev = 10
gamma = 1
seed = 42

# contrast parameters
gain = 0.7

# computing the degraded images without processing. 

degraded_images ={}

for name,img in images.items():
    # reduce contrast
    mean = float(np.mean(img))
    # (1 - gain) * mean formula to reduce only contrast and not also the darkness of the image
    degraded = cv2.convertScaleAbs(img,None,alpha=gain,beta=(1 - gain) * mean)
    # applying gaussian noise
    _, degraded = gaussian_noise(degraded,mean_gaus,stddev,gamma,seed)
    degraded_images[name] = degraded


# save the results of the degraded images, they compose the baseline
save_images("images/degraded", degraded_images)
save_images("results/baseline", degraded_images)


# computing Linear Contrast Streching
out_max = 255 # Streching  paramaters
out_min = 0 

lcs_images = {}
linearContrastStreching(degraded_images,lcs_images,out_max,out_min)
save_images("results/contrast_stretching", lcs_images)


# computing Histogram equalization

histeq_images = {}
histogram_equalization(degraded_images,histeq_images)
save_images("results/histogram_equalization", histeq_images)

# 4th method: Non-Local Means denoising
nlm_h = 3            # denoising strength on luminance 
nlm_h_color = 0      # 0 = colors are not filtered (they keep their original values)
nlm_template = 7     # patch size (odd)
nlm_search = 21      # search window size (odd)
methodChoice_images = {}
methodOfChoice(degraded_images,methodChoice_images,nlm_h,nlm_h_color,nlm_template,nlm_search)
save_images("results/nlm_denoising", methodChoice_images)



# side-by-side figures: one file per image and per method (original | degraded | method)
# saved as results/figures/<image>_<method>.png
save_figures("results/figures", images, degraded_images, {
    "contrast_stretching": ("Contrast stretching", lcs_images),
    "histogram_equalization": ("Histogram equalization", histeq_images),
    "nlm_denoising": ("NLM denoising", methodChoice_images),
})

# computing metrics

methods = {
    "baseline": degraded_images,
    "contrast_stretching": lcs_images,
    "histogram_equalization": histeq_images,
    "nlm_denoising": methodChoice_images,
}

# Computing and writing in the file metrics.csv the metrics
with open("results/metrics.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "method", "PSNR_dB", "SSIM", "MSE"])
    for name, original in images.items():
        for method, results in methods.items():
            psnr = cv2.PSNR(original, results[name])
            ssim = structural_similarity(original, results[name],
                                         channel_axis=2, data_range=255)
            # MSE: convert to float first, subtracting two uint8 arrays would wrap around
            mse = np.mean((original.astype(np.float32) - results[name].astype(np.float32)) ** 2)
            writer.writerow([name, method, round(psnr, 2), round(ssim, 4), round(float(mse), 2)])        


cv2.destroyAllWindows()