from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from skimage.restoration import richardson_lucy
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.pyplot import imread

def convert_to_grayscale(image):
    if image.ndim == 3:
        return 0.2989 * image[...,0] + 0.5870 * image[...,1] + 0.1140 * image[...,2]
    return image

def adaptive_threshold(image, sigma_level=4.0):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    return median, std, median + sigma_level * std

def detect_sources(image, fwhm_guess=3.0, threshold_sigma=4.0):
    median, std, threshold = adaptive_threshold(image, sigma_level=threshold_sigma)
    daofind = DAOStarFinder(fwhm=fwhm_guess, threshold=threshold)
    return daofind(image - median)

def extract_psf(image, sources, stamp_size=11):
    psfs = []
    half = stamp_size // 2
    for src in sources:
        y, x = int(src['ycentroid']), int(src['xcentroid'])
        stamp = image[y-half:y+half+1, x-half:x+half+1]
        if stamp.shape == (stamp_size, stamp_size):
            psfs.append(stamp)
    return np.mean(psfs, axis=0) if psfs else None

def run_deconvolution(image, psf, iterations=20):
    # return richardson_lucy(image, psf, iterations=iterations)
    return richardson_lucy(image, psf, num_iter=iterations)

def compute_radial_profile(image, sources, cluster_center):
    radii, sharpness = [], []
    for s in sources:
        y, x = s['ycentroid'], s['xcentroid']
        r = np.hypot(x - cluster_center[1], y - cluster_center[0])
        radii.append(r)
        sharpness.append(s['sharpness'])
    return np.array(radii), np.array(sharpness)

def plot_radial_profile(radii, values, label="Sharpness"):
    plt.figure(figsize=(6,4))
    plt.scatter(radii, values, s=10, alpha=0.5)
    plt.xlabel("Radial Distance (pixels)")
    plt.ylabel(label)
    plt.title(f"Radial {label} Profile")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = Path(r"C:\Users\Futurama\Documents\AstroMorph\AstroMorph\data\M92_cropped.JPG")
    image = convert_to_grayscale(imread(path))

    sources = detect_sources(image)
    psf = extract_psf(image, sources)
    if psf is not None:
        deconv = run_deconvolution(image, psf)
        radii, sharpness = compute_radial_profile(deconv, sources, cluster_center=[image.shape[0]/2, image.shape[1]/2])
        plot_radial_profile(radii, sharpness, label="Deconvolved Sharpness")
    else:
        print("Warning: PSF extraction failed. Deconvolution skipped.")
