from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import center_of_mass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from pathlib import Path

def estimate_radial_FWHM(image, cluster_center, fwhm_guess=3.0):
    # Step 1: Detect stars
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm_guess, threshold=5.*std)
    sources = daofind(image - median)

    # Step 2: Compute radial distances and FWHM
    radii = []
    fwhms = []

    for source in sources:
        y, x = source['ycentroid'], source['xcentroid']
        r = np.hypot(x - cluster_center[1], y - cluster_center[0])
        radii.append(r)
        fwhms.append(source['sharpness'])  # Or 'flux' or another proxy

        # fwhms.append(source['fwhm'])

    return np.array(radii), np.array(fwhms)

def plot_radial_profile(radii, fwhms):
    plt.figure(figsize=(6,4))
    plt.scatter(radii, fwhms, s=10, alpha=0.5)
    plt.xlabel("Radial Distance (pixels)")
    plt.ylabel("FWHM")
    plt.title("Radial FWHM Profile")
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    file_pth = rf"C:\Users\Futurama\Documents\AstroMorph\AstroMorph\data\M92_cropped.JPG"
    image = imread(file_pth)
    if image.ndim == 3:
        # Weighted average based on human perception (Rec. 601)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        image = 0.2989 * r + 0.5870 * g + 0.1140 * b

    est_fwhm = estimate_radial_FWHM(image, [0,0])
    plot_radial_profile(*est_fwhm)