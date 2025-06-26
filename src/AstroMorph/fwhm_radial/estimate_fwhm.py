from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import center_of_mass
import numpy as np
import matplotlib.pyplot as plt

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
        fwhms.append(source['fwhm'])

    return np.array(radii), np.array(fwhms)

def plot_radial_profile(radii, fwhms):
    plt.figure(figsize=(6,4))
    plt.scatter(radii, fwhms, s=10, alpha=0.5)
    plt.xlabel("Radial Distance (pixels)")
    plt.ylabel("FWHM")
    plt.title("Radial FWHM Profile")
    plt.grid(True)
    plt.show()