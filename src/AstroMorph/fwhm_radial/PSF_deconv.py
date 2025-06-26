from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from skimage.restoration import richardson_lucy




def build_psf_from_sources(image, sources, size=11):
    psfs = []
    for src in sources:
        y, x = int(src['ycentroid']), int(src['xcentroid'])
        half = size // 2
        stamp = image[y-half:y+half+1, x-half:x+half+1]
        if stamp.shape == (size, size):
            psfs.append(stamp)
    return np.mean(psfs, axis=0)

def deconvolve_image(image, psf, iterations=20):
    return richardson_lucy(image, psf, iterations=iterations)



def adaptive_threshold(image, sigma_level=4.0):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    return median, std, median + sigma_level * std

def extract_psf(image, sources, stamp_size=11):
    psfs = []
    half = stamp_size // 2
    for src in sources:
        y, x = int(src['ycentroid']), int(src['xcentroid'])
        stamp = image[y-half:y+half+1, x-half:x+half+1]
        if stamp.shape == (stamp_size, stamp_size):
            psfs.append(stamp)
    return np.mean(psfs, axis=0) if psfs else None

def run_richardson_lucy(image, psf, iterations=20):
    return richardson_lucy(image, psf, iterations=iterations)
