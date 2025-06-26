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
