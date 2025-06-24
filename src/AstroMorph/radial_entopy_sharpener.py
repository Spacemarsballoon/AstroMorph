import numpy as np
from skimage import filters, exposure
from skimage.util import img_as_float
from scipy.ndimage import gaussian_filter, generic_filter
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt


def local_entropy(image, neighborhood=9):
    """Compute local entropy in a sliding window."""
    ubyte_img = img_as_ubyte(image / image.max())
    return entropy(ubyte_img, disk(neighborhood))


def radial_weight_map(shape, center, power=2):
    """Generate a radial fall-off weight map centered on the cluster."""
    yy, xx = np.indices(shape)
    r = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    r_norm = r / r.max()
    return 1 - r_norm ** power


def entropy_sharpen(image, cluster_center, entropy_scale=1.0, contrast_boost=2.0, power=2.0):
    if image.ndim == 3:
        from skimage.color import rgb2gray
        image = rgb2gray(image)

    image = img_as_float(image)
    ent = local_entropy(image)
    weights = radial_weight_map(image.shape, cluster_center, power=power)  # 👈 updated here

    # ent_norm = (ent - ent.min()) / (ent.ptp())
    ent_norm = (ent - ent.min()) / np.ptp(ent)
    ent_weighted = ent_norm * weights
    enhanced = image + entropy_scale * (ent_weighted * (image - gaussian_filter(image, sigma=2)))
    # enhanced = rescale_intensity(enhanced, in_range='image', out_range=(0, 1))

    p_low, p_high = np.percentile(enhanced, (0.5, 99.5))
    enhanced = rescale_intensity(enhanced, in_range=(p_low, p_high), out_range=(0, 1))

    print("Before rescale:", enhanced.min(), enhanced.max())
    print("Rescaling range:", p_low, "to", p_high)

    return exposure.adjust_gamma(enhanced, gamma=1.0 / contrast_boost)

