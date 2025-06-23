import numpy as np
from skimage import filters, exposure, img_as_float
from scipy.ndimage import gaussian_filter, generic_filter
from skimage.util import img_as_ubyte


def local_entropy(image, neighborhood=9):
    """Compute local entropy in a sliding window."""
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    ubyte_img = img_as_ubyte(image / image.max())
    return entropy(ubyte_img, disk(neighborhood))


def radial_weight_map(shape, center, power=2):
    """Generate a radial fall-off weight map centered on the cluster."""
    yy, xx = np.indices(shape)
    r = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    r_norm = r / r.max()
    return 1 - r_norm ** power


def entropy_sharpen(image, cluster_center, entropy_scale=1.0, contrast_boost=2.0):
    """
    Returns a radial distance sharpened image using entropic local weighting.

    :param image:
    :param cluster_center:
    :param entropy_scale:
    :param contrast_boost:
    :return:
    """
    image = img_as_float(image)
    ent = local_entropy(image)
    weights = radial_weight_map(image.shape, cluster_center)

    # Normalize entropy and mix it into the image
    ent_norm = (ent - ent.min()) / (ent.ptp())
    ent_weighted = ent_norm * weights
    enhanced = image + entropy_scale * (ent_weighted * (image - gaussian_filter(image, sigma=2)))

    # Contrast enhancement
    return exposure.adjust_gamma(enhanced, gamma=1.0 / contrast_boost)
