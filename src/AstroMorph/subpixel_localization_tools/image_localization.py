import numpy as np
import tifffile as tiff
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter

# --- 1. Load Image Stack ---
def load_stack(path):
    path = Path(path)
    stack = tiff.imread(path)
    return stack.astype(np.float32)

# --- 2. Twinkle Event Detector ---
def detect_twinkles(stack, threshold=5):
    events = []
    for y in range(stack.shape[1]):
        for x in range(stack.shape[2]):
            signal = stack[:, y, x]
            peaks, _ = find_peaks(signal, height=threshold)
            for t in peaks:
                events.append((t, y, x, signal[t]))
    return np.array(events)

# --- 3. 2D Gaussian Fitting for Localization ---
def gaussian_2d(xy, amp, x0, y0, sigma, offset):
    x, y = xy
    g = amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + offset
    return g.ravel()



def localize_event(stack, t, y, x, window=5):
    sub = stack[t, y-window:y+window+1, x-window:x+window+1]
    if sub.shape != (2*window+1, 2*window+1):
        return None  # skip edge cases
    Y, X = np.indices(sub.shape)
    p0 = (sub.max(), window, window, 1.5, np.median(sub))
    try:
        popt, _ = curve_fit(gaussian_2d, (X.ravel(), Y.ravel()), sub.ravel(), p0=p0, maxfev=1000)
        return (t, y + popt[2] - window, x + popt[1] - window, popt[0])
    except RuntimeError:
        return None

# --- 4. Run Localization on All Events ---
def run_localization(stack, events):
    localizations = []
    for t, y, x, _ in events:
        loc = localize_event(stack, int(t), int(y), int(x))
        if loc:
            localizations.append(loc)
    return np.array(localizations)

def save_localizations(localizations, output_path="twinkle_results.csv"):
    df = pd.DataFrame(localizations, columns=["frame", "y", "x", "intensity"])
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} events to {output_path}")


def render_localization_image(localizations, shape, sigma=1.0):
    img = np.zeros(shape, dtype=np.float32)
    for _, y, x, intensity in localizations:
        if 0 <= int(y) < shape[0] and 0 <= int(x) < shape[1]:
            img[int(y), int(x)] += intensity
    return gaussian_filter(img, sigma=sigma)


if __name__=="__main__":
    file_pth = rf"path\to\synthetic_twinkle.tif"
    img_stack = load_stack(file_pth)
    twinkles = detect_twinkles(img_stack)
    events = twinkles
    localized_events = run_localization(img_stack, events)
    save_localizations(localized_events)

    rendered = render_localization_image(localized_events, img_stack.shape[1:])

    plt.imshow(rendered, cmap="hot", origin="lower")
    plt.title("Localization Intensity Map")
    plt.colorbar(label="Smoothed Intensity")
    plt.imsave("twinkle_localizations.png", rendered, cmap="hot")
    plt.show()
