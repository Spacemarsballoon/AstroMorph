import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff


def generate_star_field(size=(256, 256), n_stars=50):
    """Random subpixel stars"""
    stars = []
    for _ in range(n_stars):
        x = np.random.uniform(0, size[1])
        y = np.random.uniform(0, size[0])
        intensity = np.random.uniform(1000, 5000)
        stars.append((x, y, intensity))
    return stars


def add_psf(image, x, y, intensity, sigma=1.5):
    """Draw Gaussian PSF at subpixel location"""
    X, Y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    psf = intensity * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
    image += psf


def simulate_stack(frames=50, image_size=(256, 256), drift_per_frame=(0.01, 0.005), twinkle_std=0.1):
    stars = generate_star_field(size=image_size)
    stack = np.zeros((frames, *image_size), dtype=np.float32)

    for t in range(frames):
        frame = np.zeros(image_size, dtype=np.float32)
        dx = t * drift_per_frame[0]
        dy = t * drift_per_frame[1]

        for x0, y0, I0 in stars:
            # Apply twinkle as stochastic intensity variation
            intensity = I0 * (1 + np.random.normal(0, twinkle_std))
            add_psf(frame, x0 + dx, y0 + dy, intensity)

        # Optional: add Gaussian noise
        frame += np.random.normal(0, 2, size=image_size)
        stack[t] = frame

    return stack


def save_as_tiff(stack, filename="synthetic_twinkle_50.tif"):
    tiff.imwrite(filename, stack.astype(np.uint16))


# --- Run simulation ---
if __name__ == "__main__":
    stack = simulate_stack()
    save_as_tiff(stack)
    print("Synthetic twinkle stack saved as 'synthetic_twinkle_50.tif'")