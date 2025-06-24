import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
from radial_entopy_sharpener import entropy_sharpen

def launch_entropy_sharpen_UI(image, cluster_center):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    img_disp = ax.imshow(image, cmap='gray')
    ax.set_title("Entropy Sharpen Preview")

    # Slider axes
    ax_entropy = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_contrast = plt.axes([0.25, 0.10, 0.65, 0.03])
    ax_power = plt.axes([0.25, 0.05, 0.65, 0.03])

    s_entropy = Slider(ax_entropy, 'Entropy Scale', 0.0, 2.0, valinit=1.0)
    s_contrast = Slider(ax_contrast, 'Contrast Boost', 0.5, 3.0, valinit=1.5)
    s_power = Slider(ax_power, 'Radial Power', 0.5, 5.0, valinit=2.0)

    def update(val):
        new_img = entropy_sharpen(
            image,
            cluster_center,
            entropy_scale=s_entropy.val,
            contrast_boost=s_contrast.val
        )
        img_disp.set_data(new_img)
        fig.canvas.draw_idle()

    for slider in [s_entropy, s_contrast, s_power]:
        slider.on_changed(update)

    plt.show()