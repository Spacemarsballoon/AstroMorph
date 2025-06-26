import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
from radial_entopy_sharpener import entropy_sharpen


def launch_entropy_sharpen_UI(image, cluster_center):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    # img_disp = ax.imshow(image, cmap='gray')
    img_disp = ax.imshow(image, cmap='inferno')
    # diff_img = np.clip(new_img - image, 0, 1)
    # ax_diff.imshow(diff_img, cmap='inferno')
    # ax_diff.set_title("Sharpening Difference Map")
    ax.set_title("Entropy Sharpen Preview")

    # Slider axes
    ax_entropy = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_contrast = plt.axes([0.25, 0.10, 0.65, 0.03])
    ax_power = plt.axes([0.25, 0.05, 0.65, 0.03])

    s_entropy = Slider(ax_entropy, 'Entropy Scale', 0.01, 0.70, valinit=0.10)
    s_contrast = Slider(ax_contrast, 'Contrast Boost', 0.05, 0.50, valinit=0.15)
    s_power = Slider(ax_power, 'Radial Power', 0.05, 100.00, valinit=0.20)

    # weights = radial_weight_map(image.shape, cluster_center, power=s_power.val)
    # ax.contour(weights, levels=5, colors='cyan', alpha=0.3)

    def update(val):
        new_img = entropy_sharpen(
            image,
            cluster_center,
            entropy_scale=s_entropy.val,
            contrast_boost=s_contrast.val,
            power=s_power.val  # 👈 now interactive!
        )
        print("New image range:", new_img.min(), new_img.max())
        img_disp.set_data(new_img)
        img_disp.set_clim(0, 1)
        fig.canvas.draw()


    for slider in [s_entropy, s_contrast, s_power]:
        slider.on_changed(update)

    plt.show()




if __name__=="__main__":
    image = io.imread("C:/path/to/AstroMorph/AstroMorph/data/M51.jpg")
    launch_entropy_sharpen_UI(image,(image.shape[0] // 2, image.shape[1] // 2))