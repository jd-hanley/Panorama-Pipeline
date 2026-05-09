import cv2
import matplotlib.pyplot as plt
import math

"""
Believe this is legacy code
Display an image using matplotlib
Input:
    image (np.ndarray): Grayscale or BGR image.
Output:
    None
"""
def show_image(image):
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

"""
Display all images in the list using matplotlib
Input:
    images (list of image dictionaries)
Output:
    None
"""
def plot_images(images):

    # Determine the number of images to be plotted
    n = len(images)

    # Personal preference: use a convention of three columns
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    fig.suptitle("Input Dataset", fontsize=16)

    for ax, image in zip(axes, images):
        img = image["color"]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img_rgb)
        ax.set_title(image["name"], fontsize=10)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()



