import cv2
import math
import matplotlib.pyplot as plt

from config import NUM_LEVELS, BLUR_KERNEL

"""
Build the Gaussian Pyramid for robust, scale/rotation invariant feature description later
Input:
    images: list of image dictionaries
Output:
    level 1/2/3/4: downsampled versions of the original grayscale image (1/2, 1/4, 1/8, 1/16)
    *** Outputs added to the image dictionaries (in the "gray" list of the dictionary)
"""
def build_pyramid(images):

    for image in images:
        for _ in range(1, NUM_LEVELS):
            # We want to downsample from the last entry in the array
            curr = image["gray"][-1].copy()

            # Apply Gaussian blurring
            blurred = cv2.GaussianBlur(curr, (BLUR_KERNEL, BLUR_KERNEL), sigmaX = 1)

            # Downsample
            downsampled = blurred[::2, ::2]

            # Add to the array
            image["gray"].append(downsampled)


"""
Plot the gaussian pyramid for every image
Input:
    images: list of image dictionaries
Output:
    None
"""
def plot_pyramid(images):

    cols = 3
    rows = math.ceil(NUM_LEVELS / cols)

    for image in images:
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = axes.flatten()

        fig.suptitle(f"Gaussian Pyramid for {image['name']}")

        for i in range(NUM_LEVELS):
            ax = axes[i]
            img = image["gray"][i]

            ax.imshow(img, cmap="gray")
            ax.set_title(f"Level {i}", fontsize=10)
            ax.axis("off")

        for ax in axes[NUM_LEVELS:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()




