import cv2
import math
import matplotlib.pyplot as plt

"""
Build the Gaussian Pyramid for robust, scale/rotation invariant feature description later
Input:
    images: list of image dictionaries
Output:
    level_1/2/3/4: downsampled versions of the original grayscale image (1/2, 1/4, 1/8, 1/16)
    *** Outputs added to the image dictionaries
"""
def build_pyramid(images):

    # Iterate over all images in the list
    for image in images:
        
        current = image["level_0"].copy()

        # Build level 1 (downsample original by factor of 2)
        blurred = cv2.GaussianBlur(current, (5,5), sigmaX=1)
        downsampled = blurred[::2, ::2]
        image["level_1"] = downsampled.copy()

        # Build level 2
        blurred = cv2.GaussianBlur(downsampled, (5,5), sigmaX=1)
        downsampled = blurred[::2, ::2]
        image["level_2"] = downsampled.copy()

        # Build level 3
        blurred = cv2.GaussianBlur(downsampled, (5,5), sigmaX=1)
        downsampled = blurred[::2, ::2]
        image["level_3"] = downsampled.copy()

        # Build level 4
        blurred = cv2.GaussianBlur(downsampled, (5,5), sigmaX=1)
        downsampled = blurred[::2, ::2]
        image["level_4"] = downsampled.copy()


"""
Plot the gaussian pyramid for every image
Input:
    images: list of image dictionaries
Output:
    None
"""
def plot_pyramid(images):
    n = 5
    cols = 3
    rows = math.ceil(n / cols)

    for image in images:
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = axes.flatten()

        fig.suptitle(f"Gaussian Pyramid for {image['name']}")

        for i in range(n):
            ax = axes[i]
            img = image[f"level_{i}"]

            ax.imshow(img, cmap="gray")
            ax.set_title(f"Level {i}", fontsize=10)
            ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()




