import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from panorama_pipeline.config import NUM_LEVELS

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

"""
Display the detected corners for all images in a list, using matplotlib
Input: 
    images (list of image dictionaries): images to be plotted
Output:
    None
"""
def plot_corners(images, raw: bool):

    # Determine the number of images to be plotted
    n = len(images)

    # Personal preference: use a convention of three columns
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    if raw:
        fig.suptitle("Detected Corners", fontsize=16)
    else:
        fig.suptitle("Detected Corners after ANMS", fontsize=16)

    for ax, image in zip(axes, images):
        img = image["color"].copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if raw:
            for row, col in image["corners"][0]:
                img_rgb[row, col] = [255,0,0]
        
        else:
            for row, col in image["best_corners"][0]:
                img_rgb[row, col] = [255,0,0]

        ax.imshow(img_rgb)
        ax.set_title(image["name"], fontsize=10)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
        

"""
Provide a visualization of the harris corner score matrix at every level in the pyramid
Input:
    images: list of image dictionaries
Output: 
    None
"""
def plot_cornerness(images):

    cols = 3
    rows = math.ceil(NUM_LEVELS / cols)

    for image in images:
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = axes.flatten()

        fig.suptitle(f"Corner Response Matrix across Gaussian Pyramid for {image['name']}")

        for i in range(NUM_LEVELS):
            ax = axes[i]
            cornerness_map = image["cmaps"][i]

            cornerness_map[cornerness_map < 0] = 0

            # For plotting purposes, clip any massive outliers
            vmax = np.percentile(cornerness_map, 99.5)

            ax.imshow(cornerness_map, cmap="inferno", vmin=0, vmax=vmax)
            ax.set_title(f"Level {i}", fontsize=10)
            ax.axis("off")

        for ax in axes[NUM_LEVELS:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

"""
Plot the matches between two images
Input:
    image_a: image dictionary
    image_b: image dictionary
    matches: list of dictionaries of match information
Output: 
    None
"""
def plot_matches(image_a, image_b, matches, raw=True):

    h_a, w_a = image_a["color"].shape[:2]
    h_b, w_b = image_b["color"].shape[:2]

    canvas_h = max(h_a, h_b)
    canvas_w = w_a + w_b

    canvas = np.zeros((canvas_h, canvas_w,3), dtype=image_a["color"].dtype)

    image_a_corrected = cv2.cvtColor(image_a["color"], cv2.COLOR_BGR2RGB)
    image_b_corrected = cv2.cvtColor(image_b["color"], cv2.COLOR_BGR2RGB)
    canvas[:h_a, :w_a] = image_a_corrected
    canvas[:h_b, w_a:w_a + w_b] = image_b_corrected

    plt.figure(figsize=(14,7))
    plt.imshow(canvas)
    plt.axis("off")

    if raw:
        plt.title(
            f"Raw Matches: {image_a['name']} <--> {image_b['name']} "
            f"({len(matches)} matches)"
        )
    
    else:
        plt.title(
        f"RANSAC Inliers: {image_a['name']} <--> {image_b['name']} "
        f"({len(matches)} inliers)"
    )

    for match in matches:
        x_a, y_a = match["pt_a"]
        x_b, y_b = match["pt_b"]

        x_b_shifted = x_b + w_a

        plt.plot([x_a, x_b_shifted], [y_a,y_b], linewidth=0.8)
        plt.scatter([x_a, x_b_shifted], [y_a, y_b], s=8)
    
    plt.show()

"""
Plot matches for all image pairs
Input:
    images: list of image dictionaries
    pair_matches: dictionary of image pairs to list of matches
Output:
    None
"""
def plot_all_matches(images, pair_matches, raw=True):
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            plot_matches(images[i], images[j], pair_matches[(i, j)], raw)

"""
Display the final panorama output
Input:
    panorama_canvas: the final output
Output:
    None
"""
def plot_panorama(panorama_canvas, title="Final Panorama"):
    
    image = cv2.cvtColor(panorama_canvas, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 9))
    plt.imshow(image)
    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

