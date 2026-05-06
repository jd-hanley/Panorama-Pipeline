import cv2
import numpy as np
from plot_images import show_image
from scipy.ndimage import maximum_filter
import math
import matplotlib.pyplot as plt

'''
Detect corners in an image
Input: 
    images: (list of image dictionaries)
    threshold_factor: constant used to determine what constitutes a strong corner, relative to the max corner score
Output:
    cmap (np.ndarray): cornerness score matrix
    corners (list of (row, col) tuples): locations of strong (exceeded threshold) corners in the image
    corner_scores (list of floats): corners scores corresponding to the locations in corners
    *** Output all added to each image's dictionary
'''
def detect_corners(images, threshold_factor = 0.005):

    # Iterate over all images in the list, detect corners, update dictionary
    for image in images:

        # Convert to float for corner detection
        gray = np.float64(image["gray"])

        # Set parameters for the call to the Harris Corner Detector
        # Requires the following arguments
        # img: the grayscale image
        # blockSize: Size of the neighborhood for corner detection
        # ksize: Kernel size of the sobel operator used
        # k: Harris Detector free parameter

        block_size = 5
        k_size = 3
        k = 0.05

        # Obtain the "cornerness" score matrix
        # cmap = cv2.cornerHarris(gray, block_size, k_size, k)
        cmap = harris_corner(gray, block_size, k_size, k)

        # Apply the threshold and determine the coordinates of strong corners
        thresholded = threshold_factor * cmap.max()
        t_cmap = cmap > thresholded
        corner_coords = np.where(t_cmap)

        # Get the list of coordinates in (row, col) format
        corners = list(zip(corner_coords[0], corner_coords[1]))
        corner_scores = [cmap[row, col] for row, col in corners]

        image["cmap"] = cmap
        image["corners"] = corners
        image["corner_scores"] = corner_scores
        image["threshold_factor"] = threshold_factor
    
    return images

"""
Implement the harris corner detector
Input: 
    gray: grayscale image of interest
    block_size: size of the neighborhood for corner detection
    ksize: kernel size of the sobel operator used
    k: harris detector free parameter
Output:
    cmap: cornerness score matrix (how likely each pixel is to be a corner)
    """
def harris_corner(gray, block_size, k_size, k):
    # High level pseudocode to keep myself focused:
    #   Apply Gaussian Blur to smooth out any noise 
    #   Apply to sobel operator to find the x and y gradient values for every pixel in the image
    #   For each pixel in the image, consider the 3x3 window around it and compute the corner strength function
    #   Return the corner strength map

    # Start by applying gaussian blur
    smoothed = cv2.GaussianBlur(gray, (5,5), sigmaX=1)

    # Apply the sobel operator 
    ix = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, k_size)
    iy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, k_size)

    # Compute the components of the M matrix for every single window
    ixx = ix * ix
    iyy = iy * iy
    ixy = ix * iy

    # Construct the M matrices
    mxx = cv2.GaussianBlur(ixx, (block_size, block_size), sigmaX=1)
    myy = cv2.GaussianBlur(iyy, (block_size, block_size), sigmaX=1)
    mxy = cv2.GaussianBlur(ixy, (block_size, block_size), sigmaX=1)

    # At this point I have the components of the M matrix for every pixel in the image
    # I can now compute the determinant of every M matrix
    det = mxx * myy - mxy * mxy
    trace = mxx + myy

    # The harris corner score equation is: det(M) - k(trace(M))^2
    cmap = det - k * trace * trace

    return cmap


"""
Provide a visualization of the harris corner score matrix
Input:
    images: (list of image dictionaries)
Output: 
    None
"""
def plot_cornerness(images):
    # Determine the number of cmaps to be plotted
    n = len(images)

    # Personal preference: use a convention of four columns
    cols = 3

    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    fig.suptitle("Corner Response Score from Harris Corner Detector", fontsize=16)

    for ax, image in zip(axes, images):
        cornerness_map = image["cmap"]

        cornerness_map[cornerness_map < 0] = 0

        vmax = np.percentile(cornerness_map, 99.5)

        ax.imshow(cornerness_map, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(image["name"], fontsize=10)
        ax.axis("off")

    for ax in axes[len(images):]:
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

    # Personal preference: use a convention of four columns
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
            for row, col in image["corners"]:
                img_rgb[row, col] = [255,0,0]
        
        else:
            for row, col in image["best_corners"]:
                img_rgb[row, col] = [255,0,0]

        ax.imshow(img_rgb)
        ax.set_title(image["name"], fontsize=10)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
        

"""
Perform Adaptive Non-Maximal Supression
Input:
    images: (list of image dictionaries)
    n: desired number of strong corners
Output:
    best_corners (list of (row, col) tuples): locations of the n best corners
    """
def anms(images, n):
    """ 
    Overview of the algorithm:
        1) Locate all regional maxima in cornerness score map
        2) Allocate a distance array with length corresponding to the number of maxima
        3) Iterate over all pairs of points, determining the distance to the nearest maxima greater than the current point
        4) Sort the distance array and corresponding maxima array in descending order
        5) Return the first n entries
    Why? This gives us the n corners that are local maxima and far away from better corners
    We get corners that are both strong and spaced out 
    """

    # Takes forever... communicate with the command line
    print(f"Beginning Adaptive Non-Maximal Supression")
    count = 1
    # Iterate over all images of interest
    for image in images:
        print(f"Suppressing corners in image {count}")
        count += 1
        cmap = image["cmap"]
        # Obtain the local maxima
        # maximum_filter replaces each element with the maximum value in a specified window
        # local_maxima will set all pixels that are their own local maximum to one and all other pixels to 0
        local_maxima = (cmap == maximum_filter(cmap, size = 5)) & (cmap > image["threshold_factor"] * cmap.max())

        # Extract the (row, col) coordinates where we have local maxima
        row, col = np.where(local_maxima)
        coordinates = np.column_stack((row, col))

        # At this point, we have the coordinates of strong corner candidates, but we need to spread them out
        n_candidates = len(coordinates)
        print(n_candidates)
        # Initialize all distances to inf
        distances = np.full(n_candidates, np.inf)
        for i in range(n_candidates):
            for j in range(n_candidates):
                if cmap[coordinates[j][0], coordinates[j][1]] > cmap[coordinates[i][0], coordinates[i][1]]:
                    # We have a corner that is stronger than the one under consideration
                    distance = (coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2
                    if distance < distances[i]:
                        distances[i] = distance
        
        # Note that our list of distances is implicitly linked to our list of coordinates (same ordering)
        # I can use argsort to obtain the indices that would sort distances
        indices = np.argsort(distances)[::-1]

        # Now, sort according to those indices
        sorted_distances = distances[indices]
        sorted_coordinates = coordinates[indices]

        if len(sorted_coordinates) < n:
            image["best_corners"] = sorted_coordinates
        else:
            image["best_corners"] = sorted_coordinates[:n]
