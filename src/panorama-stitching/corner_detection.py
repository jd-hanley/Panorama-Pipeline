import cv2
import numpy as np
from plot_images import show_image

'''
Detect corners in an image
Input: 
    image (np.ndaray): image under consideration
    threshold_factor: constant used to determine what constitutes a strong corner, relative to the max corner score
Output:
    cmap (np.ndarray): cornerness score matrix
    corners (list of (row, col) tuples): locations of strong (exceeded threshold) corners in the image
    corner_scores (list of floats): corners scores corresponding to the locations in corners
'''
def detect_corners(image, threshold_factor = 0.01):
    # Check if the input image needs to be converted to grayscale
    # Also perform corner detection on a copy of the image 
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Convert to float for corner detection
    gray = np.float32(gray)

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
    cmap = cv2.cornerHarris(gray, block_size, k_size, k)

    # Apply the threshold and determine the coordinates of strong corners
    thresholded = threshold_factor * cmap.max()
    corner_mask = cmap > thresholded
    corner_coords = np.where(corner_mask)

    # Get the list of coordinates in (row, col) format
    corners = list(zip(corner_coords[0], corner_coords[1]))
    corner_scores = [cmap[row, col] for row, col in corners]

    return cmap, corners, corner_scores

"""
Display the detected corners for all images in a list, using matplotlib
Input: 
    images (list of image dictionaries): images to be plotted
Output:
    None
"""
def plot_corners(images):
    # Iterate over all images plotting the image with the corners overlaid (red)
    for image in images:

        # For plotting purposes right now, disregard cmap and corner_scores
        cmap, corners, corner_scores = detect_corners(image["gray"])
        
        # Since I will be changing the colors of the individual pixels, make a copy of the image
        copy = image["color"].copy()
        for row, col in corners:
            # Change the color of strong candidate corners to red
            copy[row, col] = [0,0,255]
        show_image(copy)
        


def anms():
    pass
