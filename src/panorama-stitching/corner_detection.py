import cv2
import numpy as np
from plot_images import show_image
from scipy.ndimage import maximum_filter

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
def detect_corners(image, threshold_factor = 0.005):
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
        

"""
Perform Adaptive Non-Maximal Supression
Input:
    cmap (np.ndarray): cornerness score matrix
    n: desired number of strong corners
Output:
    best_corners (list of (row, col) tuples): locations of the n best corners
    """
def anms(cmap, n):
    """ 
    Overview of the algorithm:
        1) Locate all regional maxima in cmap
        2) Allocate a distance array with length corresponding to the number of maxima
        3) Iterate over all pairs of points, determining the distance to the nearest maxima greater than the current point
        4) Sort the distance array and corresponding maxima array in descending order
        5) Return the first n entries
    Why? This gives us the n corners that are local maxima and far away from better corners
    We get corners that are both strong and spaced out """

    # Obtain the local maxima
    # Window size is chosen arbitrarily... come back to this later to determine something better
    local_maxima = (cmap == maximum_filter(cmap, size = 4))

    # Extract the (row, col) coordinates where we have local maxima
    row, col = np.where(local_maxima)
    coordinates = np.column_stack((row, col))

    # At this point, we have the coordinates of strong corner candidates, but we need to spread them out
    n_candidates = len(coordinates)
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

    if len(sorted_coordinates < n):
        return sorted_coordinates
    return sorted_coordinates[:n]
