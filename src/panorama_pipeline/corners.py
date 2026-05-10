import cv2
import numpy as np
from scipy.ndimage import maximum_filter
import math
import matplotlib.pyplot as plt

from panorama_pipeline.config import HARRIS_BLOCK_SIZE, HARRIS_FREE_PARAMETER, HARRIS_K_SIZE, HARRIS_THRESHOLD_FACTOR, NUM_LEVELS, BLUR_KERNEL, MAX_FILTER_WINDOW, ANMS_FEATURES_PER_LEVEL

'''
Detect corners in an image
Input: 
    images: (list of image dictionaries)
Output:
    cmap (np.ndarray): cornerness score matrix
    corners (list of (row, col) tuples): locations of strong (exceeded threshold) corners in the image
    corner_scores (list of floats): corners scores corresponding to the locations in corners
    *** Output all added to each image's dictionary at each level
'''
def detect_corners(images):

    # Iterate over all images in the list, detect corners at every level, update dictionary
    for image in images:
        for level in range(NUM_LEVELS):

            # Convert to float for corner detection
            gray = np.float64(image["gray"][level])

            # Obtain the "cornerness" score matrix using Harris Corner Detector
            cmap = harris_corner(gray, HARRIS_BLOCK_SIZE, HARRIS_K_SIZE, HARRIS_FREE_PARAMETER)

            # Apply the threshold and determine the coordinates of strong corners
            thresholded = HARRIS_THRESHOLD_FACTOR * cmap.max()
            thresholded_cmap = cmap > thresholded
            corner_coords = np.where(thresholded_cmap)

            # Get the list of coordinates in (row, col) format
            corners = list(zip(corner_coords[0], corner_coords[1]))
            corner_scores = [cmap[row, col] for row, col in corners]

            image["cmaps"].append(cmap)
            image["corners"].append(corners)
            image["corner_scores"].append(corner_scores)

"""
Implement the harris corner detector
Input: 
    gray: grayscale image of interest
    block_size: size of the neighborhood for corner detection
    k_size: kernel size of the sobel operator used
    k: harris detector free parameter
Output:
    cmap: cornerness score matrix (how likely each pixel is to be a corner)
"""
def harris_corner(gray, block_size, k_size, k):
    # High level pseudocode:
    #   Apply Gaussian Blur to smooth out any noise 
    #   Apply to sobel operator to find the x and y gradient values for every pixel in the image
    #   For each pixel in the image, consider the 3x3 window around it and compute the corner strength function
    #   Return the corner strength map

    # Start by applying gaussian blur
    smoothed = cv2.GaussianBlur(gray, (BLUR_KERNEL,BLUR_KERNEL), sigmaX=1)

    # Apply the sobel operator to get the gradient at every pixel 
    ix = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, k_size)
    iy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, k_size)

    # Compute the other components of the moment matrix
    ixx = ix * ix
    iyy = iy * iy
    ixy = ix * iy

    # Construct the M matrices
    # Recall what this is doing: the moment matrix entries are the sums of ixx, iyy, and ixy in the window of interest
    # Using gaussian blur to compute this just means the sum is weighted towards more relevant pixels in the window
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
Perform Adaptive Non-Maximal Supression
Input:
    images: (list of image dictionaries)
    n: desired number of strong corners
Output:
    best_corners (list of (row, col) tuples): locations of the n best corners
    *** Output added to the image dictionaries, not returned
"""
def anms(images):
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

    # We need to run this over every level in every image, which could add up for larger datasets
    for image in images:

        tracker = []
        for level in range(NUM_LEVELS):
            cmap = image["cmaps"][level]

            # Obtain the local maxima 
            # maximum_filter replaces each element with the maximum value in a specified window
            # local_maxima will set all pixels that are local maxima to one and all other pixels to zero
            local_maxima = (cmap == maximum_filter(cmap, size=MAX_FILTER_WINDOW)) & (cmap > HARRIS_THRESHOLD_FACTOR * cmap.max())

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
                        # Now need to determine how far away it is
                        distance = (coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2
                        if distance < distances[i]:
                            distances[i] = distance
            
            # Note that our list of distances is implicitly linked to our list of coordinates (same ordering)
            # I can use argsort to obtain the indices that would sort distances
            indices = np.argsort(distances)[::-1]

            # Now, sort according to those indices
            sorted_coordinates = coordinates[indices]

            if len(sorted_coordinates) < ANMS_FEATURES_PER_LEVEL:
                image["best_corners"].append(sorted_coordinates)
                tracker.append(len(sorted_coordinates))
            else:
                image["best_corners"].append(sorted_coordinates[:ANMS_FEATURES_PER_LEVEL])
                tracker.append(ANMS_FEATURES_PER_LEVEL)
        
        print(f"           {image['name']}: {tracker}")

"""
All keypoints across all levels have been detected, clean up the structure to make the rest of the pipeline more straightforward
Input:
    images: list of image dictionaries
Output:
    None
"""
def initialize_keypoints(images):

    for image in images:
        image["keypoints"] = {}

        for level in range(NUM_LEVELS):
            corners = image["best_corners"][level]

            image["keypoints"][level] = []
            for row, col in corners:

                # For keypoints at lower levels, compute their coordinate in the original image
                scale = 2 ** level
                orig_row = row * scale
                orig_col = col * scale

                image["keypoints"][level].append({
                    "row": row,
                    "col": col,
                    "orig_row": orig_row,
                    "orig_col": orig_col,
                    "theta": None,
                    "level": level,
                    "descriptor": None
                })


