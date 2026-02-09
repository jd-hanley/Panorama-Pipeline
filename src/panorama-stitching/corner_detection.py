import cv2
import numpy as np

def harris_corner_detector():
    pass

'''
Detect corners in the image
Input: Image
Output: Number of strong corners and the corner response map
'''
def detect_corners(img, threshold_factor = 0.01):
    # Check if the input image needs to be converted to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Convert to float for corner detection
    gray = np.float32(gray)

    # Set parameters for the call to the Harris Corner Detector
    # Requires the following arguments
    # img: the grayscale image
    # blockSize: Size of the neighborhood for corner detection
    # ksize: Kernel size of the sobel operator used
    # k: Harris Detector free parameter

    blockSize = 5
    ksize = 3
    k = 0.05

    cmap = harris_corner_detector(gray, blockSize, ksize, k)

    # Apply the threshold, obtain the coordinates of the strong corners, and match the corner scores with the coordinates
    threshold = threshold_factor * cmap.max()
    corner_mask = cmap > threshold
    corner_coords = np.where(corner_mask)

    # Get the list of coordinates in (x,y) format
    corners = list(zip(corner_coords[1], corner_coords[0]))
    corner_scores = [cmap[y, x] for x, y in corners]

    return cmap, corners, corner_scores


def plot_corners():
    pass

def anms():
    pass
