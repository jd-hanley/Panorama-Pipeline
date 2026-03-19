import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Encode the information at each feature point with a vector
Input:
    best_corners (list of (row, col) tuples): locations of the n best corners
    image (np.ndarray): grayscale or BGR image
Output:
    feature_desc (dict of (row, col) tuple : (np.ndarray)): correspondence of coordinate to 64 x 1 feature descriptor """
def create_feature_desc(image, best_corners):
    """
    Overview of a simple approach to feature descriptors:
        1) Take a square patch from the image, centered around the point of interest 
        2) Apply Gaussian blur
        3) Subsample the blurred patch to 8 x 8
        4) Reshape into a 64 x 1 vector
        5) Standardize to have a mean of 0 and a variance of 1 """
    
    # Define patch size, blur kernel, and subsample size
    patch_size = 40 # 40 x 40 patch
    patch_step = int(patch_size/2)
    subsample_size = 8 # 8 x 8 after subsampling

    kernel_size = 5 # 5 x 5 kernel for blur convolution

    # Want a grayscale image
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    feature_desc = {}

    # Generate the feature descriptors for each corner
    for corner in best_corners:
        # Grab the patch centered around the point of interest (ensure within boundaries)
        # Disregard corners whose patch is not within the boundaries
        row_start = corner[0] - patch_step
        row_end = corner[0] + patch_step
        col_start = corner[1] - patch_step
        col_end = corner[1] + patch_step

        if row_start < 0 or row_end > height or col_start < 0 or col_end > width:
            continue

        patch = gray[row_start:row_end, col_start:col_end]

        # Apply Gaussian blur to the patch
        blurred = cv2.GaussianBlur(patch, (kernel_size,kernel_size),0)

        # Subsample to 8x8
        subsample = cv2.resize(blurred, (subsample_size, subsample_size))

        # Reshape the subsampled patch to a 64x1 vector
        resized = subsample.reshape((64,1)).astype(np.float32)

        mean = np.mean(resized)
        sd = np.std(resized)

        standardized = (resized - mean) / sd

        # Add the feature descriptor to the dictionary
        feature_desc[(corner[0], corner[1])] = standardized

    return feature_desc

"""
Display feature descriptors for the best corners in an image using Matplotlib
Input:
    feature_desc (dict of (row, col) tuple : (np.ndarray)): correspondence of coordinate to 64 x 1 feature descriptor
Output: 
    None """
def show_feature_desc(feature_desc):
    
    # Stack all of the feature descriptor vectors for the image
    stacked = np.hstack(list(feature_desc.values()))

    # Plot the feature descriptors
    plt.imshow(stacked, cmap='gray')
    plt.axis("off")
    plt.show()
