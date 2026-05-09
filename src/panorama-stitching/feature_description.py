import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import NUM_LEVELS, DESCRIPTOR_SIZE, INNER_PATCH_SIZE, OUTER_PATCH_SIZE

"""
For feature descriptors to be orientation invariant, need to compute the dominant orientation using the local image gradients
Input:
    images: list of image dictionaries
Output:
    theta: keypoint orientation in radians
*** Output added to image dictionary keypoint entries
"""
def estimate_keypoint_orientations(images):
    """ 
    What is the approach here?
        - Look at an 11 x 11 window around every keypoint that has been found
        - Compute the Gaussian weighted average of the x and y gradients across all pixels in the window
        - Compute the resulting angle
    Why?
        - Gives an estimate of the dominant orientation of the feature by looking at behavior in the area
    """
    
    window_size = 11
    half = window_size // 2

    for image in images:
        for level in range(NUM_LEVELS):
            gray = np.float64(image["gray"][level])

            # Compute the gradients in the x and y directions for every pixel
            ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            keypoints = image["keypoints"][level]

            for kp in keypoints:
                row = kp["row"]
                col = kp["col"]

                # Window boundaries
                r_start = row - half
                r_end = row + half + 1
                c_start = col - half
                c_end = col + half + 1

                # Skip keypoints that too close to the border
                if r_start < 0 or c_start < 0 or r_end > gray.shape[0] or c_end > gray.shape[1]:
                    kp["theta"] = None
                    continue
                
                # Grab the image gradients in the relevant patch
                ix_patch = ix[r_start:r_end, c_start:c_end]
                iy_patch = iy[r_start:r_end, c_start:c_end]

                # Use gaussian weighting to give nearby gradients more influence
                weights = cv2.getGaussianKernel(window_size, sigma=2)
                weights = weights @ weights.T

                sx = np.sum(weights * ix_patch)
                sy = np.sum(weights * iy_patch)

                kp["theta"] = np.arctan2(sy,sx)


"""
Implementation of Multi-Scale Oriented Patch feature description for more robust feature description
Input:
    images: list of image dictionaries
Output:
    descriptors: normalized feature descriptors
*** All feature descriptors added into the keypoint dictionaries for each image
"""
def compute_mops_descriptors(images):

    for image in images:
        for level in range(NUM_LEVELS):
            gray = np.float64(image["gray"][level])
            keypoints = image["keypoints"][level]

            for kp in keypoints:

                # If the keypoint in question was near a boundary, continue
                if kp["theta"] is None:
                    kp["descriptor"] = None
                    continue

                # Relevant information for this keypoint
                theta = kp["theta"]
                row = kp["row"]
                col = kp["col"]

                # My strategy will be to take a large patch around the image
                # Start by obtaining the outer image patch
                half_outer = OUTER_PATCH_SIZE // 2

                r_start_outer = row - half_outer
                r_end_outer = row + half_outer + 1
                c_start_outer = col - half_outer
                c_end_outer = col + half_outer + 1

                # Probably going to be cutting a lot of points here
                # But if we are too close to the edge then too bad so sad
                if r_start_outer < 0 or c_start_outer < 0 or r_end_outer > gray.shape[0] or c_end_outer > gray.shape[1]:
                    kp["descriptor"] = None
                    continue
            
                # With the boundaries calculated, grab the patch of interest
                outer_patch = gray[r_start_outer:r_end_outer, c_start_outer:c_end_outer]

                # Now need to rotate the patch according to theta
                # Need to compute the rotation matrix and then apply via an affine warp

                # Convert theta to degrees
                theta_deg = np.degrees(theta)

                # Obtain the rotation matrix
                center = (OUTER_PATCH_SIZE / 2, OUTER_PATCH_SIZE / 2)
                M = cv2.getRotationMatrix2D(center, -theta_deg, 1.0)

                # Obtain the rotated patch
                rotated_patch = cv2.warpAffine(outer_patch, M, (OUTER_PATCH_SIZE, OUTER_PATCH_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                # Grab the center of the rotated patch
                half_inner = INNER_PATCH_SIZE // 2
                center_index = OUTER_PATCH_SIZE // 2

                r_start_inner = center_index - half_inner
                r_end_inner = center_index + half_inner + 1
                c_start_inner = center_index - half_inner
                c_end_inner = center_index + half_inner + 1

                inner_patch = rotated_patch[r_start_inner:r_end_inner, c_start_inner:c_end_inner]

                # Resize to 8 x 8
                descriptor_patch = cv2.resize(inner_patch, (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), interpolation=cv2.INTER_AREA)

                # Flatten to 64 x 1 vector
                descriptor = np.float64(descriptor_patch.flatten())

                # Normalize by subtracting the mean and dividing by standard deviation
                descriptor -= np.mean(descriptor)
                std = np.std(descriptor)
                if std > 1e-6:
                    descriptor /= std
                
                kp["descriptor"] = descriptor

    
            
