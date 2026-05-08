import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
For feature descriptors to be orientation invariant, need to compute the dominant orientation using the local image gradients
Input:
    images: list of image dictionaries
Output:
    theta: keypoint orientation in radians
*** Output added to image dictionary entries
"""
def estimate_keypoint_orientations(images):
    
    window_size = 11
    half = window_size // 2
    num_levels = 5

    for image in images:
        for level in range(num_levels):
            gray = np.float64(image[f"level_{level}"])

            # Compute the gradients in the x and y directions for every pixel
            ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            keypoints = image["keypoints"][level]

            for kp in keypoints:
                row = kp["row"]
                col = kp["col"]

                # Window boundaries
                r0 = row - half
                r1 = row + half + 1
                c0 = col - half
                c1 = col + half + 1

                # Skip keypoints that too close to the border
                if r0 < 0 or c0 < 0 or r1 > gray.shape[0] or c1 > gray.shape[1]:
                    kp["theta"] = None
                    continue
                
                # Grab the image gradients in the relevant patch
                ix_patch = ix[r0:r1, c0:c1]
                iy_patch = iy[r0:r1, c0:c1]

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
    num_levels = 5
    patch_size = 8
    inner_size = 40
    outer_size = 60

    for image in images:
        for level in range(num_levels):
            gray = image[f"level_{level}"].astype(np.float64)
            keypoints = image["keypoints"][level]

            for kp in keypoints:
                if kp["theta"] is None:
                    kp["descriptor"] = None
                    continue

                # Relevant information for this keypoint
                theta = kp["theta"]
                row = kp["row"]
                col = kp["col"]

                # My strategy will be to take a large patch around the image
                # Start by obtaining the outer image patch
                half_outer = outer_size // 2

                r0_outer = row - half_outer
                r1_outer = row + half_outer + 1
                c0_outer = col - half_outer
                c1_outer = col + half_outer + 1

                # Probably going to be cutting a lot of points here
                # But if we are too close to the edge then too bad
                if r0_outer < 0 or c0_outer < 0 or r1_outer > gray.shape[0] or c1_outer > gray.shape[1]:
                    kp["descriptor"] = None
                    continue
            
                # With the boundaries calculated, grab the patch of interest
                outer_patch = gray[r0_outer:r1_outer, c0_outer:c1_outer]

                # Now need to rotate the patch according to theta
                # Need to compute the rotation matrix and then apply via an affine warp

                # Convert theta to degrees
                theta_deg = np.degrees(theta)

                # Obtain the rotation matrix
                center = (outer_size / 2, outer_size / 2)
                M = cv2.getRotationMatrix2D(center, -theta_deg, 1.0)

                # Obtain the rotated patch
                rotated_patch = cv2.warpAffine(outer_patch, M, (outer_size, outer_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                # Grab the center of the rotated patch
                half_inner = inner_size // 2
                center_index = outer_size // 2

                r0_inner = center_index - half_inner
                r1_inner = center_index + half_inner + 1
                c0_inner = center_index - half_inner
                c1_inner = center_index + half_inner + 1

                inner_patch = rotated_patch[r0_inner:r1_inner, c0_inner:c1_inner]

                # Resize to 8 x 8
                descriptor_patch = cv2.resize(inner_patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

                # Flatten to 64 x 1 vector
                descriptor = descriptor_patch.flatten().astype(np.float64)

                # Normalize by subtracting the mean and dividing by standard deviation
                descriptor -= np.mean(descriptor)
                std = np.std(descriptor)
                if std > 1e-6:
                    descriptor /= std
                
                kp["descriptor"] = descriptor
    
                

# """
# Encode the information at each feature point with a vector
# Input:
#     best_corners (list of (row, col) tuples): locations of the n best corners
#     image (np.ndarray): grayscale or BGR image
# Output:
#     feature_desc (dict of (row, col) tuple : (np.ndarray)): correspondence of coordinate to 64 x 1 feature descriptor """
# def create_feature_desc(image, best_corners):
#     """
#     Overview of a simple approach to feature descriptors:
#         1) Take a square patch from the image, centered around the point of interest 
#         2) Apply Gaussian blur
#         3) Subsample the blurred patch to 8 x 8
#         4) Reshape into a 64 x 1 vector
#         5) Standardize to have a mean of 0 and a variance of 1 """
    
#     # Define patch size, blur kernel, and subsample size
#     patch_size = 40 # 40 x 40 patch
#     patch_step = int(patch_size/2)
#     subsample_size = 8 # 8 x 8 after subsampling

#     kernel_size = 5 # 5 x 5 kernel for blur convolution

#     # Want a grayscale image
#     if len(image.shape) == 2:
#         gray = image.copy()
#     else:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     height, width = gray.shape

#     feature_desc = {}

#     # Generate the feature descriptors for each corner
#     for corner in best_corners:
#         # Grab the patch centered around the point of interest (ensure within boundaries)
#         # Disregard corners whose patch is not within the boundaries
#         row_start = corner[0] - patch_step
#         row_end = corner[0] + patch_step
#         col_start = corner[1] - patch_step
#         col_end = corner[1] + patch_step

#         if row_start < 0 or row_end > height or col_start < 0 or col_end > width:
#             continue

#         patch = gray[row_start:row_end, col_start:col_end]

#         # Apply Gaussian blur to the patch
#         blurred = cv2.GaussianBlur(patch, (kernel_size,kernel_size),0)

#         # Subsample to 8x8
#         subsample = cv2.resize(blurred, (subsample_size, subsample_size))

#         # Reshape the subsampled patch to a 64x1 vector
#         resized = subsample.reshape((64,1)).astype(np.float32)

#         mean = np.mean(resized)
#         sd = np.std(resized)

#         standardized = (resized - mean) / sd

#         # Add the feature descriptor to the dictionary
#         feature_desc[(corner[0], corner[1])] = standardized

#     return feature_desc

# """
# Display feature descriptors for the best corners in an image using Matplotlib
# Input:
#     feature_desc (dict of (row, col) tuple : (np.ndarray)): correspondence of coordinate to 64 x 1 feature descriptor
# Output: 
#     None """
# def show_feature_desc(feature_desc):
    
#     # Stack all of the feature descriptor vectors for the image
#     stacked = np.hstack(list(feature_desc.values()))

#     # Plot the feature descriptors
#     plt.imshow(stacked, cmap='gray')
#     plt.axis("off")
#     plt.show()
