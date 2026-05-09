import random
import numpy as np
from homography import compute_homography
from feature_matching import plot_matches

from config import RANSAC_ITERATIONS, RANSAC_THRESHOLD, RANSAC_INLIER_COUNT, RANSAC_MIN_PTS

"""
Perform Random Sample Consensus (RANSAC) to find the best homography matrix
Input: 
    matched_pairs: list of image coordinate pair structures between two images
Output:
    models: data structure containing H matrix, inlier points, number of inliers, and mean error
"""
def ransac(matched_pairs):

    # We need a minimum of four points to compute homography
    if len(matched_pairs) < 4:
        return None
    
    best_H = None
    best_inliers = []
    max_inliers = 0

    iterations = RANSAC_ITERATIONS
    while iterations > 0:
        iterations -= 1

        # Randomly select 4 pairs of matched points
        sample = random.sample(matched_pairs, RANSAC_MIN_PTS)

        # Compute the homography matrix
        H = compute_homography(sample)

        # Recall I said that Homography is from image a to image b
        inliers = []

        for pair in matched_pairs:

            # Project point from image a into image b, compute distance to point in image b
            # Make sure the point is homogeneous
            x_a, y_a = pair["pt_a"]
            pt_a = np.array([x_a, y_a, 1.0])

            # Project the point into the second image using the current H matrix
            proj_pt_b = H @ pt_a

            # Return to cartesian coordinates
            if abs(proj_pt_b[2]) < 1e-12:
                continue
            proj_pt_b /= proj_pt_b[2]

            # Compute the distance between the projected and the actual point
            x_b, y_b = pair["pt_b"]
            distance = np.sqrt((x_b - proj_pt_b[0])**2 + (y_b - proj_pt_b[1])**2)

            # If the distance is less than the threshold, we have an inlier
            if distance < RANSAC_THRESHOLD:
                inliers.append(pair)
        
        # Terminate early if we find a homography that meets the inlier count
        if len(inliers) > RANSAC_INLIER_COUNT:
            H = compute_homography(inliers)
            return {
                "H": H,
                "inliers": inliers,
                "num_inliers": len(inliers),
            }

        # Otherwise check if this is better than the best so far
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers
            best_H = H.copy()
    
    # If we run all iterations and finish we just return the best result we found
    best_H = compute_homography(best_inliers)
    return {
        "H": best_H,
        "inliers": best_inliers,
        "num_inliers": max_inliers
    }


"""
Estimate all pairwise homography matrices and remove outliers in the feature correspondences for later plotting
Input:
    pair_matches: dictionary of image pairs (i, j) to feature matching information
Output:
    pair_models: dictionary of image pairs (i, j) to model information (H matrix, inliers, inlier count)
"""
def estimate_all_pairwise_homographies(pair_matches):

    print(f"Estimating homographies between all image pairs......")

    pair_models = {}

    for pair, matches in pair_matches.items():
        model = ransac(matches)

        if model is None:
            continue

        pair_models[pair] = model
        print(f"{pair}: {model['num_inliers']} inliers")

    return pair_models

"""
Display all matches after running RANSAC
"""
def plot_all_inlier_matches(images, pair_models):
    for (i, j), model in pair_models.items():
        plot_matches(images[i], images[j], model["inliers"], raw=False)

