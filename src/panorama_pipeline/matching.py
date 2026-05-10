import numpy as np
import matplotlib.pyplot as plt
import cv2

from panorama_pipeline.config import RATIO_THRESHOLD

"""
Flatten all keypoints into a single list for every image
Input:
    images: list of image dictionaries
Output:
    None
"""
def flatten_keypoints(images):
    for image in images:
        image["all_keypoints"] = [
            kp 
            for level in image["keypoints"].values()
            for kp in level
        ]

"""
Perform feature matching between the feature descriptors of two images
Input:
    image_a: image dictionary
    image_b: image dictionary
Output:
    matches: list of match dictionaries
"""
def match_image_pair(image_a, image_b):
    """
    Overview of the algorithm:
        - Compute the sum of square differences between each descriptor to every descriptor in the second image
        - Compute the ratio of the best match to the second best match
        - If the ratio is below the threshold, add the match and all relevant information to the result
    """

    matches = []

    for kp_a in image_a["all_keypoints"]:

        if kp_a["theta"] is None or kp_a["descriptor"] is None:
            continue

        distances = []

        for kp_b in image_b["all_keypoints"]:

            if kp_b["theta"] is None or kp_b["descriptor"] is None:
                continue

            # Compute the difference between the two descriptors and the sum of squared differences
            diff = kp_a["descriptor"] - kp_b["descriptor"]
            ssd = np.sum(diff * diff)
            distances.append((ssd, kp_b))
        
        if len(distances) < 2:
            continue

        # Sort according to the ssd term
        distances.sort(key=lambda x: x[0])

        best_distance, best_kp = distances[0]
        second_distance, second_kp = distances[1]

        if best_distance / second_distance < RATIO_THRESHOLD:
            matches.append({
                "kp_a": kp_a,
                "kp_b": best_kp,
                "distance": best_distance,
                "ratio": best_distance / second_distance,
                "pt_a": (kp_a["orig_col"], kp_a["orig_row"]),
                "pt_b": (best_kp["orig_col"], best_kp["orig_row"])
            })
    
    return matches

"""
Peform feature matching between all pairs of images
Input:
    images: list of image dictionaries
Output:
    pair_matches: dictionary of matches between image pairs
"""
def match_all_image_pairs(images):

    pair_matches = {}

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            matches = match_image_pair(images[i], images[j])

            # Store the result
            pair_matches[(i, j)] = matches

            print(f"           image {i} <-> image {j}: {len(matches)} matches")
    
    return pair_matches
