import numpy as np
import matplotlib.pyplot as plt
import cv2

from config import RATIO_THRESHOLD

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
    matches: list of dictionaries
"""
def match_image_pair(image_a, image_b):
    """
    Overview of the algorithm:
        - Compute the sum of square differences between each descriptor to every descriptor in the second image
        - Compute the ratio of the best match to the second best match
        - If the ratio is below the threshold, add the match and all relevant information to the result
    """
    # Start by flattening the keypoints to a single data structure
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

    print("Beginning feature descriptor matching between images......")
    pair_matches = {}

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            matches = match_image_pair(images[i], images[j])

            # Store the result
            pair_matches[(i, j)] = matches

            print(f"Image {i} <-> Image {j}: {len(matches)} matches")
    
    return pair_matches

"""
Plot the matches between two images
Input:
    image_a: image dictionary
    image_b: image dictionary
    matches: list of dictionaries of match information
Output: 
    None
"""
def plot_matches(image_a, image_b, matches):

    h_a, w_a = image_a["color"].shape[:2]
    h_b, w_b = image_b["color"].shape[:2]

    canvas_h = max(h_a, h_b)
    canvas_w = w_a + w_b

    canvas = np.zeros((canvas_h, canvas_w,3), dtype=image_a["color"].dtype)

    image_a_corrected = cv2.cvtColor(image_a["color"], cv2.COLOR_BGR2RGB)
    image_b_corrected = cv2.cvtColor(image_b["color"], cv2.COLOR_BGR2RGB)
    canvas[:h_a, :w_a] = image_a_corrected
    canvas[:h_b, w_a:w_a + w_b] = image_b_corrected

    plt.figure(figsize=(14,7))
    plt.imshow(canvas)
    plt.axis("off")

    for match in matches:
        x_a, y_a = match["pt_a"]
        x_b, y_b = match["pt_b"]

        x_b_shifted = x_b + w_a

        plt.plot([x_a, x_b_shifted], [y_a,y_b], linewidth=0.8)
        plt.scatter([x_a, x_b_shifted], [y_a, y_b], s=8)
    
    plt.show()

"""
Plot matches for all image pairs
Input:
    images: list of image dictionaries
    pair_matches: dictionary of image pairs to list of matches
Output:
    None
"""
def plot_all_matches(images, pair_matches):
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            plot_matches(images[i], images[j], pair_matches[(i, j)])
