import numpy as np

"""
Perform feature matching between the feature descriptors for two images
Input:
    desc_1 (dict of (row, col) tuple : (np.ndarray)): correspondence of coordinate to 64 x 1 feature descriptor for first image
    desc_2 (dict of (row, col) tuple : (np.ndarray)): correspondence of coordinate to 64 x 1 feature descriptor for second image
Output:
    matches: feature correspondences between images """
def match_features(desc_1, desc_2):
    """
    Overview of the algorithm:
        1) Compute the sum of square differences from one descriptor to every descriptor in the second image
        2) Compute the ratio from the best match to the second best match
        3) If the ratio is below a threshold, add the match the result """
    
    # Define the ratio threshold
    threshold = 0.6

    matches = []

    for (row_1, col_1), vec_1 in desc_1.items():

        # Maintain a record of the SSD
        ssd = []

        # Iterate over every descriptor in the second image
        for (row_2, col_2), vec_2 in desc_2.items():

            diff = vec_1 - vec_2
            score = np.sum(diff**2)

            # Store both the coordinate and the ssd score
            ssd.append({
                "coordinate": (row_2, col_2),
                "score": score
            })
        
        # Sort the list by the ssd score
        ssd.sort(key=lambda x: x['ssd'])

        if len(ssd) > 2:

            # Best match has the lowest ssd 
            best_match = ssd[0]
            runner_up = ssd[1]

            # If the ratio between the best and second best is below threshold, then this is a good match
            ratio = best_match["score"] / runner_up["score"]

            if ratio < threshold:
                matches.append([(row_1, col_1), best_match("coordinate")])


    return matches

def show_feature_matches():
    pass
