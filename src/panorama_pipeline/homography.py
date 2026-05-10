import numpy as np

"""
Implement homography/perspective transform given pairs of matched points between images
Input:
    selected_pairs: list of image coordinate pair structures between two images
Output:
    H: 3 x 3 homography matrix from the first image to the second image
"""
def compute_homography(selected_pairs):
    # From earlier, we have dictionaries of matches, keep that in mind for the structure of the input
    intermediate_matrices = []
    # Construct the matrix from the homography equation at each point pair
    for pair in selected_pairs:
        curr = np.zeros((2,9))
        curr[0,:] = [pair["pt_a"][0],
                     pair["pt_a"][1],
                     1,
                     0,
                     0,
                     0,
                     -pair["pt_b"][0] * pair["pt_a"][0],
                     -pair["pt_b"][0] * pair["pt_a"][1],
                     -pair["pt_b"][0]]
        curr[1,:] = [0,
                     0,
                     0,
                     pair["pt_a"][0],
                     pair["pt_a"][1],
                     1,
                     -pair["pt_b"][1] * pair["pt_a"][0],
                     -pair["pt_b"][1] * pair["pt_a"][1],
                     -pair["pt_b"][1]]
        intermediate_matrices.append(curr)
    
        # Build the full matrix
        a = np.vstack(intermediate_matrices)

        # Calculate homography via constrained least squares
        _, _, Vt = np.linalg.svd(a)

        h = Vt[-1, :]
        H = h.reshape(3, 3)

        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

    return H

