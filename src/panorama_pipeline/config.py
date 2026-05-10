
# Gaussian Pyramid Construction
NUM_LEVELS = 5

# MOPS Feature Descriptor Implementation
OUTER_PATCH_SIZE = 60
INNER_PATCH_SIZE = 40
DESCRIPTOR_SIZE = 8

# ANMS
ANMS_FEATURES_PER_LEVEL = 500
MAX_FILTER_WINDOW = 5

# Corner Detection
HARRIS_THRESHOLD_FACTOR = 0.05
HARRIS_BLOCK_SIZE = 5               # Size of the neighborhood for corner detection
HARRIS_K_SIZE = 3                   # Kernel size of the sobel operator used
HARRIS_FREE_PARAMETER = 0.05        # Harris Detector free parameter

# Feature Matching
RATIO_THRESHOLD = 0.4               # Ratio to tune what is considered a match for features

# RANSAC
RANSAC_ITERATIONS = 1000            # Number of iterations RANSAC will run by default
RANSAC_INLIER_COUNT = 200           # Number of inliers necessary for model to be considered good
RANSAC_THRESHOLD = 0.4              # Distance threshold for a point to be considered an inlier
RANSAC_MIN_PTS = 4                  # Minimum number of points needed for a model

# Blending
FEATHER_WIDTH = 100

# MISC
BLUR_KERNEL = 5                     # Used anywhere the library Gaussian blur function is called