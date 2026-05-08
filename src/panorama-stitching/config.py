
# Gaussian Pyramid Construction
NUM_LEVELS = 5

# MOPS Feature Descriptor Implementation
OUTER_PATCH_SIZE = 60
INNER_PATCH_SIZE = 40
DESCRIPTOR_SIZE = 8

# ANMS
ANMS_FEATURES_PER_LEVEL = 250

# Corner Detection
HARRIS_THRESHOLD_FACTOR = 0.005 
HARRIS_BLOCK_SIZE = 5               # Size of the neighborhood for corner detection
HARRIS_K_SIZE = 3                   # Kernel size of the sobel operator used
HARRIS_FREE_PARAMETER = 0.05        # Harris Detector free parameter

# Feature Matching
RATIO_THRESHOLD = 0.6               # Ratio to tune what is considered a match for features

# MISC
BLUR_KERNEL = 5