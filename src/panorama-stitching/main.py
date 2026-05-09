from pathlib import Path
import sys

from load_images import load_images
from plot_images import plot_images

from pyramid import (
    build_pyramid,
    plot_pyramid,
)

from corner_detection import (
    detect_corners,
    plot_corners,
    plot_cornerness,
    anms,
    initialize_keypoints,
)

from feature_description import (
    estimate_keypoint_orientations,
    compute_mops_descriptors,
)

from feature_matching import (
    match_all_image_pairs,
    flatten_keypoints,
    plot_all_matches
)

from ransac import estimate_all_pairwise_homographies, plot_all_inlier_matches

def main():
    dataset_name = sys.argv[1]

    dataset_path = Path("../../datasets") / dataset_name

    images = load_images(dataset_path)
    plot_images(images)
    build_pyramid(images)
    plot_pyramid(images)
    detect_corners(images)
    plot_cornerness(images)
    plot_corners(images, True)
    anms(images, 500)
    plot_corners(images, False)
    initialize_keypoints(images)
    estimate_keypoint_orientations(images)
    compute_mops_descriptors(images)
    flatten_keypoints(images)
    pair_matches = match_all_image_pairs(images)
    plot_all_matches(images, pair_matches)
    pair_models = estimate_all_pairwise_homographies(pair_matches)
    plot_all_inlier_matches(images, pair_models)


if __name__ == "__main__":
    main()