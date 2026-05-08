from load_images import load_images
from plot_images import plot_images
from corner_detection import detect_corners
from corner_detection import plot_corners
from corner_detection import plot_cornerness
from corner_detection import anms
from pyramid import build_pyramid
from pyramid import plot_pyramid
from corner_detection import initialize_keypoints
from feature_description import estimate_keypoint_orientations
from feature_description import compute_mops_descriptors

import sys

def main():
    dataset = sys.argv[1]
    dataset = "../../datasets/" + dataset
    images = load_images(dataset)
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
    # plot_corners(images, True)
    # anms(images, 500)
    # plot_corners(images, False)


if __name__ == "__main__":
    main()