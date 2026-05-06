from load_images import load_images
from plot_images import plot_images
from corner_detection import detect_corners
from corner_detection import plot_corners
from corner_detection import plot_cornerness
from corner_detection import anms

import sys

def main():
    dataset = sys.argv[1]
    dataset = "../../datasets/" + dataset
    images = load_images(dataset)
    plot_images(images)
    detect_corners(images)
    plot_cornerness(images)
    plot_corners(images, True)
    anms(images, 500)
    plot_corners(images, False)


if __name__ == "__main__":
    main()