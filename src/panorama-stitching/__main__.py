from load_images import load_images
from plot_images import plot_images
from corner_detection import detect_corners
from corner_detection import plot_corners

def main():
    images = load_images("../../datasets/victoria_library")
    plot_images(images)
    plot_corners(images)

if __name__ == "__main__":
    main()