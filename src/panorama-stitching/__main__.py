from load_images import load_images
from plot_images import plot_images
from plot_images import show_image

def main():
    images = load_images("images")
    plot_images(images)

if __name__ == "__main__":
    main()