import cv2
import matplotlib.pyplot as plt

def show_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def plot_images(images):
    for image in images:
        show_image(image)
