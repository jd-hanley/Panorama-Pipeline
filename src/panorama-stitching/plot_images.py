import cv2
import matplotlib.pyplot as plt

"""
Display an image using matplotlib
Input:
    image (np.ndarray): Grayscale or BGR image.
Output:
    None
"""
def show_image(image):
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

"""
Display all images in a list using matplotlib
Input:
    images (list of image dictionaries)
Output:
    None"""
def plot_images(images):
    for image in images:
        show_image(image["color"])
