import cv2
import matplotlib.pyplot as plt

"""
Display the final panorama output
Input:
    panorama_canvas: the final output
Output:
    None
"""
def plot_panorama(panorama_canvas, title="Final Panorama"):
    
    image = cv2.cvtColor(panorama_canvas, cv2.COLOR_BGR2RGB)

    h, w = panorama_canvas.shape[:2]

    plt.figure(figsize=(16, 9))
    plt.imshow(image)
    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def save_panorama(panorama, filename="final_panorama.png"):
    cv2.imwrite(filename, panorama)
    print(f"Saved panorama to: {filename}")