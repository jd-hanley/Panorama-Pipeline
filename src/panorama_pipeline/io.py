from pathlib import Path
import cv2

"""
Load images from a specified folder
Input: 
    path (string): path to folder containing image dataset
Output: 
    images (list of dictionaries): dictionaries containing images and relevant information for the pipeline
"""
def load_images(path: str):

    # Valid file extensions: jpg, jpeg, png
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in Path(path).iterdir() if p.suffix.lower() in exts])

    images = []
    for i, path in enumerate(paths):
        color = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError(f"Failed to load image: {path}")

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        images.append({
            "index": i,
            "path": str(path),
            "name": path.name,
            "color": color,
            # List of grayscale images at all levels
            "gray": [gray],
            # Corner response maps by level
            "cmaps": [],
            # Corner coordinates by level
            "corners": [],
            # Corner scores by level
            "corner_scores": [],
            # Best corners after ANMS by level
            "best_corners": [],
            # Keypoints contains info including the location and feature descriptor
            "keypoints": [],
            "shape": color.shape[:2],
            "sorted_corners": None
        })

    print(f"           loaded {len(images)} images")
    return images

def save_panorama(panorama, filename="final_panorama.png"):
    cv2.imwrite(filename, panorama)
    print(f"Saved panorama to: {filename}")