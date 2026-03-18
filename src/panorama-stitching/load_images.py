from pathlib import Path
import cv2

"""
Load images from a specified folder
Input: 
    folder (string): path to folder containing image dataset
Output: 
    images (list of dictionaries): dictionaries containing images and relevant information"""
def load_images(folder):

    # Valid file extensions: jpg, jpeg, png
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in exts])

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
            "gray": gray,
            "shape": color.shape[:2],
        })

    return images