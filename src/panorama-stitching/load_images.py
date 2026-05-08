from pathlib import Path
import cv2

"""
Load images from a specified folder
Input: 
    folder (string): path to folder containing image dataset
Output: 
    images (list of dictionaries): dictionaries containing images and relevant information"""
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
            # Images
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

    return images