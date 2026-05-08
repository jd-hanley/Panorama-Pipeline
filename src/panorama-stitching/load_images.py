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
            "level_0": gray,
            "level_1": None,
            "level_2": None,
            "level_3": None,
            "level_4": None,
            # Corner response maps by level
            "level_0_cmap": None,
            "level_1_cmap": None,
            "level_2_cmap": None,
            "level_3_cmap": None,
            "level_4_cmap": None,
            # Corner coordinates by level
            "level_0_corners": None,
            "level_1_corners": None,
            "level_2_corners": None,
            "level_3_corners": None,
            "level_4_corners": None,
            # Corner scores by level
            "level_0_corner_scores": None,
            "level_1_corner_scores": None,
            "level_2_corner_scores": None,
            "level_3_corner_scores": None,
            "level_4_corner_scores": None,
            # Best corners after ANMS by level
            "level_0_best_corners": None,
            "level_1_best_corners": None,
            "level_2_best_corners": None,
            "level_3_best_corners": None,
            "level_4_best_corners": None,
            "keypoints": None,
            "shape": color.shape[:2],
            "sorted_corners": None
        })

    return images