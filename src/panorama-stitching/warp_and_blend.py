import numpy as np
import math

from config import FEATHER_WIDTH

"""
Use the transforms and images to determine the bounds of the final canvas
Input:
    images: list of image dictionaries
    transforms: dictionary mapping images to the transform to the reference image
Output:
    min_x
    max_x
    min_y
    max_y
"""
def compute_canvas_bounds(images, transforms):
    xs = []
    ys = []

    for i, image in enumerate(images):
        h, w = image["color"].shape[:2]

        corners = [np.array([0,0,1]), np.array([w-1, 0, 1]), np.array([0, h-1, 1]), np.array([w-1, h-1, 1])]

        T = transforms[i]

        for corner in corners:
            # Compute the coordinate in the base frame
            p = T @ corner

            if abs(p[2]) < 1e-12:
                continue

            p /= p[2]

            x,y = p[0], p[1]

            xs.append(x)
            ys.append(y)
    
    min_x = math.floor(min(xs))
    max_x = math.ceil(max(xs))
    min_y = math.floor(min(ys))
    max_y = math.ceil(max(ys))

    return min_x, max_x, min_y, max_y

"""
Use the mininum coordinates in the x and y directions to compose the translational offset
Input:
    min_x: minimum x coordinate
    min_y: minimum y coordinate
Output:
    offset: 3 x 3 translational offset matrix
"""
def build_offset(min_x, min_y):
    
    return np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0,    1  ]
    ])

"""
With the offset matrix computed, iterate through all images and apply the offset matrix to the homographies computed
Input:
    transforms: dictionary mapping images to their perspective transforms to the base image
    offset: translational offset matrix based on the output canvas
Output:
    transforms: modified dictionary mapping images to their perspective transforms now including the translational offset
"""
def apply_offset(transforms, offset):
    
    for index, transform in transforms.items():
        transforms[index] = offset @ transform
    
    return transforms

"""
Use the computed bounds to allocate the panorama canvas
Input:
    x_range: size of the canvas in the x direction
    y_range: size of the canvas in the y direction
Output:
    panorama_canvas: appropriately sized canvas to contain the full panorama
"""
def build_canvas(x_range, y_range):
    
    return np.zeros((y_range, x_range, 3), dtype=np.float32)

""" 
Build the weight matrix for simple blending
Input: 
    x_range: size of the canvas in the x direction
    y_range: size of the canvas in the y direction
Output:
    weights: empty but correctly sized matrix to stored weights
"""
def build_weights(x_range, y_range):

    return np.zeros((y_range, x_range), dtype=np.float32)

"""
Take in a single image and the output canvas, iterate over the entire canvas and perform inverse warping
Input:
    image: single image dictionary
    transform: perspective transform matrix with translational offset included
    panorama_canvas: output image 
Output:
    panorama_canvas: panorama canvas with pixels filled in
"""
def inverse_warp(image, transform, panorama_canvas, weights):

    p_h, p_w = panorama_canvas.shape[:2]
    i_h, i_w = image["color"].shape[:2]

    # Compute the inverse transformation matrix for this particular image
    H_inv = np.linalg.inv(transform)

    for y in range(p_h):
        for x in range(p_w):

            # Build up the homogeneous coordinate
            pt = np.array([x, y, 1.0])

            # Apply the inverse transform 
            proj_pt = H_inv @ pt

            # Normalize to cartesian
            if abs(proj_pt[2]) < 1e-12:
                continue
            proj_pt /= proj_pt[2]

            row, col = proj_pt[1], proj_pt[0]

            # Determine if we are within the bounds of the image
            if row < 0 or col < 0 or row >= i_h - 1 or col >= i_w - 1:
                continue

            bgr = bilinear_interpolation(row, col, image)

            # Implement feather blending
            # Determine how close the pixel is to an edge and weight accordingly
            dist_left = col
            dist_top = row
            dist_right = i_w - 1 - col
            dist_bottom = i_h - 1 - row

            border_dist = min(dist_left, dist_right, dist_top, dist_bottom)

            feather = border_dist / FEATHER_WIDTH
            feather = np.clip(feather, 0.0, 1.0)

            panorama_canvas[y, x] += bgr * feather
            weights[y, x] += feather
    
    return panorama_canvas


"""
Warp all images into the output canvas
Input:
    images: list of image dictionaries
    transforms: dictionary of transforms from each image to output canvas, including translational offset
    panorama_canvas: output image
Output:
    panorama_canvas: output image with all images warped in
"""
def warp_all_images(images, transforms, panorama_canvas, weights):

    for i, image in enumerate(images):
        panorama_canvas = inverse_warp(image, transforms[i], panorama_canvas, weights)
    
    return panorama_canvas

"""
Perform bilinear interpolation to determine the B,G,R values for a non-discrete pixel location
Input:
    image: image dictionary
    row: image row coordinate
    col: image pixel coordinate
Output:
    b: blue channel value
    g: green channel value
    r: red channel value
"""
def bilinear_interpolation(row, col, image):

    color = image["color"]

    # Determine the relevant surrounding pixels
    row_0 = int(np.floor(row))
    row_1 = row_0 + 1
    col_0 = int(np.floor(col))
    col_1 = col_0 + 1

    # Compute the weights
    top_left = (row_1 - row) * (col_1 - col)
    bottom_right = (row - row_0) * (col - col_0)
    bottom_left = (row - row_0) * (col_1 - col)
    top_right = (row_1 - row) * (col - col_0)

    # Compute the resulting color
    weighted_sum = top_left * color[row_0, col_0] + bottom_right * color[row_1, col_1] + bottom_left * color[row_1,col_0] + top_right * color[row_0, col_1]

    return weighted_sum

"""
Use simple blending to blend the result 
Input:
    panorama_canvas: output with all images warped in
    weights: tracks how many images contributed to a single pixel
Output:
    panorama_canvas: final blended output
"""
def blend(panorama_canvas, weights):

    valid = weights > 0
    panorama_canvas[valid] = panorama_canvas[valid] / weights[valid, None]

    return np.clip(panorama_canvas, 0, 255).astype(np.uint8)
