from pathlib import Path
import sys

from load_images import load_images
from plot_images import plot_images

from pyramid import (
    build_pyramid,
    plot_pyramid,
)

from corner_detection import (
    detect_corners,
    plot_corners,
    plot_cornerness,
    anms,
    initialize_keypoints,
)

from feature_description import (
    estimate_keypoint_orientations,
    compute_mops_descriptors,
)

from feature_matching import (
    match_all_image_pairs,
    flatten_keypoints,
    plot_all_matches
)

from ransac import (
    estimate_all_pairwise_homographies, 
    plot_all_inlier_matches
)

from graph import (
    build_image_graph,
    choose_reference,
    dijkstra, 
    compute_transforms_to_reference
)

from warp_and_blend import (
    compute_canvas_bounds,
    build_offset,
    apply_offset,
    build_canvas,
    warp_all_images,
    blend,
    build_weights
)

from handle_output import (
    plot_panorama,
    save_panorama
)

def main():
    dataset_name = sys.argv[1]

    dataset_path = Path("../../datasets") / dataset_name

    images = load_images(dataset_path)
    plot_images(images)
    build_pyramid(images)
    # plot_pyramid(images)
    detect_corners(images)
    # plot_cornerness(images)
    # plot_corners(images, True)
    anms(images, 500)
    # plot_corners(images, False)
    initialize_keypoints(images)
    estimate_keypoint_orientations(images)
    compute_mops_descriptors(images)
    flatten_keypoints(images)
    pair_matches = match_all_image_pairs(images)
    # print(pair_matches)
    # plot_all_matches(images, pair_matches)
    pair_models = estimate_all_pairwise_homographies(pair_matches)
    # print(pair_models)
    # plot_all_inlier_matches(images, pair_models)
    image_graph = build_image_graph(pair_models)
    reference_node = choose_reference(image_graph)
    # print(reference_node)
    parents = dijkstra(image_graph, reference_node)
    # print(parents)
    transforms = compute_transforms_to_reference(image_graph, parents)
    # print(transforms)
    bounds = compute_canvas_bounds(images, transforms)
    offset = build_offset(bounds[0], bounds[2])
    transforms = apply_offset(transforms, offset)
    output = build_canvas(bounds[1] - bounds[0], bounds[3] - bounds[2])
    weights = build_weights(bounds[1] - bounds[0], bounds[3] - bounds[2])
    output = warp_all_images(images, transforms, output, weights)
    output = blend(output, weights)
    plot_panorama(output)
    save_panorama(output)


if __name__ == "__main__":
    main()