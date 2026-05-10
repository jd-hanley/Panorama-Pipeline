from pathlib import Path
import argparse

from panorama_pipeline.io import load_images
from panorama_pipeline.pyramid import build_pyramid
from panorama_pipeline.corners import detect_corners, anms, initialize_keypoints
from panorama_pipeline.descriptors import estimate_keypoint_orientations, compute_mops_descriptors
from panorama_pipeline.matching import flatten_keypoints, match_all_image_pairs
from panorama_pipeline.ransac import estimate_all_pairwise_homographies
from panorama_pipeline.graph import build_image_graph, choose_reference, dijkstra, compute_transforms_to_reference
from panorama_pipeline.warp import (
    compute_canvas_bounds,
    build_offset,
    apply_offset,
    build_canvas,
    build_weights,
    warp_all_images,
    blend,
)
from panorama_pipeline.visualization import (
    plot_images,
    plot_pyramid,
    plot_cornerness,
    plot_corners,
    plot_matches,
    plot_all_matches,
    plot_panorama,
)
from panorama_pipeline.io import save_panorama

def parse_args():
    parser = argparse.ArgumentParser(
        description="Panorama stitching pipeline using Harris corners, MOPS descriptors, RANSAC, and inverse warping."
    )

    parser.add_argument("dataset", help="Name of dataset folder inside /datasets")

    parser.add_argument("--show-images", action="store_true")
    parser.add_argument("--show-pyramid", action="store_true")
    parser.add_argument("--show-cornerness", action="store_true")
    parser.add_argument("--show-raw-corners", action="store_true")
    parser.add_argument("--show-anms-corners", action="store_true")
    parser.add_argument("--show-matches", action="store_true")
    parser.add_argument("--show-inliers", action="store_true")
    parser.add_argument("--show-final", action="store_true")

    parser.add_argument("--save-final", action="store_true")
    parser.add_argument("--output", default="final_panorama.png")

    return parser.parse_args()

def main():
    args = parse_args()

    dataset_path = Path("datasets") / args.dataset

    print("[ 1 / 10 ] Loading images")
    images = load_images(dataset_path)

    if args.show_images:
        plot_images(images)
    
    print("[ 2 / 10 ] Building Gaussian Pyramids")
    build_pyramid(images)

    if args.show_pyramid:
        plot_pyramid(images)
    
    print("[ 3 / 10 ] Detecting Harris Corners")
    detect_corners(images)

    if args.show_raw_corners:
        plot_cornerness(images)
    
    if args.show_raw_corners:
        plot_corners(images, raw=True)
    
    print("[ 4 / 10 ] Running Adaptive Non-Maximal Supression")
    anms(images)

    if args.show_anms_corners:
        plot_corners(images, raw=False)

    print("[ 5 / 10 ] Computing MOPS Descriptors")
    initialize_keypoints(images)
    estimate_keypoint_orientations(images)
    compute_mops_descriptors(images)
    flatten_keypoints(images)

    print("[ 6 / 10 ] Matching descriptors")
    pair_matches = match_all_image_pairs(images)

    if args.show_matches:
        plot_all_matches(images, pair_matches)
    
    print("[ 7 / 10 ] Estimating pair-wise perspective transforms with RANSAC")
    pair_models = estimate_all_pairwise_homographies(pair_matches)

    if args.show_inliers:
        plot_all_matches(images, pair_matches, False)
    
    print("[ 8 / 10 ] Building image graph and global transforms")
    image_graph = build_image_graph(pair_models)
    reference_node = choose_reference(image_graph)
    parents = dijkstra(image_graph, reference_node)
    transforms = compute_transforms_to_reference(image_graph, parents)

    print("[ 9 / 10 ] Warping images into a common frame")
    min_x, max_x, min_y, max_y = compute_canvas_bounds(images, transforms)
    offset = build_offset(min_x, min_y)
    transforms = apply_offset(transforms, offset)

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    output = build_canvas(canvas_w, canvas_h)
    weights = build_weights(canvas_w, canvas_h)
    output = warp_all_images(images, transforms, output, weights)

    print("[ 10 / 10 ] Blending panorama")
    output = blend(output, weights)

    if args.show_final:
        plot_panorama(output)

    if args.save_final:
        save_panorama(output, args.output)

    print("Pipeline complete.")












    # dataset_name = sys.argv[1]

    # dataset_path = Path("../../datasets") / dataset_name

    # images = load_images(dataset_path)
    # plot_images(images)
    # build_pyramid(images)
    # # plot_pyramid(images)
    # detect_corners(images)
    # # plot_cornerness(images)
    # # plot_corners(images, True)
    # anms(images, 500)
    # # plot_corners(images, False)
    # initialize_keypoints(images)
    # estimate_keypoint_orientations(images)
    # compute_mops_descriptors(images)
    # flatten_keypoints(images)
    # pair_matches = match_all_image_pairs(images)
    # # print(pair_matches)
    # # plot_all_matches(images, pair_matches)
    # pair_models = estimate_all_pairwise_homographies(pair_matches)
    # # print(pair_models)
    # # plot_all_inlier_matches(images, pair_models)
    # image_graph = build_image_graph(pair_models)
    # reference_node = choose_reference(image_graph)
    # # print(reference_node)
    # parents = dijkstra(image_graph, reference_node)
    # # print(parents)
    # transforms = compute_transforms_to_reference(image_graph, parents)
    # # print(transforms)
    # bounds = compute_canvas_bounds(images, transforms)
    # offset = build_offset(bounds[0], bounds[2])
    # transforms = apply_offset(transforms, offset)
    # output = build_canvas(bounds[1] - bounds[0], bounds[3] - bounds[2])
    # weights = build_weights(bounds[1] - bounds[0], bounds[3] - bounds[2])
    # output = warp_all_images(images, transforms, output, weights)
    # output = blend(output, weights)
    # plot_panorama(output)
    # save_panorama(output)


if __name__ == "__main__":
    main()