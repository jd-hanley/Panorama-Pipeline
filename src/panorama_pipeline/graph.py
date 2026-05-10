import numpy as np
import heapq

"""
From the pairwise models, build a weighted graph connecting images
Input:
    pair_models: dictionary of image pairs (i, j) to model information (H matrix, inliers, inlier count)
Output:
    image_graph: dictionary representing a adjacency list representation of the weighted image graph for Dijkstra's
"""
def build_image_graph(pair_models):

    graph = {}

    for pair, model in pair_models.items():
        src = pair[0]
        dst = pair[1]

        if src not in graph:
            graph[src] = []
        if dst not in graph:
            graph[dst] = []
        
        graph[src].append({
            "neighbor": dst, 
            "H": model["H"],
            "cost": 1.0 / model["num_inliers"]
        })

        graph[dst].append({
            "neighbor": src, 
            "H": np.linalg.inv(model["H"]),
            "cost": 1.0 / model["num_inliers"]
        })
    
    return graph

"""
From the image graph, determine which image will serve as the reference image
The reference image will be the base frame into which all other images are projected
Input:
    image_graph: dictionary representing a adjacency list representation of the weighted image graph for Dijkstra's
Output:
    best_node: the entry in the graph with the maximum total inliers across all connections
"""
def choose_reference(image_graph):
    best_node = None
    best_score = -1

    for node, neighbors in image_graph.items():
        score = 0
        for neighbor in neighbors:
            score += (1.0 / neighbor["cost"])
        
        if score > best_score:
            best_node = node
            best_score = score
    
    return best_node

"""
Implement Dijkstra's algorithm to get the shortest path from every node to the reference node
Input:
    image_graph: adjacency list representation of the images
    source: the reference node that was previously determined
Output:
    parents: dictionary containing the parent of each node, to be used to reconstruct paths
"""
def dijkstra(image_graph, source):
    distances = {node: float("inf") for node in image_graph}
    parents = {node: None for node in image_graph}

    # Start at the source node
    distances[source] = 0.0

    pq = [(0.0, source)]

    while pq:
        # Dequeue to the front element
        curr_dist, curr_node = heapq.heappop(pq)

        # If the distance is greater than the node's current then continue (old entry)
        if curr_dist > distances[curr_node]:
            continue

        # Need to look through all neighbors of the current node and update their distances and parents
        for edge in image_graph[curr_node]:
            neighbor = edge["neighbor"]
            cost = edge["cost"]

            # Implement the relaxation rule
            if curr_dist + cost < distances[neighbor]:
                distances[neighbor] = curr_dist + cost
                parents[neighbor] = curr_node
                heapq.heappush(pq, (distances[neighbor], neighbor))

    return parents

"""
Using the graph representation and the parents output from Dijkstra's compute the homography from every image to the reference
Input:
    image_graph: adjacency list representation of the images
    parents: dictionary mapping each node to its parent in its shortest path to the reference
Output:
    final_homographies: dictionary mapping each image to its homography to the base image
"""
def compute_transforms_to_reference(image_graph, parents):

    final_homographies = {}

    for node in image_graph:

        T = np.eye(3)

        curr = node
        parent = parents[curr]

        while parent is not None:

            # Need to multiply by the homography from the current node to the parent
            for edge in image_graph[curr]:
                if edge["neighbor"] == parent:
                    H = edge["H"]
                    T = H @ T
                    break
            
            curr = parent
            parent = parents[parent]
        
        final_homographies[node] = T
    
    return final_homographies


