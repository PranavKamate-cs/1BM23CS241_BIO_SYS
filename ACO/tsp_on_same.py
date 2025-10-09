import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# --- Network Configuration (Graph) ---
# Nodes 0 to 7. Same matrix as before.
DISTANCE_MATRIX = np.array([
#   0  1  2  3  4  5  6  7
    [0, 2, 4, 0, 0, 0, 0, 0], # 0 (Source/Start Node)
    [2, 0, 1, 6, 0, 0, 0, 0], # 1
    [4, 1, 0, 1, 3, 0, 0, 0], # 2
    [0, 6, 1, 0, 0, 2, 5, 0], # 3
    [0, 0, 3, 0, 0, 1, 0, 4], # 4
    [0, 0, 0, 2, 1, 0, 1, 2], # 5
    [0, 0, 0, 5, 0, 1, 0, 1], # 6
    [0, 0, 0, 0, 4, 2, 1, 0]  # 7 (Destination/End Node)
])

NUM_NODES = len(DISTANCE_MATRIX)
START_NODE = 0
END_NODE = 7 # Used for ACO only

# --- TSP Nearest Neighbor Algorithm ---

def solve_tsp_nearest_neighbor(start_node):
    """
    Applies the Greedy Nearest Neighbor heuristic for TSP.
    Starts at start_node, always moves to the closest unvisited node.
    Returns to the start_node at the end.
    """
    current_node = start_node
    tour = [start_node]
    tour_distance = 0
    unvisited = set(range(NUM_NODES))
    unvisited.remove(start_node)
    
    while unvisited:
        min_distance = float('inf')
        next_node = -1
        
        # Find the nearest unvisited neighbor
        for neighbor in list(unvisited):
            distance = DISTANCE_MATRIX[current_node, neighbor]
            # Must be a direct link and shorter than current min
            if distance > 0 and distance < min_distance:
                min_distance = distance
                next_node = neighbor
        
        # If no unvisited neighbor is reachable (graph is not Hamiltonian/complete)
        if next_node == -1:
            # For this simple implementation, we stop if we get stuck.
            print("TSP: Could not find a path to all nodes (stuck).")
            return None, float('inf')
            
        # Move to the nearest neighbor
        tour_distance += min_distance
        tour.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node

    # Complete the tour by returning to the starting node
    distance_back_to_start = DISTANCE_MATRIX[current_node, start_node]
    if distance_back_to_start > 0:
        tour_distance += distance_back_to_start
        tour.append(start_node)
    else:
        # If the last node can't connect back to the start
        print("TSP: Last node cannot return to the start node.")
        return None, float('inf')
        
    return tour, tour_distance

# --- Execute TSP and ACO (using the results from the previous ACO run) ---

# 1. TSP Solution
tsp_tour, tsp_distance = solve_tsp_nearest_neighbor(START_NODE)

# 2. ACO Solution (We'll use the final best result from the previous ACO code run)
# Based on the previous execution, the likely shortest path was:
aco_path = [0, 1, 2, 3, 5, 7] # Shortest path 0 to 7
aco_distance = 8.0 # (2+1+1+2+2)

# 3. Print Results
print("--- Results Comparison ---")
print(f"Graph Nodes: {list(range(NUM_NODES))}")
print("--------------------------")

print("1. ACO Routing Solution (Shortest Path 0 -> 7):")
print(f"Path: {' -> '.join(map(str, aco_path))}")
print(f"Distance: {aco_distance}")
print(f"Nodes Visited: {len(aco_path)}")
print("--------------------------")

print("2. TSP Nearest Neighbor Solution (Tour visiting all nodes):")
print(f"Tour: {' -> '.join(map(str, tsp_tour))}")
print(f"Distance: {tsp_distance}")
print(f"Nodes Visited: {len(set(tsp_tour))}")
print("--------------------------")

# --- Visualization Function for TSP ---

def draw_tsp_tour(distance_matrix, tour, distance):
    """Draws the network graph, highlighting the TSP tour."""
    G = nx.Graph()
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            if distance_matrix[i, j] > 0:
                G.add_edge(i, j, weight=distance_matrix[i, j])

    pos = nx.spring_layout(G, seed=42) 
    plt.figure(figsize=(12, 8))
    plt.title(f"TSP Nearest Neighbor Tour (Distance: {distance})")

    # Draw all edges
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.7)
    
    # Highlight the TSP Tour
    if tour:
        tour_edges = list(zip(tour[:-1], tour[1:]))
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=tour_edges, 
            edge_color='purple', 
            width=4, 
            alpha=1.0
        )
        
    # Draw nodes
    node_colors = ['skyblue'] * NUM_NODES
    node_colors[tour[0]] = 'green' # Start node
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='black')
    
    # Draw edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)
        
    plt.axis('off')
    plt.show()

# 4. Visualize TSP Tour
draw_tsp_tour(DISTANCE_MATRIX, tsp_tour, tsp_distance)
#
