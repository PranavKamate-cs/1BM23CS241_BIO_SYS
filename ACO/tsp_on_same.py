import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- Network Configuration (Graph) ---
DISTANCE_MATRIX = np.array([
#   0  1  2  3  4  5  6  7
    [0, 2, 4, 0, 0, 0, 0, 0], # 0
    [2, 0, 1, 6, 0, 0, 0, 0], # 1
    [4, 1, 0, 1, 3, 0, 0, 0], # 2
    [0, 6, 1, 0, 0, 2, 5, 0], # 3
    [0, 0, 3, 0, 0, 1, 0, 4], # 4
    [0, 0, 0, 2, 1, 0, 1, 2], # 5
    [0, 0, 0, 5, 0, 1, 0, 1], # 6
    [0, 0, 0, 0, 4, 2, 1, 0]  # 7
])

NUM_NODES = len(DISTANCE_MATRIX)
START_NODE = 0

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
            if distance > 0 and distance < min_distance:
                min_distance = distance
                next_node = neighbor
        
        # If no unvisited neighbor is reachable
        if next_node == -1:
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
        print("TSP: Last node cannot return to the start node.")
        return None, float('inf')
        
    return tour, tour_distance

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
    node_colors[tour[0]] = 'green'  # Start node
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='black')
    
    # Draw edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)
        
    plt.axis('off')
    plt.show()

# --- Execute TSP ---
tsp_tour, tsp_distance = solve_tsp_nearest_neighbor(START_NODE)

# --- Print Results ---
print("--- TSP Nearest Neighbor Results ---")
print(f"Graph Nodes: {list(range(NUM_NODES))}")
print("--------------------------")

if tsp_tour is not None:
    print("TSP Nearest Neighbor Solution (Tour visiting all nodes):")
    print(f"Tour: {' -> '.join(map(str, tsp_tour))}")
    print(f"Distance: {tsp_distance}")
    print(f"Nodes Visited: {len(set(tsp_tour))}")
    print("--------------------------")
    draw_tsp_tour(DISTANCE_MATRIX, tsp_tour, tsp_distance)
else:
    print("No complete TSP tour found.")
    print("--------------------------")
