import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# --- More Complex Network Configuration (Graph) ---
# Nodes 0 to 7. Distances are non-zero for connected links.
DISTANCE_MATRIX = np.array([
#   0  1  2  3  4  5  6  7
    [0, 2, 4, 0, 0, 0, 0, 0], # 0 (Source)
    [2, 0, 1, 6, 0, 0, 0, 0], # 1
    [4, 1, 0, 1, 3, 0, 0, 0], # 2
    [0, 6, 1, 0, 0, 2, 5, 0], # 3
    [0, 0, 3, 0, 0, 1, 0, 4], # 4
    [0, 0, 0, 2, 1, 0, 1, 2], # 5
    [0, 0, 0, 5, 0, 1, 0, 1], # 6
    [0, 0, 0, 0, 4, 2, 1, 0]  # 7 (Destination)
])
# Note: The shortest path is likely 0 -> 1 -> 2 -> 3 -> 5 -> 7 (2+1+1+2+2 = 8)
# or 0 -> 2 -> 4 -> 5 -> 7 (4+3+1+2 = 10) or 0 -> 2 -> 3 -> 6 -> 7 (4+1+5+1 = 11)

NUM_NODES = len(DISTANCE_MATRIX)
START_NODE = 0
END_NODE = 7

# --- ACO Parameters ---
NUM_ANTS = 20         # Increased ants for faster exploration
NUM_ITERATIONS = 75   # Increased iterations to show clear convergence
EVAPORATION_RATE = 0.4  # Slightly lower evaporation
PHEROMONE_DEPOSIT = 1.0 
ALPHA = 1.0           
BETA = 3.0            # Increased BETA to make distance (heuristic) a stronger factor

# Initialize Pheromone Matrix
PHEROMONE_MATRIX = np.ones((NUM_NODES, NUM_NODES)) * 0.1

# --- Graph Visualization Setup ---

# Create a NetworkX graph object from the distance matrix
G = nx.Graph()
for i in range(NUM_NODES):
    for j in range(i + 1, NUM_NODES):
        if DISTANCE_MATRIX[i, j] > 0:
            G.add_edge(i, j, weight=DISTANCE_MATRIX[i, j])

# Set a fixed layout for consistent plotting
pos = nx.spring_layout(G, seed=42) 

def draw_network(G, pos, best_path, iteration, shortest_distance):
    """Draws the network graph, highlighting the best path and link pheromones."""
    plt.figure(figsize=(12, 8))
    plt.title(f"ACO Routing - Iteration {iteration} | Best Distance: {shortest_distance}")

    # 1. Update edge attributes for drawing
    # Avoid errors if all pheromones are zero by checking max_pheromone
    max_pheromone = np.max(PHEROMONE_MATRIX) if np.max(PHEROMONE_MATRIX) > 0 else 1.0 
    
    edge_weights = nx.get_edge_attributes(G, 'weight')
    edge_labels = {}
    edge_color_map = []
    edge_width_map = []
    
    for u, v in G.edges():
        pher = PHEROMONE_MATRIX[u, v]
        
        # Pheromone for label (rounded)
        edge_labels[(u, v)] = f"{edge_weights[(u, v)]} | P:{pher:.2f}"
        
        # Color: Map pheromone intensity (green for high)
        color_intensity = pher / max_pheromone
        edge_color_map.append((1 - color_intensity, color_intensity, 0)) # Red (low) to Green (high)
        
        # Width: Thicker lines for higher pheromone (min width 1, max 6)
        edge_width_map.append(1.5 + (pher / max_pheromone) * 4.5)

    # 2. Draw all edges with color/width based on Pheromone
    nx.draw_networkx_edges(
        G, pos, 
        edge_color=edge_color_map, 
        width=edge_width_map, 
        alpha=0.7
    )
    
    # 3. Draw nodes
    node_colors = ['#ADD8E6'] * NUM_NODES # Light blue default
    node_colors[START_NODE] = '#008000' # Green source
    node_colors[END_NODE] = '#FF0000'   # Red destination
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200)
    
    # 4. Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='black')
    
    # 5. Draw edge labels (Distance | Pheromone)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)
    
    # 6. Highlight the current Best Path
    if best_path:
        path_edges = list(zip(best_path, best_path[1:]))
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=path_edges, 
            edge_color='blue', 
            width=5, 
            alpha=1.0
        )
        
    plt.axis('off')
    plt.show()

# -------------------------------------------------------------
# --- Core ACO Functions (Rely on PHEROMONE_MATRIX and DISTANCE_MATRIX) ---
# -------------------------------------------------------------

def get_allowed_neighbors(current_node, visited_nodes):
    neighbors = []
    for neighbor in range(NUM_NODES):
        if DISTANCE_MATRIX[current_node, neighbor] > 0 and neighbor not in visited_nodes:
            neighbors.append(neighbor)
    return neighbors

def calculate_transition_probability(current_node, neighbors):
    probabilities = {}
    total_attractiveness = 0
    
    for next_node in neighbors:
        tau = PHEROMONE_MATRIX[current_node, next_node] ** ALPHA
        distance = DISTANCE_MATRIX[current_node, next_node]
        # Heuristic (eta): Inverse of distance, strongly favors short links
        eta = (1.0 / distance) ** BETA 

        attractiveness = tau * eta
        total_attractiveness += attractiveness
        probabilities[next_node] = attractiveness

    if total_attractiveness == 0:
        return {node: 1.0 / len(neighbors) for node in neighbors}
    else:
        return {node: prob / total_attractiveness for node, prob in probabilities.items()}

def run_ant_simulation():
    current_node = START_NODE
    path = [START_NODE]
    path_distance = 0
    pheromone_path = []

    while current_node != END_NODE:
        visited_nodes = set(path)
        neighbors = get_allowed_neighbors(current_node, visited_nodes)

        if not neighbors:
            # Trapped: failed path (simulates a dropped packet/link failure)
            return None, float('inf'), None

        probabilities = calculate_transition_probability(current_node, neighbors)
        next_node = random.choices(
            list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]

        pheromone_path.append((current_node, next_node))
        path_distance += DISTANCE_MATRIX[current_node, next_node]
        path.append(next_node)
        current_node = next_node

    return path, path_distance, pheromone_path

def update_pheromones(paths_found):
    global PHEROMONE_MATRIX
    
    # 1. Evaporation: Fades all old trails
    PHEROMONE_MATRIX = PHEROMONE_MATRIX * (1.0 - EVAPORATION_RATE)

    # 2. Deposit (Reinforcement): Strengthens successful trails
    for path_data in paths_found:
        path_distance = path_data['distance']
        pheromone_path = path_data['pheromone_path']
        
        if path_distance > 0:
            # The shorter the distance, the larger the deposit
            pheromone_delta = PHEROMONE_DEPOSIT / path_distance
            
            for i, j in pheromone_path:
                PHEROMONE_MATRIX[i, j] += pheromone_delta
                PHEROMONE_MATRIX[j, i] += pheromone_delta
                
# --- Main ACO Execution ---

best_path = None
shortest_distance = float('inf')
print("Starting ACO Simulation with a Complex Network (8 nodes)...")

# Visualize the initial state
draw_network(G, pos, best_path, 0, shortest_distance)
# 

for iteration in range(NUM_ITERATIONS):
    all_ant_results = []
    
    # Run all ants and gather results
    for ant in range(NUM_ANTS):
        path, distance, pheromone_path = run_ant_simulation()
        if path:
            all_ant_results.append({
                'path': path, 
                'distance': distance, 
                'pheromone_path': pheromone_path
            })
            
            # Update best path found so far
            if distance < shortest_distance:
                shortest_distance = distance
                best_path = path

    # Update pheromone trails
    update_pheromones(all_ant_results)
    
    # Visualize the graph every 25 iterations and at the end
    if (iteration + 1) % 25 == 0 or iteration == NUM_ITERATIONS - 1 or iteration == 0:
        print(f"Iteration {iteration + 1}: Shortest distance found so far = {shortest_distance} on path: {' -> '.join(map(str, best_path)) if best_path else 'None'}")
        draw_network(G, pos, best_path, iteration + 1, shortest_distance)


# --- Final Results ---
print("\n--- Simulation Complete ---")
print(f"Final Shortest Distance: {shortest_distance}")
print(f"Optimal Path: {' -> '.join(map(str, best_path))}")
