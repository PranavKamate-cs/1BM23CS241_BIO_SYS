import numpy as np
import matplotlib.pyplot as plt
import heapq

# ==============================
# Parameters & Environment
# ==============================
GRID_SIZE = 20
START = np.array([0, 0])    # [row, col]
GOAL  = np.array([19, 19])  # [row, col]
OBSTACLE_DENSITY = 0.2  # 20% obstacles

np.random.seed(0)
obstacle_map = (np.random.rand(GRID_SIZE, GRID_SIZE) < OBSTACLE_DENSITY).astype(int)
obstacle_map[START[0], START[1]] = 0
obstacle_map[GOAL[0], GOAL[1]] = 0

# ==============================
# Helpers: validity, repair, collisions
# ==============================
def is_valid(point):
    r, c = int(round(point[0])), int(round(point[1]))
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and obstacle_map[r, c] == 0

def find_nearest_free_cell(r, c, max_radius=6):
    r0, c0 = int(round(r)), int(round(c))
    if 0 <= r0 < GRID_SIZE and 0 <= c0 < GRID_SIZE and obstacle_map[r0, c0] == 0:
        return r0, c0
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and obstacle_map[nr, nc] == 0:
                    return nr, nc
    # fallback: return START or any free cell
    for nr in range(GRID_SIZE):
        for nc in range(GRID_SIZE):
            if obstacle_map[nr, nc] == 0:
                return nr, nc
    return r0, c0

def repair_path(path):
    repaired = path.copy()
    for i in range(len(repaired)):
        repaired[i] = np.clip(repaired[i], 0, GRID_SIZE - 1)
        r, c = repaired[i]
        if obstacle_map[int(round(r)), int(round(c))] == 1:
            nr, nc = find_nearest_free_cell(r, c)
            repaired[i] = np.array([nr, nc], dtype=float)
    repaired[0] = START.astype(float)
    repaired[-1] = GOAL.astype(float)
    return repaired

def segment_collision(p1, p2, samples=12):
    for t in np.linspace(0, 1, samples):
        pt = p1 * (1 - t) + p2 * t
        if not is_valid(pt):
            return True
    return False

def fitness(path):
    length = 0.0
    collisions = 0
    for i in range(len(path) - 1):
        length += np.linalg.norm(path[i + 1] - path[i])
        if segment_collision(path[i], path[i + 1], samples=12):
            collisions += 1
    # stronger penalty so optimizer avoids collisions
    return length + 100000.0 * collisions

# ==============================
# Initialize population (same idea as before)
# ==============================
def initialize_population(num_wolves, num_points):
    population = []
    for _ in range(num_wolves):
        xs = np.linspace(START[0], GOAL[0], num_points) + np.random.uniform(-1.0, 1.0, num_points)
        ys = np.linspace(START[1], GOAL[1], num_points) + np.random.uniform(-1.0, 1.0, num_points)
        path = np.stack([xs, ys], axis=1).astype(float)
        path[0] = START.astype(float)
        path[-1] = GOAL.astype(float)
        path = np.clip(path, 0, GRID_SIZE - 1)
        path = repair_path(path)
        population.append(path)
    return np.array(population)

# ==============================
# Grey Wolf Optimizer (as before)
# ==============================
def gwo_path_planning(num_wolves=30, num_points=20, max_iter=120):
    wolves = initialize_population(num_wolves, num_points)
    fitness_values = np.array([fitness(w) for w in wolves])
    sorted_idx = np.argsort(fitness_values)
    wolves = wolves[sorted_idx]
    fitness_values = fitness_values[sorted_idx]
    alpha, beta, delta = wolves[0].copy(), wolves[1].copy(), wolves[2].copy()
    best_scores = []

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        for i in range(len(wolves)):
            X = wolves[i].copy()
            for j in range(1, num_points - 1):
                # alpha
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha[j] - X[j])
                X1 = alpha[j] - A1 * D_alpha
                # beta
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta[j] - X[j])
                X2 = beta[j] - A2 * D_beta
                # delta
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta[j] - X[j])
                X3 = delta[j] - A3 * D_delta

                new_point = (X1 + X2 + X3) / 3.0
                X[j] = np.clip(new_point, 0, GRID_SIZE - 1)

            X = repair_path(X)
            wolves[i] = X
            fitness_values[i] = fitness(X)

        sorted_idx = np.argsort(fitness_values)
        wolves = wolves[sorted_idx]
        fitness_values = fitness_values[sorted_idx]
        alpha, beta, delta = wolves[0].copy(), wolves[1].copy(), wolves[2].copy()
        best_scores.append(fitness_values[0])
        if t % 10 == 0:
            print(f"Iter {t:03d}: Best fitness = {fitness_values[0]:.2f}")

    return alpha, best_scores

# ==============================
# A* on the grid (4-neighbors)
# ==============================
def astar(start_cell, goal_cell, grid):
    """start_cell and goal_cell are (r,c) integers. Returns list of cells from start to goal (inclusive) or None."""
    R, C = grid.shape
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start_cell, goal_cell), 0, start_cell, None))
    came_from = {}
    gscore = {start_cell: 0}
    closed = set()

    while open_heap:
        f, g, current, parent = heapq.heappop(open_heap)
        if current in closed:
            continue
        came_from[current] = parent
        if current == goal_cell:
            # reconstruct
            path = []
            cur = current
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1]
        closed.add(current)

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = current[0]+dr, current[1]+dc
            nb = (nr, nc)
            if 0 <= nr < R and 0 <= nc < C and grid[nr, nc] == 0:
                tentative_g = g + 1
                if nb in closed:
                    continue
                prev_g = gscore.get(nb, 1e9)
                if tentative_g < prev_g:
                    gscore[nb] = tentative_g
                    heapq.heappush(open_heap, (tentative_g + heuristic(nb, goal_cell), tentative_g, nb, current))
    return None

# ==============================
# Post-process best GWO path with A*: replace colliding segments
# ==============================
def postprocess_with_astar(best_path):
    """best_path: array of floats shape (N,2) [row,col].
       Returns discrete grid path (list of (r,c) ints) connected by A* where needed."""
    discrete_path = []
    for i in range(len(best_path)-1):
        a = np.round(best_path[i]).astype(int)
        b = np.round(best_path[i+1]).astype(int)
        if segment_collision(best_path[i], best_path[i+1], samples=20):
            astar_segment = astar(tuple(a), tuple(b), obstacle_map)
            if astar_segment is None:
                # fallback: include the endpoints at least
                if not discrete_path or discrete_path[-1] != tuple(a):
                    discrete_path.append(tuple(a))
                discrete_path.append(tuple(b))
            else:
                # append astar path but avoid duplicating the first cell if already in discrete_path
                if discrete_path and discrete_path[-1] == astar_segment[0]:
                    discrete_path.extend(astar_segment[1:])
                else:
                    discrete_path.extend(astar_segment)
        else:
            # straight segment is free: just ensure endpoints are present
            if not discrete_path or discrete_path[-1] != tuple(a):
                discrete_path.append(tuple(a))
            # ensure next appended later (no duplication)
            # will append b in next iteration or after loop
    # append final goal explicitly
    if not discrete_path or discrete_path[-1] != tuple(np.round(best_path[-1]).astype(int)):
        discrete_path.append(tuple(np.round(best_path[-1]).astype(int)))
    return discrete_path

# ==============================
# Run GWO, postprocess, and plot
# ==============================
best_path, convergence = gwo_path_planning(num_wolves=30, num_points=25, max_iter=150)

# Post-process: replace colliding straight segments with grid A* paths
discrete = postprocess_with_astar(best_path)  # list of (r,c)
discrete = np.array(discrete)  # shape (M,2)

print("Final discrete path length (cells):", len(discrete))

# Plot
plt.figure(figsize=(7,7))
plt.imshow(obstacle_map, cmap='gray_r', origin='lower')
# plot discrete path: cols->x, rows->y
plt.plot(discrete[:,1], discrete[:,0], 'r-', lw=2, label='Best (A*-stitched) Path')
plt.scatter(START[1], START[0], c='green', s=100, label='Start')
plt.scatter(GOAL[1],  GOAL[0],  c='blue',  s=100, label='Goal')
plt.title("Robot Path Planning (GWO waypoints + A* stitching)")
plt.legend()
plt.grid(True)
plt.xlim(-0.5, GRID_SIZE-0.5)
plt.ylim(-0.5, GRID_SIZE-0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.figure()
plt.plot(convergence, '-', lw=2)
plt.title("Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()
