"""
PCOO-based scalable multi-agent coordination demo (single-file).
- Grid of organisms; each organism encodes boid-like behavior weights.
- Each generation runs a joint simulation to evaluate all agents' fitness.
- PCOO local updates (neighborhood crossover/mutation/selection) with double-buffering.
- Produces plots and saves top genomes to CSV.

Dependencies:
    pip install numpy matplotlib pandas

Optional (Jupyter interactive table):
    caas_jupyter_tools (only used in the original run; not required)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

np.random.seed(42)

# ----- Parameters -----
GRID_W, GRID_H = 8, 8            # grid of organisms
NUM_AGENTS = GRID_W * GRID_H
GENOME_DIM = 4                   # [w_align, w_cohere, w_sep, w_goal]
P_SWAP = 0.02
P_MUT = 0.15
MAX_GEN = 30
SIM_STEPS = 120                  # steps per simulation used for fitness evaluation
ARENA_SIZE = 100.0               # square arena 0..ARENA_SIZE in both axes
NEIGH_RADIUS = 8.0               # sensing radius for agent behaviors
DT = 1.0                         # time step for movement update
MAX_SPEED = 2.0                  # cap agent speed

# ----- Utility helpers -----
def limit_vec(v, maxnorm):
    norm = np.linalg.norm(v)
    if norm > maxnorm and norm > 0:
        return v / norm * maxnorm
    return v

# ----- Genome -----
def random_genome():
    return np.random.uniform(-1, 1, size=(GENOME_DIM,))

# ----- Initialize population grid -----
population = np.array([[random_genome() for _ in range(GRID_W)] for _ in range(GRID_H)])  # (H, W, GENOME_DIM)

# ----- Simulation to evaluate fitness of all agents together -----
def evaluate_population(pop):
    H, W, _ = pop.shape
    N = H * W
    positions = np.random.uniform(0, ARENA_SIZE, size=(N, 2))
    velocities = np.random.normal(0, 1, size=(N, 2))
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    speeds[speeds == 0] = 1.0
    velocities = (velocities / speeds) * 0.5

    genomes = pop.reshape((N, GENOME_DIM))
    goal = np.random.uniform(ARENA_SIZE*0.2, ARENA_SIZE*0.8, size=(2,))

    dist_to_goal_history = np.zeros((N, SIM_STEPS))
    collision_counts = np.zeros(N)
    grid_bins = 20
    bin_size = ARENA_SIZE / grid_bins
    visited_bins = [set() for _ in range(N)]

    for t in range(SIM_STEPS):
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=2) + np.eye(N) * 1e6
        neighbors_mask = dists < NEIGH_RADIUS

        align = np.zeros((N,2))
        cohesion = np.zeros((N,2))
        separation = np.zeros((N,2))
        to_goal = (goal - positions)

        for i in range(N):
            neigh_idx = np.where(neighbors_mask[i])[0]
            if neigh_idx.size > 0:
                align[i] = np.mean(velocities[neigh_idx], axis=0) - velocities[i]
                cohesion[i] = (np.mean(positions[neigh_idx], axis=0) - positions[i])
                close = np.where(dists[i] < (NEIGH_RADIUS * 0.5))[0]
                if close.size > 0:
                    sep = np.sum((positions[i] - positions[close]) / (dists[i, close][:,None] + 1e-6), axis=0)
                    separation[i] = sep

        steer = (genomes[:,0:1] * align +
                 genomes[:,1:2] * cohesion +
                 genomes[:,2:3] * separation +
                 genomes[:,3:4] * to_goal)

        velocities += steer * DT * 0.05
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        too_fast = speeds[:,0] > MAX_SPEED
        if np.any(too_fast):
            velocities[too_fast] = (velocities[too_fast] / speeds[too_fast]) * MAX_SPEED
        positions += velocities * DT
        positions = np.mod(positions, ARENA_SIZE)

        dist_to_goal = np.linalg.norm(positions - goal, axis=1)
        dist_to_goal_history[:, t] = dist_to_goal

        coll_pairs = np.where(dists < 1.0)
        for a, b in zip(*coll_pairs):
            if a < b:
                collision_counts[a] += 1
                collision_counts[b] += 1

        bins = np.floor(positions / bin_size).astype(int)
        bins = np.clip(bins, 0, grid_bins-1)
        for i in range(N):
            visited_bins[i].add((int(bins[i,0]), int(bins[i,1])))

    mean_dist = np.mean(dist_to_goal_history, axis=1)
    unique_bins = np.array([len(s) for s in visited_bins])

    max_possible = ARENA_SIZE * math.sqrt(2)
    norm_dist = 1.0 - (mean_dist / max_possible)            # higher better
    norm_coll = 1.0 - (collision_counts / (SIM_STEPS + 1)) # higher better
    norm_cov = unique_bins / (grid_bins * grid_bins)       # 0..1

    fitness = 0.5 * norm_dist + 0.3 * norm_cov + 0.2 * norm_coll
    fitness = np.clip(fitness, 0.0, 1.0)

    fitness_grid = fitness.reshape((H, W))
    diagnostics = {
        "mean_dist": mean_dist.reshape((H, W)),
        "collisions": collision_counts.reshape((H, W)),
        "coverage_bins": unique_bins.reshape((H, W)),
        "final_positions": positions.reshape((H, W, 2)),
        "goal": goal
    }
    return fitness_grid, diagnostics

# ----- Genetic operators -----
def mutate(genome, p_mut=P_MUT):
    g = genome.copy()
    for i in range(len(g)):
        if np.random.rand() < p_mut:
            g[i] += np.random.normal(0, 0.2)
    return np.clip(g, -1, 1)

def crossover(a, b):
    alpha = np.random.rand()
    child = alpha * a + (1-alpha) * b
    return np.clip(child, -1, 1)

def get_neighbors_coords(x, y, W=GRID_W, H=GRID_H):
    coords = []
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not (dx==0 and dy==0):
                coords.append((nx, ny))
    return coords

# ----- Main PCOO loop -----
print("Name: PRANAV GAJANAN KAMATE\nUSN: 1BM23CS241")
best_over_time = []
avg_over_time = []

population_next = population.copy()

for gen in range(1, MAX_GEN+1):
    fitness_grid, diag = evaluate_population(population)
    avg_fit = float(np.mean(fitness_grid))
    best_fit = float(np.max(fitness_grid))
    best_over_time.append(best_fit)
    avg_over_time.append(avg_fit)
    print(f"Generation {gen}/{MAX_GEN}  avg_fit={avg_fit:.4f}  best_fit={best_fit:.4f}")

    H, W, _ = population.shape
    population_next = population.copy()

    for y in range(H):
        for x in range(W):
            Xi = population[y, x]
            neigh = get_neighbors_coords(x, y, W, H)
            if len(neigh) > 0:
                px, py = neigh[np.random.randint(len(neigh))]
                partner = population[py, px]
            else:
                partner = Xi.copy()

            if np.random.rand() < 0.6:
                child = crossover(Xi, partner)
            else:
                child = mutate(Xi, p_mut=P_MUT)

            # selection heuristic: move toward best neighbor if it helps
            best_ng = Xi.copy()
            best_ng_fit = fitness_grid[y, x]
            for (nx, ny) in neigh:
                if fitness_grid[ny, nx] > best_ng_fit:
                    best_ng_fit = fitness_grid[ny, nx]
                    best_ng = population[ny, nx]

            dist_child_to_best = np.linalg.norm(child - best_ng)
            dist_x_to_best = np.linalg.norm(Xi - best_ng)
            if best_ng_fit > fitness_grid[y, x] and dist_child_to_best < dist_x_to_best:
                population_next[y, x] = mutate(child, p_mut=0.05)
            else:
                if np.random.rand() < 0.02:
                    population_next[y, x] = mutate(Xi, p_mut=0.2)
                else:
                    population_next[y, x] = Xi

    population = population_next.copy()

# Final evaluation and results
final_fitness, final_diag = evaluate_population(population)
best_idx = np.unravel_index(np.argmax(final_fitness), final_fitness.shape)
best_genome = population[best_idx]
best_fitness_val = final_fitness[best_idx]

print("\nBest organism found at grid position (row, col):", best_idx)
print("Best genome weights [align, cohesion, separation, goal]:", np.round(best_genome, 3))
print("Best fitness:", float(best_fitness_val))

H, W, _ = population.shape
all_genomes = population.reshape((H*W, GENOME_DIM))
all_fitness = final_fitness.reshape((H*W,))
top_k = 8
top_idx = np.argsort(-all_fitness)[:top_k]
top_table = pd.DataFrame({
    "grid_pos": [f"{i//W},{i%W}" for i in top_idx],
    "fitness": np.round(all_fitness[top_idx], 4),
    "align": np.round(all_genomes[top_idx,0], 3),
    "cohere": np.round(all_genomes[top_idx,1], 3),
    "sep": np.round(all_genomes[top_idx,2], 3),
    "goal": np.round(all_genomes[top_idx,3], 3)
})

# Plotting
plt.figure(figsize=(6,3))
plt.plot(best_over_time)
plt.title("Best fitness over generations")
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,3))
plt.plot(avg_over_time)
plt.title("Average fitness over generations")
plt.xlabel("Generation")
plt.ylabel("Average fitness")
plt.grid(True)
plt.show()

final_positions = final_diag["final_positions"].reshape((-1,2))
goal = final_diag["goal"]
plt.figure(figsize=(5,5))
plt.scatter(final_positions[:,0], final_positions[:,1], s=12)
plt.scatter([goal[0]], [goal[1]], s=80, marker='*')
plt.title("Final agent positions and shared goal")
plt.xlim(0, ARENA_SIZE)
plt.ylim(0, ARENA_SIZE)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Save top genomes CSV
out_path = "top_genomes.csv"
top_table.to_csv(out_path, index=False)
print(f"\nTop genomes saved to {out_path}")
