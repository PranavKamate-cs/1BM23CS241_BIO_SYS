import numpy as np
print("Name: PRANAV GAJANAN KAMATE\nUSN: 1BM23CS241")
# ==========================
# 1️⃣ Problem Data
# ==========================
demand = 500  # MW

# Generator data: [a, b, c, Pmin, Pmax]
generators = np.array([
    [500, 5.3, 0.004, 100, 400],
    [400, 5.5, 0.006, 100, 350],
    [200, 5.8, 0.009,  50, 300]
])

num_gens = len(generators)

# ==========================
# 2️⃣ Helper Functions
# ==========================

def cost_function(P):
    """Calculate total generation cost + penalty for power balance violation."""
    total_cost = 0
    for i in range(num_gens):
        a, b, c, *_ = generators[i]
        total_cost += a + b*P[i] + c*(P[i]**2)

    # Penalty for power mismatch
    penalty = 1e5 * (np.sum(P) - demand)**2
    return total_cost + penalty

def random_solution():
    """Generate a random feasible power allocation that satisfies limits approximately."""
    P = np.array([np.random.uniform(gen[3], gen[4]) for gen in generators])
    # Adjust proportionally to match total demand
    P = P * demand / np.sum(P)
    return P

def levy_flight(beta=1.5):
    """Generate step size using Lévy distribution."""
    sigma = (np.math.gamma(1+beta) * np.sin(np.pi*beta/2) /
             (np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, size=num_gens)
    v = np.random.normal(0, 1, size=num_gens)
    step = u / np.abs(v)**(1/beta)
    return step

def apply_bounds(P):
    """Ensure generator limits are respected."""
    for i in range(num_gens):
        P[i] = np.clip(P[i], generators[i][3], generators[i][4])
    # Adjust to meet demand approximately
    P = P * demand / np.sum(P)
    return P

# ==========================
# 3️⃣ Cuckoo Search Parameters
# ==========================
n = 20              # Number of nests
pa = 0.25           # Discovery probability
max_iter = 200

# ==========================
# 4️⃣ Initialization
# ==========================
nests = np.array([random_solution() for _ in range(n)])
fitness = np.array([cost_function(P) for P in nests])
best_idx = np.argmin(fitness)
best = nests[best_idx].copy()

# ==========================
# 5️⃣ Main Loop
# ==========================
for t in range(max_iter):
    for i in range(n):
        # Lévy flight from current nest
        step = levy_flight()
        new_nest = nests[i] + step * (nests[i] - best) * 0.01
        new_nest = apply_bounds(new_nest)

        new_fitness = cost_function(new_nest)
        j = np.random.randint(n)

        if new_fitness < fitness[j]:
            nests[j] = new_nest
            fitness[j] = new_fitness

    # Abandon a fraction of worst nests
    num_abandon = int(pa * n)
    worst_idx = np.argsort(fitness)[-num_abandon:]
    for idx in worst_idx:
        nests[idx] = random_solution()
        fitness[idx] = cost_function(nests[idx])

    # Update global best
    current_best_idx = np.argmin(fitness)
    if fitness[current_best_idx] < cost_function(best):
        best = nests[current_best_idx].copy()

# ==========================
# 6️⃣ Results
# ==========================
print("=== Cuckoo Search Economic Dispatch ===")
for i in range(num_gens):
    print(f"Generator {i+1}: {best[i]:.2f} MW")

print(f"Total Power: {np.sum(best):.2f} MW")
print(f"Total Cost: ₹ {cost_function(best):.2f}")
