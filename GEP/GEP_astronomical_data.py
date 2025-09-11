import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ================================
# Dataset (same as GA example)
# ================================
n_points = 300
t = np.sort(rng.uniform(0.0, 100.0, n_points))
A_true, P_true, phi_true, m0_true = 0.35, 12.7, 0.6, 13.2
noise_sigma = 0.05

y = m0_true + A_true * np.sin(2 * np.pi * t / P_true + phi_true) \
    + rng.normal(0, noise_sigma, size=n_points)
yerr = np.full_like(y, noise_sigma)

# ================================
# GEP Building Blocks
# ================================
functions = {
    "add": (2, lambda a, b: a + b),
    "sub": (2, lambda a, b: a - b),
    "mul": (2, lambda a, b: a * b),
    # safe division
    "div": (2, lambda a, b: np.where(np.abs(b) > 1e-8, a / b, 1.0)),
    "sin": (1, np.sin),
    "cos": (1, np.cos)
}
terminals = ["t", "const"]

def random_expression(max_depth=3):
    """ Recursively build a random expression tree. """
    if max_depth == 0 or rng.random() < 0.3:
        if rng.random() < 0.5:
            return ("t",)  # variable
        else:
            return ("const", rng.uniform(-2, 2))  # random constant
    func = rng.choice(list(functions.keys()))
    arity, _ = functions[func]
    return (func,) + tuple(random_expression(max_depth - 1) for _ in range(arity))

def evaluate(expr, t_val):
    """ Evaluate expression tree for given t values (vectorized). """
    op = expr[0]
    if op == "t":
        return t_val
    elif op == "const":
        return np.full_like(t_val, expr[1], dtype=float)
    else:
        arity, func = functions[op]
        args = [evaluate(arg, t_val) for arg in expr[1:1+arity]]
        return func(*args)

def fitness(expr, t, y):
    """ Mean squared error fitness. """
    try:
        y_pred = evaluate(expr, t)
        mse = np.mean((y - y_pred) ** 2)
        return 1 / (1 + mse)  # higher is better
    except Exception:
        return 1e-6

def mutate(expr, prob=0.1, max_depth=3):
    """ Randomly mutate an expression. """
    if rng.random() < prob:
        return random_expression(max_depth)
    op = expr[0]
    if op in ("t", "const"):
        return expr
    arity, _ = functions[op]
    return (op,) + tuple(mutate(arg, prob, max_depth) for arg in expr[1:1+arity])

def crossover(e1, e2, prob=0.7):
    """ Random subtree crossover. """
    if rng.random() > prob:
        return e1
    if e1[0] in ("t", "const"):
        return e2
    arity, _ = functions[e1[0]]
    i = rng.integers(arity)
    new_args = list(e1[1:])
    new_args[i] = crossover(new_args[i], e2, prob)
    return (e1[0],) + tuple(new_args)

# Pretty printer
def expr_to_str(expr):
    """ Convert an expression tree to a human-readable string. """
    op = expr[0]
    if op == "t":
        return "t"
    elif op == "const":
        return f"{expr[1]:.3f}"
    elif op in ("add", "sub", "mul", "div"):
        a, b = expr[1], expr[2]
        a_str, b_str = expr_to_str(a), expr_to_str(b)
        if op == "add":
            return f"({a_str} + {b_str})"
        elif op == "sub":
            return f"({a_str} - {b_str})"
        elif op == "mul":
            return f"({a_str} * {b_str})"
        elif op == "div":
            return f"({a_str} / {b_str})"
    elif op in ("sin", "cos"):
        return f"{op}({expr_to_str(expr[1])})"
    else:
        return str(expr)  # fallback

# ================================
# Run GEP
# ================================
pop_size = 100
generations = 50

# Initialize population
population = [random_expression(max_depth=4) for _ in range(pop_size)]
fitnesses = [fitness(ind, t, y) for ind in population]

best_curve = []

for gen in range(generations):
    # Elitism
    elite_idx = np.argmax(fitnesses)
    elite = population[elite_idx]

    next_pop = [elite]  # carry best

    while len(next_pop) < pop_size:
        # Tournament selection for parent1
        idx1 = rng.choice(len(population), size=3, replace=False)
        parent1 = max((population[i] for i in idx1), key=lambda ind: fitness(ind, t, y))

        # Tournament selection for parent2
        idx2 = rng.choice(len(population), size=3, replace=False)
        parent2 = max((population[i] for i in idx2), key=lambda ind: fitness(ind, t, y))

        # Crossover + mutation
        child = crossover(parent1, parent2)
        child = mutate(child, prob=0.2)
        next_pop.append(child)

    population = next_pop
    fitnesses = [fitness(ind, t, y) for ind in population]
    best_fit = max(fitnesses)
    best_curve.append(best_fit)

# Best solution
best_idx = np.argmax(fitnesses)
best_expr = population[best_idx]
print("Best evolved expression (raw):", best_expr)
print("Best evolved expression (pretty):", expr_to_str(best_expr))

# ================================
# Plot results
# ================================
tt = np.linspace(t.min(), t.max(), 1000)
y_pred = evaluate(best_expr, tt)

plt.figure(figsize=(8, 5))
plt.scatter(t, y, s=12, label="observed")
plt.plot(tt, y_pred, label="GEP best-fit", color="red")
plt.gca().invert_yaxis()
plt.xlabel("Time (days)")
plt.ylabel("Magnitude")
plt.title("Light Curve Fit with Gene Expression Programming (custom)")
plt.legend()
plt.tight_layout()
plt.show()

# Convergence
plt.figure(figsize=(8, 4.5))
plt.plot(best_curve)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("GEP Convergence")
plt.tight_layout()
plt.show()
