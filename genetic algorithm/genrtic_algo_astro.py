import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def light_curve_model(t, A, P, phi, m0):
    return m0 + A * np.sin(2 * np.pi * t / P + phi)

def chi2(params, t, y, yerr):
    A, P, phi, m0 = params
    y_model = light_curve_model(t, A, P, phi, m0)
    r = (y - y_model) / yerr
    return float(np.sum(r * r))

def fitness(params, t, y, yerr):
    c2 = chi2(params, t, y, yerr)
    return 1.0 / (1.0 + c2)

def run_ga(t, y, yerr, bounds, cfg):
    pop_size, generations, elite_frac, tournament_k, mutation_rate, mutation_scale, crossover_rate, rng = cfg
    def clip_to_bounds(ind):
        for i, (lo, hi) in enumerate(bounds):
            ind[i] = np.clip(ind[i], lo, hi)
        return ind

    def random_individual():
        return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)

    def tournament_select(pop, fitnesses, k):
        idx = rng.integers(0, len(pop), size=k)
        best = idx[np.argmax(fitnesses[idx])]
        return pop[best].copy()

    def blend_crossover(p1, p2):
        alpha = rng.uniform(0.0, 1.0, size=p1.shape)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        return c1, c2

    def mutate(ind):
        ind = ind.copy()
        for i, (lo, hi) in enumerate(bounds):
            if rng.random() < mutation_rate:
                span = (hi - lo)
                ind[i] += rng.normal(0.0, mutation_scale * span)
        return clip_to_bounds(ind)

    pop = np.array([random_individual() for _ in range(pop_size)])
    fit = np.array([fitness(ind, t, y, yerr) for ind in pop])
    best_curve = [float(np.max(fit))]
    n_elite = max(1, int(elite_frac * pop_size))

    for _ in range(generations):
        elite_idx = np.argsort(fit)[-n_elite:]
        elites = pop[elite_idx].copy()
        next_pop = []
        while len(next_pop) < pop_size - n_elite:
            p1 = tournament_select(pop, fit, tournament_k)
            p2 = tournament_select(pop, fit, tournament_k)
            if rng.random() < crossover_rate:
                c1, c2 = blend_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = mutate(c1)
            c2 = mutate(c2)
            next_pop.append(c1)
            if len(next_pop) < pop_size - n_elite:
                next_pop.append(c2)
        pop = np.vstack([elites, np.array(next_pop)])
        fit = np.array([fitness(ind, t, y, yerr) for ind in pop])
        best_curve.append(float(np.max(fit)))

    best_idx = int(np.argmax(fit))
    best_params = pop[best_idx].copy()
    return best_params, best_curve

if __name__ == "__main__":
    # Generate synthetic dataset
    n_points = 300
    t = np.sort(rng.uniform(0.0, 100.0, n_points))
    A_true, P_true, phi_true, m0_true = 0.35, 12.7, 0.6, 13.2
    noise_sigma = 0.05
    y = m0_true + A_true * np.sin(2 * np.pi * t / P_true + phi_true) + rng.normal(0, noise_sigma, size=n_points)
    yerr = np.full_like(y, noise_sigma)

    # Bounds: (A, P, phi, m0)
    bounds = [(0.0, 1.0), (5.0, 20.0), (0.0, 2 * np.pi), (12.0, 14.0)]
    cfg = (200, 250, 0.05, 3, 0.15, 0.07, 0.9, np.random.default_rng(2025))

    best_params, best_curve = run_ga(t, y, yerr, bounds, cfg)
    A_hat, P_hat, phi_hat, m0_hat = best_params

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.scatter(t, y, s=12, label="observed")
    tt = np.linspace(t.min(), t.max(), 1000)
    plt.plot(tt, m0_hat + A_hat * np.sin(2 * np.pi * tt / P_hat + phi_hat), label="GA best-fit")
    plt.gca().invert_yaxis()
    plt.xlabel("Time (days)")
    plt.ylabel("Magnitude")
    plt.title("Light Curve Fit with Genetic Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Convergence
    plt.figure(figsize=(8, 4.5))
    plt.plot(best_curve)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Convergence")
    plt.tight_layout()
    plt.show()
