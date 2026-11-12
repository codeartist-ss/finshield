"""Classical baselines: exact brute force, greedy, simulated annealing."""
import numpy as np

from .qubo import all_bitstrings, original_objective


def bits_to_int(x):
    return int(sum(int(b) << i for i, b in enumerate(x)))


def int_to_bits(k, n):
    return np.array([(k >> i) & 1 for i in range(n)], dtype=int)


def cost_table(p):
    """Penalty-free cost C(x) for all 2^n plans (real money terms only)."""
    X = all_bitstrings(p.n)
    return np.array([original_objective(p, x, with_penalty=False) for x in X])


def brute_force(p):
    """Exact optimum over all plans with exactly K teams (1,024 for n=10)."""
    X = all_bitstrings(p.n)
    C = cost_table(p)
    feasible = X.sum(axis=1) == p.K
    idx = np.where(feasible)[0]
    best = idx[np.argmin(C[idx])]
    return {"x": X[best], "cost": float(C[best]), "cost_table": C,
            "feasible": feasible, "cmin": float(C[idx].min()),
            "cmax": float(C[idx].max())}


def greedy(p):
    """Add one team at a time where it lowers the cost most; stops at K teams."""
    x = np.zeros(p.n, dtype=int)
    for _ in range(p.K):
        best_i, best_c = None, np.inf
        for i in range(p.n):
            if x[i]:
                continue
            trial = x.copy()
            trial[i] = 1
            c = original_objective(p, trial, with_penalty=False)
            if c < best_c:
                best_i, best_c = i, c
        x[best_i] = 1
    return {"x": x, "cost": original_objective(p, x, with_penalty=False)}


def simulated_annealing(p, energies, seed=0, steps=3000, restarts=5,
                        t_start=None, t_end=0.5):
    """Single-bit-flip annealing on the SAME penalised QUBO the quantum code sees.

    energies: QUBO energy table (index = bitstring integer). Using the table
    keeps the code tiny; for n=10 this is exact and fast.
    Returns the best FEASIBLE plan seen (exactly K teams), like QAOA's output.
    """
    rng = np.random.default_rng(seed)
    n = p.n
    t_start = t_start or float(np.ptp(energies)) * 0.05
    best_x, best_c = None, np.inf
    for _ in range(restarts):
        x = rng.integers(0, 2, n)
        e = energies[bits_to_int(x)]
        for step in range(steps):
            t = t_start * (t_end / t_start) ** (step / steps)  # geometric cooling
            i = rng.integers(n)
            y = x.copy()
            y[i] ^= 1
            ey = energies[bits_to_int(y)]
            if ey <= e or rng.random() < np.exp(-(ey - e) / t):
                x, e = y, ey
            if x.sum() == p.K:
                c = original_objective(p, x, with_penalty=False)
                if c < best_c:
                    best_x, best_c = x.copy(), c
    return {"x": best_x, "cost": float(best_c)}
