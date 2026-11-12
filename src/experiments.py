"""Experiment drivers shared by run_demo.py and run_all.py."""
import time

import numpy as np

from .baselines import brute_force, greedy, simulated_annealing
from .config import build_params, scenario_names
from .metrics import finance_summary, random_sampling_metrics, sample_metrics
from .qaoa import QAOASolver
from .qubo import min_penalty


def solve_instance(p, ps=(1, 2, 3), restarts=5, maxiter=150, shots_tune=1000,
                   shots_final=8192, seed=0, sa_steps=3000, sa_restarts=5):
    """Run exact, greedy, SA and QAOA(p) on one instance; return a results dict."""
    bf = brute_force(p)
    solver = QAOASolver(p)
    out = {"params": p, "bf": bf, "greedy": greedy(p), "solver": solver}

    t0 = time.time()
    out["sa"] = simulated_annealing(p, solver.energies, seed=seed, steps=sa_steps,
                                    restarts=sa_restarts)
    out["sa"]["runtime_s"] = time.time() - t0
    out["random"] = random_sampling_metrics(p, bf, solver.energies, shots_final, seed)

    out["qaoa"] = {}
    for pp in ps:
        res = solver.run(pp, restarts=restarts, seed=seed, maxiter=maxiter,
                         shots_tune=shots_tune, shots_final=shots_final)
        m_best = sample_metrics(p, res["best"]["counts"], bf, solver.energies)
        per_restart = [sample_metrics(p, r["counts"], bf, solver.energies) for r in res["all"]]
        out["qaoa"][pp] = {
            "metrics": m_best, "restart_metrics": per_restart,
            "runtime_s": res["runtime_s"], "evals": [r["evals"] for r in res["all"]],
            "shots_tune": shots_tune, "shots_final": shots_final,
            "best_run": res["best"],
            "plan": plan_from_metrics(p, m_best),
        }
    return out


def plan_from_metrics(p, m):
    """The recommended plan = cheapest feasible plan among the final samples."""
    if m["best_int"] is None:
        return None
    return np.array([(m["best_int"] >> i) & 1 for i in range(p.n)])


def random_instance(seed, n=10, K=3, edge_prob=0.4):
    """Synthetic random network of the same size; fixed seed -> reproducible."""
    from .config import Params
    rng = np.random.default_rng(seed)
    r = rng.uniform(15, 90, n).round()
    d = (0.35 * r * rng.uniform(0.5, 1.5, n)).round()
    W = np.zeros((n, n))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                w = float(rng.integers(3, 16))
                W[i, j] = W[j, i] = w
                edges.append((i, j, w))
    p = Params(names=[f"N{i}" for i in range(n)], shorts=[f"N{i}" for i in range(n)],
               pos=[(np.cos(2 * np.pi * i / n) / 2 + .5, np.sin(2 * np.pi * i / n) / 2 + .5)
                    for i in range(n)],
               r=r, W=W, d=d, K=K, lam=0.5, mu=1.0, A=100.0, label=f"R{seed}",
               title=f"random instance {seed}", edges=edges)
    # Give each random instance 1.5x the smallest penalty that works for it,
    # the same headroom rule as the main network (A = 100 vs minimum 58-66).
    return p.with_A(round(1.5 * min_penalty(p), 1))
