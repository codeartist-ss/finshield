"""Finance-unit outputs and quantum-sample metrics (definitions in README)."""
import numpy as np

from .qubo import original_objective


def finance_summary(p, x):
    """Plain-money view of a plan, in synthetic currency units."""
    x = np.asarray(x)
    nothing = original_objective(p, np.zeros(p.n), with_penalty=False)
    residual = float(np.sum(p.r * (1 - x)))
    contagion = sum(p.lam * w * (1 - x[i]) * (1 - x[j]) for i, j, w in p.edges)
    disruption = float(p.mu * np.sum(p.d * x))
    loss_avoided = nothing - residual - contagion
    return {"loss_do_nothing": nothing, "residual_loss": residual + contagion,
            "loss_avoided": loss_avoided, "disruption_cost": disruption,
            "net_benefit": loss_avoided - disruption,
            "contained": [p.names[i] for i in range(p.n) if x[i]]}


def sample_metrics(p, counts_by_int, bf, qubo_energies):
    """Metrics from a sampled distribution {bitstring_int: shots}.

    bf: output of brute_force(p).  qubo_energies: penalised QUBO energy table.
    Quality = (Cmax - C)/(Cmax - Cmin) where Cmin/Cmax are the best/worst cost
    among FEASIBLE plans (exactly K teams); 1 = optimal, 0 = worst feasible.
    """
    n = p.n
    total = sum(counts_by_int.values())
    C, feas = bf["cost_table"], bf["feasible"]
    cmin, cmax = bf["cmin"], bf["cmax"]
    # Ties happen (two different plans can cost exactly the same), so "optimal"
    # means ANY feasible plan within 1e-9 of the best cost.
    opt_ints = set(np.where(feas & (C <= cmin + 1e-9))[0].tolist())
    span = max(cmax - cmin, 1e-12)

    p_opt = sum(v for k, v in counts_by_int.items() if k in opt_ints) / total
    feas_counts = {k: v for k, v in counts_by_int.items() if feas[k]}
    n_feas = sum(feas_counts.values())
    p_feas = n_feas / total

    if feas_counts:
        best_int = min(feas_counts, key=lambda k: C[k])
        best_cost = float(C[best_int])
        q_best = (cmax - best_cost) / span
        mean_cost = sum(C[k] * v for k, v in feas_counts.items()) / n_feas
        q_mean = (cmax - mean_cost) / span
    else:  # nothing feasible sampled: report 0 quality and flag it
        best_int, best_cost, q_best, q_mean = None, np.nan, 0.0, 0.0

    # Approximation ratio on the whole (penalised) QUBO energy landscape.
    e_mean = sum(qubo_energies[k] * v for k, v in counts_by_int.items()) / total
    e_min, e_max = float(qubo_energies.min()), float(qubo_energies.max())
    approx_ratio = (e_max - e_mean) / (e_max - e_min)

    return {"p_optimal": p_opt, "p_feasible": p_feas, "q_best": q_best,
            "q_mean_feasible": q_mean, "best_int": best_int,
            "best_cost": best_cost, "cost_gap": best_cost - bf["cost"],
            "approx_ratio": approx_ratio, "n_optimal_plans": len(opt_ints), "feasible_found": bool(feas_counts)}


def random_sampling_metrics(p, bf, qubo_energies, shots=8192, seed=0):
    """Reference: sample plans uniformly at random (no quantum, no cleverness).

    Any QAOA claim must be read against this. With 8,192 shots over 1,024
    plans, random sampling already hits the optimum at least once, so
    'best sampled quality' alone proves little; probabilities are the fair test.
    """
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 2 ** p.n, shots)
    counts = {}
    for k in draws:
        counts[int(k)] = counts.get(int(k), 0) + 1
    return sample_metrics(p, counts, bf, qubo_energies)
