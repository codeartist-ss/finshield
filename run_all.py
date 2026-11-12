"""Reproduce every result file.

  python run_all.py --quick     about 2 minutes (fewer restarts / instances)
  python run_all.py --full      the settings used for the reported numbers

Outputs go to results/ (CSV + PNG). All money values are SYNTHETIC.
"""
import argparse
import csv
import os
import time

import numpy as np

from src.config import build_params, load_config, scenario_names
from src.experiments import random_instance, solve_instance
from src.metrics import finance_summary
from src.plotting import (architecture_diagram, hero_figure, plot_methods,
                          plot_p_curves, plot_penalty_sweep)
from src.qubo import min_penalty, safe_penalty

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")

SETTINGS = {
    "quick": dict(restarts=3, maxiter=100, n_random=6, sweep_mults=[0.5, 1.0, 1.5, 3.0, 6.0],
                  sweep_ps=(2,), sweep_restarts=2),
    "full": dict(restarts=8, maxiter=200, n_random=20,
                 sweep_mults=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
                 sweep_ps=(1, 2, 3), sweep_restarts=4),
}
KEYS = ["p_optimal", "p_feasible", "q_best", "q_mean_feasible", "cost_gap", "approx_ratio"]


def write_csv(name, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote results/{name}")


def stage_scenarios(cfg, S):
    print("\n[1/3] Three scenarios on the main network")
    qrows, frows, mrows = [], [], []
    for s in scenario_names(cfg):
        p = build_params(cfg, s)
        res = solve_instance(p, restarts=S["restarts"], maxiter=S["maxiter"], seed=0)
        bf = res["bf"]
        fin_opt = finance_summary(p, bf["x"])
        for pp, q in res["qaoa"].items():
            m = q["metrics"]
            qrows.append({"scenario": s, "p": pp, **{k: round(float(m[k]), 5) for k in KEYS},
                          "restart_mean_p_optimal": round(float(np.mean([r["p_optimal"] for r in q["restart_metrics"]])), 5),
                          "restart_mean_p_feasible": round(float(np.mean([r["p_feasible"] for r in q["restart_metrics"]])), 5),
                          "random_p_optimal": round(float(res["random"]["p_optimal"]), 5),
                          "random_p_feasible": round(float(res["random"]["p_feasible"]), 5),
                          "shots_tune": q["shots_tune"], "shots_final": q["shots_final"],
                          "mean_cobyla_evals": round(float(np.mean(q["evals"])), 1),
                          "runtime_s_all_restarts": round(q["runtime_s"], 2)})
            plan = q["plan"]
            mrows.append({"scenario": s, "method": f"QAOA p={pp}",
                          "net_benefit": finance_summary(p, plan)["net_benefit"],
                          "runtime_s": q["runtime_s"]})
        for name, x, rt in (("Exact (brute force)", bf["x"], None), ("Greedy", res["greedy"]["x"], None),
                            ("Simulated annealing", res["sa"]["x"], res["sa"]["runtime_s"])):
            f = finance_summary(p, x)
            frows.append({"scenario": s, "title": p.title, "method": name,
                          "contained": " + ".join(f["contained"]),
                          "loss_do_nothing": f["loss_do_nothing"], "loss_avoided": f["loss_avoided"],
                          "disruption_cost": f["disruption_cost"], "net_benefit": f["net_benefit"]})
            mrows.append({"scenario": s, "method": name, "net_benefit": f["net_benefit"],
                          "runtime_s": rt if rt is not None else 1e-3})
        # hero figure uses the QAOA depth with the cheapest sampled plan
        best_p = min(res["qaoa"], key=lambda q: (res["qaoa"][q]["metrics"]["best_cost"],
                                                 -res["qaoa"][q]["metrics"]["p_optimal"]))
        plan = res["qaoa"][best_p]["plan"]
        f = finance_summary(p, plan)
        frows.append({"scenario": s, "title": p.title, "method": f"QAOA p={best_p} (best sampled)",
                      "contained": " + ".join(f["contained"]), "loss_do_nothing": f["loss_do_nothing"],
                      "loss_avoided": f["loss_avoided"], "disruption_cost": f["disruption_cost"],
                      "net_benefit": f["net_benefit"]})
        hero_figure(p, plan, f, os.path.join(OUT, f"hero_scenario_{s}.png"), f"QAOA p={best_p}")
    # exact-brute-force runtime: time one call for the runtime chart
    t0 = time.time(); from src.baselines import brute_force; brute_force(build_params(cfg, "A"))
    bf_t = time.time() - t0
    for r in mrows:
        if r["method"].startswith("Exact") or r["method"] == "Greedy":
            r["runtime_s"] = bf_t if r["method"].startswith("Exact") else 1e-3
    write_csv("scenario_qaoa_metrics.csv", qrows)
    write_csv("scenario_finance.csv", frows)
    plot_methods(mrows, os.path.join(OUT, "methods_comparison.png"))
    return qrows


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def stage_random(S):
    print(f"\n[2/3] {S['n_random']} random synthetic instances (seeds 100..)")
    rows, per_p, rnd = [], {1: [], 2: [], 3: []}, []
    for k in range(S["n_random"]):
        p = random_instance(100 + k)
        res = solve_instance(p, restarts=max(2, S["restarts"] // 2), maxiter=S["maxiter"], seed=k)
        rnd.append(res["random"])
        gap_greedy = res["greedy"]["cost"] - res["bf"]["cost"]
        for pp, q in res["qaoa"].items():
            m = q["metrics"]
            per_p[pp].append(m)
            rows.append({"seed": 100 + k, "p": pp, "A": p.A, **{k2: round(float(m[k2]), 5) for k2 in KEYS},
                         "greedy_cost_gap": round(float(gap_greedy), 3),
                         "sa_cost_gap": round(float(res["sa"]["cost"] - res["bf"]["cost"]), 3),
                         "random_p_optimal": round(float(res["random"]["p_optimal"]), 5)})
        print(f"  instance {k + 1}/{S['n_random']} done")
    write_csv("random_instances.csv", rows)
    summary = {pp: {k: mean_std([m[k] for m in ms]) for k in KEYS} for pp, ms in per_p.items()}
    ref = {k: float(np.mean([r[k] for r in rnd])) for k in KEYS}
    write_csv("random_instances_summary.csv",
              [{"p": pp, **{f"{k}_mean": round(v[0], 5) for k, v in d.items()},
                **{f"{k}_std": round(v[1], 5) for k, v in d.items()}} for pp, d in summary.items()]
              + [{"p": "uniform random sampling", **{f"{k}_mean": round(v, 5) for k, v in ref.items()}}])
    plot_p_curves(summary, ref, os.path.join(OUT, "p_vs_success_quality.png"),
                  f"QAOA depth p vs success and quality, {S['n_random']} random synthetic instances "
                  "(error bars = std across instances)")
    return summary, ref


def stage_sweep(cfg, S):
    print("\n[3/3] Penalty sweep on scenario A")
    from src.experiments import solve_instance as _solve
    p = build_params(cfg, "A")
    a_min, a_safe = min_penalty(p), safe_penalty(p, 1.0)
    rows = []
    for mult in S["sweep_mults"]:
        A = round(a_min * mult, 1)
        res = _solve(p.with_A(A), ps=S["sweep_ps"], restarts=S["sweep_restarts"], maxiter=S["maxiter"],
                     seed=1, sa_steps=500, sa_restarts=1)
        for pp, q in res["qaoa"].items():
            m = q["metrics"]
            rows.append({"A": A, "A_over_min": mult, "p": pp, **{k: round(float(m[k]), 5) for k in KEYS}})
        print(f"  A = {A}")
    write_csv("penalty_sweep.csv", rows)
    pp = S["sweep_ps"][-1] if 2 not in S["sweep_ps"] else 2
    plot_penalty_sweep([r for r in rows if r["p"] == pp], a_min, a_safe, p.A,
                       os.path.join(OUT, "penalty_sweep.png"), f"p={pp}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    args = ap.parse_args()
    S = SETTINGS["quick" if args.quick else "full"]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    cfg, t0 = load_config(), time.time()
    stage_scenarios(cfg, S)
    stage_random(S)
    stage_sweep(cfg, S)
    architecture_diagram(os.path.join(ROOT, "docs", "architecture.png"))
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Files are in results/ and docs/.")


if __name__ == "__main__":
    main()
