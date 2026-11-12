"""FinShield demo (for the screen recording). Prints the story, saves hero figures.

Run:  python run_demo.py            (all three scenarios, about 30-60 s)
      python run_demo.py --scenario A
All money values are SYNTHETIC USD millions.
"""
import argparse
import os
import time

from src.config import build_params, load_config, scenario_names
from src.experiments import solve_instance
from src.metrics import finance_summary
from src.plotting import hero_figure

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def wrap(text, width=88):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    return "\n    ".join(out + [line])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["A", "B", "C"], default=None)
    args = ap.parse_args()
    cfg = load_config()
    os.makedirs(RESULTS, exist_ok=True)
    t_all = time.time()
    print("=" * 92)
    print("FinShield: quantum-assisted containment planning (SYNTHETIC data, simulator only)")
    print("=" * 92)

    for s in ([args.scenario] if args.scenario else scenario_names(cfg)):
        p = build_params(cfg, s)
        print(f"\n--- Scenario {s}: {p.title} ---")
        print("    " + wrap(p.narrative))
        res = solve_instance(p, ps=(1, 2, 3), restarts=4, maxiter=120, seed=0)

        # Recommended plan: the QAOA depth whose sampled best plan is cheapest;
        # ties go to the highest probability of sampling the optimum.
        best_p = min(res["qaoa"], key=lambda q: (res["qaoa"][q]["metrics"]["best_cost"],
                                                 -res["qaoa"][q]["metrics"]["p_optimal"]))
        q = res["qaoa"][best_p]
        plan = q["plan"]
        fin = finance_summary(p, plan)
        opt = finance_summary(p, res["bf"]["x"])
        print(f"\n  Teams ({p.K}) go to: {', '.join(fin['contained'])}")
        print(f"  Expected loss if we do nothing : {fin['loss_do_nothing']:8.1f}")
        print(f"  Loss avoided by the plan       : {fin['loss_avoided']:8.1f}")
        print(f"  Disruption cost of containing  : {fin['disruption_cost']:8.1f}")
        print(f"  NET BENEFIT                    : {fin['net_benefit']:8.1f}  (synthetic USD m)")
        m = q["metrics"]
        print(f"\n  QAOA (p={best_p}) vs exact: cost gap {m['cost_gap']:.1f}; "
              f"exact net benefit {opt['net_benefit']:.1f}; "
              f"P(sample optimal plan) = {m['p_optimal']:.2%} vs {res['random']['p_optimal']:.2%} "
              f"for random sampling.")
        print(f"  Classical: greedy cost {res['greedy']['cost']:.1f}, simulated annealing "
              f"{res['sa']['cost']:.1f}, exact {res['bf']['cost']:.1f}. Brute force is fastest at "
              f"10 qubits; we claim no quantum speedup.")
        path = os.path.join(RESULTS, f"hero_scenario_{s}.png")
        hero_figure(p, plan, fin, path, plan_source=f"QAOA p={best_p}")
        print(f"  Saved hero figure: results/hero_scenario_{s}.png")

    print(f"\nDone in {time.time() - t_all:.0f} s.")


if __name__ == "__main__":
    main()
