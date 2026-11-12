"""Stage-1 classical pipeline: exact + greedy for each scenario, in finance terms.

Run:  python run_classical.py
"""
from src.baselines import brute_force, greedy
from src.config import build_params, load_config, scenario_names
from src.metrics import finance_summary


def main():
    cfg = load_config()
    for s in scenario_names(cfg):
        p = build_params(cfg, s)
        bf, gr = brute_force(p), greedy(p)
        f = finance_summary(p, bf["x"])
        print(f"\nScenario {s}: {p.title}  (all values {p.currency})")
        print("  exact plan :", ", ".join(f["contained"]))
        print("  greedy plan:", ", ".join(finance_summary(p, gr["x"])["contained"]))
        print(f"  do nothing -> expected loss {f['loss_do_nothing']:.1f}")
        print(f"  loss avoided {f['loss_avoided']:.1f} | disruption cost {f['disruption_cost']:.1f}"
              f" | NET BENEFIT {f['net_benefit']:.1f}")


if __name__ == "__main__":
    main()
