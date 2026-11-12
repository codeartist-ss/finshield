"""All figures. Plain matplotlib only (no extra installs)."""
import matplotlib
matplotlib.use("Agg")  # no window needed; we only save PNG files
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

GREEN, RED, GREY, BLUE = "#2e8b57", "#c0392b", "#7f8c8d", "#2c6fbb"


def _draw_network(ax, p, x, title, show_plan):
    """Nodes sized by risk; contained nodes green with a thick ring."""
    for i, j, w in p.edges:
        (x1, y1), (x2, y2) = p.pos[i], p.pos[j]
        cut = show_plan and (x[i] or x[j])
        ax.plot([x1, x2], [y1, y2], color=GREY, lw=0.4 + w / 6,
                alpha=0.25 if cut else 0.6, ls="--" if cut else "-", zorder=1)
    for i in range(p.n):
        contained = show_plan and x[i]
        ax.scatter(*p.pos[i], s=250 + 9 * p.r[i], zorder=2,
                   color=GREEN if contained else RED, alpha=0.9 if contained else 0.75,
                   edgecolors="black" if contained else "none", linewidths=3 if contained else 0)
        ax.text(p.pos[i][0], p.pos[i][1] - 0.05 - np.sqrt(250 + 9 * p.r[i]) / 900, p.shorts[i], ha="center", fontsize=11,
                fontweight="bold" if contained else "normal")
        ax.text(*p.pos[i], "TEAM" if contained else f"{p.r[i]:.0f}", ha="center",
                va="center", fontsize=8, color="white", fontweight="bold", zorder=3)
    ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.02, 1.0)
    ax.set_title(title, fontsize=14); ax.axis("off")


def hero_figure(p, x, fin, path, plan_source="QAOA"):
    """Scenario hero image: do-nothing vs FinShield plan with the money numbers."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    _draw_network(axes[0], p, x, f"Do nothing: expected loss {fin['loss_do_nothing']:.0f}", False)
    _draw_network(axes[1], p, x, f"FinShield plan ({plan_source}): {p.K} containment teams", True)
    box = (f"Loss avoided      {fin['loss_avoided']:7.1f}\n"
           f"Disruption cost   {fin['disruption_cost']:7.1f}\n"
           f"NET BENEFIT       {fin['net_benefit']:7.1f}")
    axes[1].text(0.5, -0.1, box, transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=14, family="monospace",
                 bbox=dict(boxstyle="round", fc="#eef7f0", ec=GREEN))
    fig.suptitle(f"Scenario {p.label}: {p.title}   (all values {p.currency}; red number = node risk)",
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.005, "SYNTHETIC, ILLUSTRATIVE DATA. Node size = risk; line width = contagion strength.",
             ha="center", fontsize=10, color=GREY)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_p_curves(per_p, random_ref, path, title):
    """per_p: {p: {'p_optimal': (mean, std), ...}}; random_ref: {'p_optimal': v, ...}."""
    ps = sorted(per_p)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    spec = [("p_optimal", "P(sample the optimal plan)"),
            ("p_feasible", "P(sample any feasible plan)"),
            ("q_mean_feasible", "Mean quality of feasible samples (0-1)")]
    for ax, (key, lab) in zip(axes, spec):
        mean = [per_p[q][key][0] for q in ps]
        std = [per_p[q][key][1] for q in ps]
        ax.errorbar(ps, mean, yerr=std, marker="o", lw=2, capsize=5, color=BLUE, label="QAOA")
        ax.axhline(random_ref[key], color=RED, ls="--", label="uniform random sampling")
        ax.set_xticks(ps); ax.set_xlabel("QAOA depth p", fontsize=12); ax.set_title(lab, fontsize=12)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_methods(rows, path):
    """rows: list of dicts {scenario, method, net_benefit, runtime_s}."""
    scen = sorted({r["scenario"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    width = 0.8 / len(methods)
    for mi, m in enumerate(methods):
        vals = [next(r["net_benefit"] for r in rows if r["scenario"] == s and r["method"] == m)
                for s in scen]
        times = [next(r["runtime_s"] for r in rows if r["scenario"] == s and r["method"] == m)
                 for s in scen]
        xs = np.arange(len(scen)) + mi * width
        axes[0].bar(xs, vals, width, label=m)
        axes[1].bar(xs, times, width, label=m)
    for ax in axes:
        ax.set_xticks(np.arange(len(scen)) + 0.4 - width / 2)
        ax.set_xticklabels([f"Scenario {s}" for s in scen], fontsize=12)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_title("Net benefit of the recommended plan (synthetic USD m)")
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.15); axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_yscale("log"); axes[1].set_title("Runtime in seconds (log scale)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_penalty_sweep(sweep, a_min, a_safe, a_used, path, p_label):
    """sweep: list of dicts with A, p_optimal, p_feasible, q_mean_feasible."""
    A = [s["A"] for s in sweep]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(A, [s["p_feasible"] for s in sweep], "o-", label="P(feasible plan)")
    ax.plot(A, [s["p_optimal"] for s in sweep], "s-", label="P(optimal plan)")
    ax.plot(A, [s["q_mean_feasible"] for s in sweep], "^-", label="mean quality of feasible samples")
    ax.axvline(a_min, color=RED, ls="--", label=f"smallest A that works = {a_min:.0f}")
    ax.axvline(a_used, color=GREEN, ls=":", label=f"A used in main runs = {a_used:.0f}")
    ax.set_xlabel("Penalty weight A", fontsize=12); ax.set_ylabel("value (0-1)", fontsize=12)
    ax.set_title(f"Penalty sweep, scenario A, QAOA {p_label}. (Provable safe bound = {a_safe:.0f})")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def architecture_diagram(path):
    """Pipeline diagram drawn with boxes and arrows; big fonts for the video."""
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color,
                                    ec="black", lw=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=14)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=28,
                                     lw=2.5, color="#333"))

    box(0.4, 6.2, 3.6, 1.8, "Scenario data\n(synthetic JSON config:\n10 nodes, links, risks,\n3 attack scenarios)", "#fdebd0")
    box(4.8, 6.2, 3.2, 1.8, "QUBO builder\n(qiskit-optimization\nQuadraticProgram)", "#d6eaf8")
    box(8.8, 6.2, 2.9, 1.8, "Ising Hamiltonian\n(10 qubits)", "#d6eaf8")
    box(12.2, 6.2, 3.4, 1.8, "QAOA loop, p = 1, 2, 3\ncircuit on Aer simulator\n+ COBYLA tunes angles", "#d5f5e3")
    box(12.2, 3.4, 3.4, 1.8, "Containment plan\n(which nodes get\nthe K teams)", "#d5f5e3")
    box(6.0, 3.4, 5.2, 1.8, "Comparison:\nbrute force (exact), greedy,\nsimulated annealing", "#fadbd8")
    box(0.4, 3.4, 4.6, 1.8, "Finance output:\nloss avoided, disruption cost,\nnet benefit (synthetic USD m)", "#fdebd0")
    box(3.0, 0.5, 10.0, 1.8, "Metrics and plots: P(optimal), P(feasible), quality, cost gap, runtime,\nnetwork hero figures, p curves, penalty sweep", "#e8daef")
    arrow(4.0, 7.1, 4.8, 7.1); arrow(8.0, 7.1, 8.8, 7.1); arrow(11.7, 7.1, 12.2, 7.1)
    arrow(13.9, 6.2, 13.9, 5.2)
    arrow(12.2, 4.3, 11.2, 4.3); arrow(6.0, 4.3, 5.0, 4.3)
    arrow(8.6, 3.4, 8.0, 2.3)
    ax.text(8, 8.65, "FinShield architecture", ha="center", fontsize=22, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
