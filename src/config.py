"""Load the editable JSON config and turn one scenario into plain numbers.

Why a small dataclass: every other module only needs numbers (r, W, d, K...),
so they never touch JSON or scenario names.
"""
import json
import os
from dataclasses import dataclass, field, replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "finshield_config.json")


@dataclass
class Params:
    names: list          # long node names
    shorts: list         # short labels for plots
    pos: list            # (x, y) for network drawings
    r: np.ndarray        # risk / exposure of each node (synthetic USD m)
    W: np.ndarray        # symmetric n x n contagion matrix (0 where no edge)
    d: np.ndarray        # disruption cost of containing each node
    K: int               # number of containment teams
    lam: float           # weight of contagion term
    mu: float            # weight of disruption term
    A: float             # team-limit penalty weight
    title: str = ""
    narrative: str = ""
    label: str = ""
    currency: str = "synthetic USD millions"
    edges: list = field(default_factory=list)  # (i, j, w) with i < j

    @property
    def n(self):
        return len(self.r)

    def with_A(self, A):
        return replace(self, A=float(A))


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def scenario_names(cfg):
    return list(cfg["scenarios"].keys())


def build_params(cfg, scenario="A"):
    """Return Params for one scenario ('A', 'B' or 'C')."""
    nodes = sorted(cfg["nodes"], key=lambda nd: nd["id"])
    n = len(nodes)
    sc = cfg["scenarios"][scenario]
    mult = {int(k): float(v) for k, v in sc.get("multipliers", {}).items()}
    r = np.array([nd["base_risk"] * mult.get(nd["id"], 1.0) for nd in nodes], dtype=float)
    d = np.array([nd["disruption"] for nd in nodes], dtype=float)
    W = np.zeros((n, n))
    edges = []
    for i, j, w in cfg["edges"]:
        a, b = min(i, j), max(i, j)
        W[a, b] = W[b, a] = float(w)
        edges.append((a, b, float(w)))
    m = cfg["model"]
    return Params(
        names=[nd["name"] for nd in nodes],
        shorts=[nd["short"] for nd in nodes],
        pos=[tuple(nd["pos"]) for nd in nodes],
        r=r, W=W, d=d, K=int(m["K_teams"]),
        lam=float(m["lambda_contagion"]), mu=float(m["mu_disruption"]),
        A=float(m["penalty_A"]),
        title=sc["title"], narrative=sc["narrative"], label=scenario,
        currency=cfg.get("currency_label", "synthetic USD millions"),
        edges=sorted(edges),
    )
