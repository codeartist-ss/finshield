# FinShield: quantum-assisted containment planning for cyber attacks on financial infrastructure

*Qiskit Fall Fest 2026 Hackathon. Team: [TEAM NAME]. Domain: Disaster Response.*

**All data in this repository is synthetic and illustrative. No real institution, system, statistic or attacker is described. All money is in "synthetic USD millions". Everything runs on a classical simulator (Qiskit Aer); no real quantum hardware was used.**

## The problem, who it serves, and why this is Disaster Response

A coordinated cyber attack on financial infrastructure spreads like a cascading disaster: through interbank links, shared cloud providers and payment rails. A crisis unit has only **K = 3 containment teams** for **N = 10 critical nodes** (central bank RTGS, three commercial banks, payment switch, card processor, clearing house, cloud provider, ATM network, mobile wallet gateway). Containing a node stops the attack spreading through it but disrupts its service. *Where should the teams go in the first hour?* FinShield answers this and reports the result in money: expected loss avoided, disruption cost, and net benefit compared with doing nothing. It is defensive and resilience-focused, and names no real attacker.

It is built for central-bank crisis units, financial-sector CERTs and payment-system operators.

**Why Disaster Response:** it is the same structure as disaster logistics: emergency response under cascading failure, scarce response teams, interdependent critical infrastructure, and a one-shot first-hour allocation decision. Instead of placing rescue teams across a flooded region, we place containment teams across a network of dependent financial services.

## QUBO and QAOA in plain English

* Each node gets a yes/no variable `x_i` (1 = a containment team goes there). Ten nodes means ten yes/no choices, so 2^10 = 1,024 possible plans.
* We write the "badness" of a plan as one number (a cost) that adds up residual risk, contagion between neighbours that are both unprotected, disruption from containing, and a penalty if we do not use exactly K teams. A cost that is a sum of single and pairwise yes/no terms is called a **QUBO**. It maps one-to-one onto the **Ising Hamiltonian** that a quantum computer works with, using 10 qubits.
* **QAOA** prepares a quantum state on those qubits, applies p alternating "cost" and "mixing" layers whose 2p angles are tuned by a classical optimizer (COBYLA), then measures. Good plans should be measured more often than bad ones. See `docs/EXPLAINED.md` for the glossary.

The cost we minimize:

```
C(x) = sum_i r_i (1 - x_i)                         residual risk
     + lambda * sum_{i<j} w_ij (1 - x_i)(1 - x_j)  contagion between unprotected neighbours
     + mu * sum_i d_i x_i                          disruption cost of containing
     + A * (sum_i x_i - K)^2                       team-limit penalty
```

It is a budgeted weighted vertex-cover variant (NP-hard in general). Three attack scenarios use the same network; only the risks `r_i` change: **A** cloud provider compromise, **B** payment switch DDoS, **C** clearing house ransomware. Story texts live in `config/finshield_config.json`.

## Setup (Windows CMD)

```
python -m venv finshield-env
finshield-env\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Always run Python from the activated `finshield-env`.

## Reproduce each result

| What | Command | Output |
|---|---|---|
| Unit tests (QUBO equals the original objective for all 1,024 bitstrings, Ising matches QUBO, exact optimum has exactly K teams, qubit order check) | `python -m unittest discover -s tests -v` | console |
| Classical pipeline only (exact + greedy, finance output) | `python run_classical.py` | console |
| 3-scenario demo for the video (about 15 s) | `python run_demo.py` (or `--scenario A`) | console + `results/hero_scenario_*.png` |
| Every result, quick (about 30 s) | `python run_all.py --quick` | `results/`, `docs/architecture.png` |
| Every result, full (about 2.5 min in our run) | `python run_all.py --full` | `results/`, `docs/architecture.png` |

Seeds are fixed everywhere (numpy and the Aer simulator). Numbers can still differ slightly on another machine or package version; see `results/SUMMARY.md`.

## Which QAOA path is used

**We wrote QAOA ourselves** (`src/qaoa.py`): Hadamards, then p times [cost layer from the Ising terms (`RZ` and `RZZ` gates), `RX` mixer], sampled on `AerSimulator`, with SciPy COBYLA tuning the angles. We did **not** use the high-level `QAOA` class that ships in `qiskit_optimization.minimum_eigensolvers`, because we wanted full control of shots, per-restart records and the final sample distribution. The QUBO is built with `qiskit_optimization.QuadraticProgram` and converted with `qiskit_optimization.translators.to_ising`. We use `AerSimulator.run` directly, not `SamplerV2`. The Hamiltonian coefficients are divided by the largest coefficient so one angle range suits every instance (a positive scale does not change the minimizer).

Settings: p = 1, 2, 3; several random restarts per p (8 on the main network in `--full`); COBYLA with up to 200 iterations; 1,000 shots per objective evaluation and 8,192 shots for the final sample; the best restart is chosen by lowest final expected energy (it does not peek at the known optimum).

## Metrics (definitions)

* **P(optimal)**: fraction of final shots that are an optimal feasible plan. Ties count (two different plans with identical cost are both optimal; this happens in scenario C).
* **P(feasible)**: fraction of final shots with exactly K teams. Uniform random guessing gives C(10,3)/1024 = 11.7%.
* **Quality** = (Cmax - C) / (Cmax - Cmin), where Cmin and Cmax are the best and worst cost over all feasible plans (penalty-free cost C), so 1 is optimal and 0 is the worst feasible plan. We report it for (i) the best feasible plan sampled (`q_best`) and (ii) the average over feasible samples (`q_mean_feasible`). If no feasible plan is sampled, quality is set to 0 and flagged.
* **Cost gap**: C(best sampled feasible plan) minus C(exact optimum), in synthetic USD m.
* **Approximation ratio**: (Emax - mean sampled QUBO energy) / (Emax - Emin) over the whole 1,024-plan penalized landscape. 1 is perfect, 0 is the worst.
* **Runtime**: wall-clock seconds, all restarts included.
* **Reference**: every table also reports *uniform random sampling* with the same 8,192 shots. Because 8,192 shots cover 1,024 plans, random sampling also finds the optimum at least once, so `q_best` is about 1 for any sampler. Judge QAOA by the probabilities and the mean quality, not by `q_best` alone.
* **Finance outputs**: loss avoided = (do-nothing loss) - (residual risk + contagion with the plan); disruption cost = mu * sum d_i x_i; net benefit = loss avoided - disruption cost.

## Repository structure

```
config/finshield_config.json   ONE editable file: nodes, links, numbers, scenarios, narratives
src/config.py                  loads config -> numbers
src/qubo.py                    QUBO (QuadraticProgram), Ising, penalty helpers
src/qaoa.py                    hand-written QAOA on Aer + COBYLA
src/baselines.py               brute force, greedy, simulated annealing
src/metrics.py                 finance output + sample metrics + random-sampling reference
src/experiments.py             drivers, random instance generator
src/plotting.py                all figures, architecture diagram
run_classical.py  run_demo.py  run_all.py
tests/                         unit tests (python -m unittest)
results/                       CSV + PNG + SUMMARY.md
docs/                          architecture.png, EXPLAINED.md
```

## Honest results and limitations

* **No quantum speedup, and none claimed.** Brute force over 1,024 plans takes about 0.014 s. Greedy and our simulated annealing found the exact optimum on all 3 scenarios and all 20 random instances. At this size classical methods win.
* QAOA does put more probability on good plans than random sampling (p = 2 beat random on P(optimal) in 18 of 20 random instances), but the absolute probability of sampling the optimum stays below about 2% in every run. The *mean quality of feasible samples* is about the same as random sampling. See `results/SUMMARY.md`.
* Depth is not monotone: p = 3 was not reliably better than p = 2 (COBYLA local minima, shot noise, our modest restart budget).
* The penalty weight matters and the sweep is noisy; we did not tune A to flatter QAOA.
* Simulator only, no hardware noise. A noisy-simulator run was **not done** (cut for time).
* Synthetic numbers. We adjusted the synthetic numbers once (before any QAOA runs) so the three scenarios give visibly different plans; the greedy and exact answers were identical either way.
* Containing exactly K nodes is a modelling choice; using fewer teams could be cheaper.

## Credits

* Qiskit, Qiskit Aer and Qiskit Optimization (Apache-2.0). Our QAOA circuit follows the standard recipe described in the Qiskit documentation and textbook (Hadamard start, cost layer, RX mixer); the code is our own. The QUBO-to-Ising conversion uses `qiskit_optimization.translators.to_ising`. The QUBO expansion, vertex-cover-style formulation, penalty method and simulated annealing follow textbook ideas, written by us.
* SciPy (COBYLA), NumPy, Matplotlib.

## AI assistance (please confirm wording)

An AI assistant (Claude, by Anthropic) helped write the code, tests and documents under the team's direction, inspected the installed Qiskit packages, and ran the code and tests in its own sandbox. The team chose the problem, the formulation and the scenarios, and reviewed and ran the result. [TEAM: edit this paragraph to say exactly what you did yourselves.]
