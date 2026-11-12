# Results summary (for writing the .docx report)

All data is **synthetic**; money is in **synthetic USD millions**. Simulator only (Qiskit Aer). Numbers below come from `python run_all.py --full` (seeds fixed). Re-run on your machine and update any number that differs slightly. Package versions: qiskit 2.6.2, qiskit-aer 0.18.2, qiskit-optimization 0.7.0.

## Setup in one paragraph
10 nodes = 10 qubits, K = 3 teams, lambda = 0.5, mu = 1.0, penalty A = 100 (smallest working A: 58 / 61 / 66 for scenarios A / B / C; provable safe bound 236 / 257 / 324). QAOA p = 1, 2, 3, COBYLA, 8 random restarts per p, 1,000 shots per evaluation, 8,192 final shots. Hand-written QAOA circuit (see README for why).

## Finance results (exact optimum; files: `scenario_finance.csv`, `hero_scenario_A/B/C.png`)

| Scenario | Do-nothing loss | Teams go to | Loss avoided | Disruption cost | Net benefit |
|---|---|---|---|---|---|
| A Cloud provider compromise | 757.0 | Payment Switch, Clearing House, Cloud Provider | 438.0 | 73.0 | 365.0 |
| B Payment switch DDoS | 840.5 | Payment Switch, Card Processor, Clearing House | 437.5 | 58.0 | 379.5 |
| C Clearing house ransomware | 891.0 | RTGS, Commercial Bank A, Clearing House | 579.5 | 90.0 | 489.5 |

In scenario C there is a tie: QAOA's plan (RTGS, Payment Switch, Clearing House) has the same net benefit, 489.5 (loss avoided 582.5, disruption 93.0). Greedy, simulated annealing and QAOA's best sampled plan reached the exact optimal cost in all three scenarios.

## QAOA metrics, main network (`scenario_qaoa_metrics.csv`)

| Scen. | p | P(optimal) | P(feasible) | Mean quality of feasible | Approx. ratio | Runtime, all restarts |
|---|---|---|---|---|---|---|
| A | 1 | 0.15% | 12.2% | 0.38 | 0.852 | 1.4 s |
| A | 2 | 0.70% | 61.4% | 0.44 | 0.957 | 2.2 s |
| A | 3 | 0.65% | 60.4% | 0.43 | 0.940 | 3.8 s |
| B | 1 | 0.15% | 11.8% | 0.40 | 0.855 | 1.4 s |
| B | 2 | 0.40% | 37.5% | 0.42 | 0.954 | 2.2 s |
| B | 3 | 0.57% | 34.8% | 0.52 | 0.940 | 3.7 s |
| C | 1 | 0.10% | 2.3% | 0.51 | 0.845 | 1.3 s |
| C | 2 | 1.26% | 52.8% | 0.44 | 0.940 | 2.4 s |
| C | 3 | 1.69% | 37.5% | 0.48 | 0.897 | 3.3 s |

Uniform random sampling: P(optimal) about 0.07-0.16% (it depends on the sample), P(feasible) 11.6%. The best sampled feasible plan had quality 1.0 and cost gap 0 in every row, **but random sampling also achieves that with 8,192 shots**, so this is not evidence of quantum advantage. Best restart is selected by lowest final expected energy, not by the known optimum. Mean COBYLA evaluations per restart: 27 (p=1), 43 (p=2), 56 (p=3); COBYLA stopped on its tolerance well before the 200-iteration cap.

## 20 random synthetic instances (`random_instances_summary.csv`, plot `p_vs_success_quality.png`)

Same size (10 nodes, K = 3), fixed seeds 100-119, penalty = 1.5 x the smallest working A for each instance (61.5-90).

| p | P(optimal) mean | P(feasible) mean | Mean quality of feasible | Best-sampled quality | Approx. ratio |
|---|---|---|---|---|---|
| 1 | 0.12% | 15.6% | 0.500 | 1.000 | 0.887 |
| 2 | 0.47% | 25.9% | 0.538 | 1.000 | 0.936 |
| 3 | 0.65% | 25.3% | 0.496 | 0.995 | 0.919 |
| Uniform random | 0.10% | 11.8% | 0.499 | 1.000 | 0.868 |

Instances where P(optimal) beat random sampling: p=1 in 6 of 20, p=2 in 18 of 20, p=3 in 14 of 20. Greedy and simulated annealing found the exact optimum in 20 of 20 instances.

## Method comparison (`methods_comparison.png`)
Brute force about 0.014 s, greedy about 0.0004 s, simulated annealing about 0.3 s, QAOA 1.3-3.8 s (all restarts, simulator). All classical methods matched the exact optimum. **Brute force wins at 10 qubits; no quantum speedup is claimed.**

## Penalty sweep (`penalty_sweep.csv`, `penalty_sweep.png`)
Scenario A, A from 0.5x to 8x the smallest working penalty (29 to 464), p = 1, 2, 3, 4 restarts each. Below 58 the QUBO's own optimum is not a K-team plan, so low-A results are not meaningful for finding the right answer. Above that there is **no clean trend**: for p = 2, P(feasible) ranged about 0.24-0.51 and P(optimal) about 0.2-0.5%, and p = 3 was erratic (for example P(feasible) 5% at 6x). Largest A tested (464) gave the highest P(feasible) at p = 2 (51%), but a huge penalty is not free: it shrinks the relative size of the real finance terms. Treat the sweep as noisy.

## What did not work / limitations (real observations)
1. No speed or quality advantage over classical methods at this size; classical ones were exact every time.
2. QAOA success probability for the optimal plan stayed low (at most about 2% in any run).
3. At p = 1 QAOA was barely better than random sampling (P(feasible) 12.2% vs 11.6% in scenario A; 2.3% in scenario C, worse than random).
4. Mean quality of feasible samples was about the same as random sampling.
5. p = 3 was not reliably better than p = 2; results varied between restarts, and between instances (std across instances in the plot).
6. COBYLA stopped early (about 27-56 evaluations); with a fixed shot seed the objective has a fixed noise pattern, so some convergence may be to noise-shaped local minima. We did not test other optimizers.
7. The penalty sweep is noisy (few restarts), and the penalty must be chosen carefully.
8. Simulator only; the optional noisy-simulator run was not done.
9. Synthetic data, scenario numbers adjusted once for distinct plans before any QAOA run. K = 3 fixed by design.
