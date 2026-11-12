# FinShield explained (for the team, in plain English)

## Glossary

* **Qubit**: the quantum version of a bit. A normal bit is 0 or 1. A qubit can be in a blend of both. We use 10 qubits, one per network node.
* **Superposition**: that blend. Ten qubits in superposition represent all 1,024 possible plans at once, each with a "weight" (amplitude).
* **Measurement**: looking at the qubits. It forces each to a plain 0 or 1, giving one concrete plan, picked at random according to the weights. We repeat this many times ("shots") and count what we see.
* **QUBO** (Quadratic Unconstrained Binary Optimization): a cost made of yes/no variables where every term involves one variable or a pair. Ours: x_i = 1 means node i gets a containment team.
* **Ising Hamiltonian**: the same cost written with Z operators (values +1 or -1) instead of 0/1 variables, which is the form a quantum circuit can use. The conversion is x = (1 - Z)/2. A unit test checks both give the same number for all 1,024 plans.
* **QAOA** (Quantum Approximate Optimization Algorithm): start with all plans equally weighted; repeat p times (a "cost layer" that tilts weights by how good each plan is, then a "mixer" that spreads weight between neighbouring plans); measure. A normal computer tunes the 2p angles to make low-cost plans likely.
* **p (depth)**: how many cost+mixer rounds. More rounds can express more but are harder to tune.
* **Approximation ratio**: how close the average sampled cost is to the best possible, scaled so 1 is perfect and 0 is the worst. We use (Emax - average)/(Emax - Emin).
* **Feasible plan**: a plan using exactly K = 3 teams.
* **Penalty weight A**: makes plans with the wrong number of teams expensive.

## Pipeline in one page

1. **Data** (`config/finshield_config.json`): 10 nodes with risk and disruption cost, 27 links with contagion strength, 3 scenarios. All synthetic.
2. **QUBO** (`src/qubo.py`): expand the finance formula into constant + linear + pairwise terms, stored in a Qiskit `QuadraticProgram`. A test compares it to the original formula for all 1,024 plans.
3. **Ising** (`to_ising`): turn the QUBO into a Hamiltonian on 10 qubits.
4. **QAOA** (`src/qaoa.py`): build the circuit (H, then p times RZ/RZZ cost layer and RX mixer), run on the Aer simulator with 1,000 shots per evaluation, let COBYLA adjust the angles, then sample 8,192 shots at the best angles. Repeat for several random starts and keep the one with the lowest average cost.
5. **Plan**: the cheapest feasible plan among the samples.
6. **Comparison** (`src/baselines.py`): brute force (exact), greedy, simulated annealing.
7. **Metrics and plots**: probabilities, quality, cost gap, runtime, and money terms (loss avoided, disruption cost, net benefit).

## Ten likely judge questions

1. **Is this really Disaster Response?** Yes in structure: cascading failure across interdependent critical infrastructure, scarce response teams, and a first-hour allocation decision, the same shape as disaster logistics. The "disaster" is a cyber incident in the financial system.
2. **Why quantum if brute force wins?** It does win at 10 nodes (about 0.014 s). We do not claim a speedup. The point is a working, verified QAOA formulation on a problem whose size grows exponentially (2^N plans), and an honest measurement of how QAOA behaves. Whether it helps at large scale on real hardware is an open question we did not test.
3. **Isn't this just vertex cover?** It is a budgeted, weighted variant of it, with a disruption cost and risk values. That is deliberate: vertex cover is a standard NP-hard problem that maps cleanly to a QUBO.
4. **Where does the data come from?** We invented it. It is synthetic and illustrative; no real institution, statistic or attack is described. You can edit every number in one config file.
5. **Why do results change with p?** More layers give the circuit more freedom, but there are more angles to tune and the optimizer can get stuck. In our runs p = 2 was usually better than p = 1, and p = 3 was not reliably better than p = 2.
6. **Did the quantum part do real work?** The circuit is the sampler: plans come from measuring it. It puts more probability on feasible and good plans than uniform random sampling does (for example p = 2 beat random on P(optimal) in 18 of 20 random instances), but the absolute success probability is small (under about 2%).
7. **How do you know the QUBO is right?** A unit test checks the QUBO equals the original formula for all 1,024 plans, that the Ising Hamiltonian gives the same energies, and that the exact optimum has exactly K teams.
8. **Why is the penalty A set to 100?** It must be big enough that the best plan uses exactly K teams. The smallest A that works is 58-66 for our scenarios, so we used about 1.5x headroom. A provable safe bound is 236-324, but huge penalties make QAOA harder. The penalty sweep shows how results move.
9. **Did you tune things to make QAOA look good?** No. Seeds and settings were fixed, the best restart is chosen without using the known optimum, and the README reports the weak spots. The one adjustment was to the synthetic scenario numbers, made to give distinct plans per scenario before QAOA was run.
10. **Is `q_best = 1.0` impressive?** Not by itself: with 8,192 shots over 1,024 plans, random sampling also finds the best plan. That is why we report probabilities and compare against a random-sampling reference.
