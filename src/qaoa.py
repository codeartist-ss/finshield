"""Hand-written QAOA on the Aer simulator (the README's 'fallback path').

Why not qiskit_optimization's QAOA class: we want full control over (a) the
shots used while tuning vs. the final sampling, (b) per-restart records, and
(c) a plain, explainable circuit. The circuit follows the standard QAOA recipe
described in the Qiskit documentation / textbook: Hadamards, then p times
[cost layer exp(-i*gamma*H_C), RX mixer exp(-i*beta*sum X)].

Ising Hamiltonian terms come from qiskit_optimization's to_ising().
"""
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .qubo import build_quadratic_program, qubo_energy_table, to_ising_hamiltonian


def ising_terms(op, n):
    """Split a SparsePauliOp into {qubit: coeff} (Z) and {(a, b): coeff} (ZZ)."""
    lin, quad = {}, {}
    for label, coeff in zip(op.paulis.to_labels(), op.coeffs):
        qubits = [n - 1 - k for k, ch in enumerate(label) if ch == "Z"]  # label is little-endian
        c = float(np.real(coeff))
        if len(qubits) == 1:
            lin[qubits[0]] = lin.get(qubits[0], 0.0) + c
        elif len(qubits) == 2:
            quad[tuple(sorted(qubits))] = quad.get(tuple(sorted(qubits)), 0.0) + c
    return lin, quad


def build_circuit(n, lin, quad, p):
    """Parametrised QAOA circuit with p layers. Coefficients are pre-scaled."""
    gam, bet = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for q, c in lin.items():
            qc.rz(2 * gam[layer] * c, q)            # exp(-i*gamma*c*Z)
        for (a, b), c in quad.items():
            qc.rzz(2 * gam[layer] * c, a, b)        # exp(-i*gamma*c*ZZ)
        qc.rx(2 * bet[layer], range(n))             # mixer
    qc.measure_all()
    return qc, gam, bet


class QAOASolver:
    """Prepares the QUBO/Ising for one instance; run() tunes and samples."""

    def __init__(self, params, backend=None):
        self.p = params
        self.n = params.n
        self.qp = build_quadratic_program(params)
        self.energies = qubo_energy_table(self.qp)          # QUBO energy per bitstring
        op, self.offset = to_ising_hamiltonian(self.qp)
        lin, quad = ising_terms(op, self.n)
        # Divide by the largest coefficient so one set of angle ranges works for
        # every instance. A positive scale does not move the minimiser.
        self.scale = max([abs(v) for v in list(lin.values()) + list(quad.values())])
        self.lin = {k: v / self.scale for k, v in lin.items()}
        self.quad = {k: v / self.scale for k, v in quad.items()}
        self.backend = backend or AerSimulator(method="statevector")
        self._cache = {}

    def _circuit(self, p):
        if p not in self._cache:
            qc, gam, bet = build_circuit(self.n, self.lin, self.quad, p)
            self._cache[p] = (transpile(qc, self.backend, optimization_level=1), gam, bet)
        return self._cache[p]

    def _counts(self, theta, p, shots, seed):
        qc, gam, bet = self._circuit(p)
        bound = qc.assign_parameters({**dict(zip(gam, theta[:p])), **dict(zip(bet, theta[p:]))})
        res = self.backend.run(bound, shots=shots, seed_simulator=seed).result()
        # Qiskit bitstrings are little-endian: int(key, 2) has qubit 0 as bit 0.
        return {int(k.replace(" ", ""), 2): v for k, v in res.get_counts().items()}

    def _expected_energy(self, counts):
        total = sum(counts.values())
        return sum(self.energies[k] * v for k, v in counts.items()) / total

    def run_once(self, p, seed, shots_tune=1000, shots_final=8192, maxiter=80):
        """One random start: COBYLA tunes 2p angles, then a final big sample."""
        rng = np.random.default_rng(seed)
        theta0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
        evals = [0]

        def objective(theta):
            evals[0] += 1
            # Same simulator seed every call -> the objective is deterministic,
            # which COBYLA needs; the price is a fixed shot-noise pattern.
            return self._expected_energy(self._counts(theta, p, shots_tune, seed))

        t0 = time.time()
        res = minimize(objective, theta0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.4})
        counts = self._counts(res.x, p, shots_final, seed + 10_000)
        return {"p": p, "seed": seed, "theta": res.x, "counts": counts,
                "final_energy": self._expected_energy(counts),
                "evals": evals[0], "shots_tune": shots_tune,
                "shots_final": shots_final, "runtime_s": time.time() - t0}

    def run(self, p, restarts=5, seed=0, **kw):
        """Several restarts; 'best' = lowest final expected energy (no oracle use)."""
        runs = [self.run_once(p, seed + 100 * r + p, **kw) for r in range(restarts)]
        best = min(runs, key=lambda r: r["final_energy"])
        return {"best": best, "all": runs,
                "runtime_s": sum(r["runtime_s"] for r in runs)}
