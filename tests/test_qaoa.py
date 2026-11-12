"""QAOA plumbing tests (bit-order and Hamiltonian consistency)."""
import unittest

import numpy as np
from qiskit.quantum_info import Statevector

from src.config import build_params, load_config
from src.qaoa import QAOASolver, build_circuit


class TestQaoa(unittest.TestCase):
    def setUp(self):
        self.p = build_params(load_config(), "A")
        self.solver = QAOASolver(self.p)

    def test_circuit_distribution_matches_energy_table(self):
        """<H> from the statevector must equal sum_k prob_k * energy_k.

        This catches Qiskit little-endian mistakes: if qubit order were wrong the
        two numbers would differ.
        """
        s = self.solver
        qc, gam, bet = build_circuit(s.n, s.lin, s.quad, 2)
        qc.remove_final_measurements()
        theta = np.array([0.4, 0.7, 0.3, 0.2])
        bound = qc.assign_parameters({**dict(zip(gam, theta[:2])), **dict(zip(bet, theta[2:]))})
        sv = Statevector(bound)
        probs = sv.probabilities()                      # index k <-> bitstring int k
        from_table = float(probs @ s.energies)
        op, offset = __import__("src.qubo", fromlist=["x"]).to_ising_hamiltonian(s.qp)
        from_operator = float(np.real(sv.expectation_value(op))) + offset
        self.assertAlmostEqual(from_table, from_operator, places=6)

    def test_sampled_counts_match_statevector(self):
        s = self.solver
        theta = np.array([0.4, 0.3])
        counts = s._counts(theta, 1, 200_000, seed=5)
        sampled = s._expected_energy(counts)
        qc, gam, bet = build_circuit(s.n, s.lin, s.quad, 1)
        qc.remove_final_measurements()
        sv = Statevector(qc.assign_parameters({gam[0]: 0.4, bet[0]: 0.3}))
        exact = float(sv.probabilities() @ s.energies)
        self.assertLess(abs(sampled - exact), 0.01 * np.ptp(s.energies))


if __name__ == "__main__":
    unittest.main()
