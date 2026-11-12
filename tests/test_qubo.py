"""Run with:  python -m unittest discover -s tests -v"""
import unittest

import numpy as np

from src.baselines import brute_force, greedy
from src.config import build_params, load_config, scenario_names
from src.qubo import (all_bitstrings, build_quadratic_program, min_penalty,
                      original_objective, qubo_energy_table, to_ising_hamiltonian)


class TestQubo(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_qubo_equals_original_for_all_1024_bitstrings(self):
        for s in scenario_names(self.cfg):
            p = build_params(self.cfg, s)
            qp = build_quadratic_program(p)
            fast = qubo_energy_table(qp)
            for k, x in enumerate(all_bitstrings(p.n)):
                want = original_objective(p, x)
                self.assertAlmostEqual(qp.objective.evaluate(x), want, places=6)
                self.assertAlmostEqual(fast[k], want, places=6)

    def test_ising_energy_equals_qubo(self):
        p = build_params(self.cfg, "A")
        qp = build_quadratic_program(p)
        op, offset = to_ising_hamiltonian(qp)
        diag = np.real(np.diag(op.to_matrix())) + offset   # <H> + offset per basis state
        np.testing.assert_allclose(diag, qubo_energy_table(qp), atol=1e-6)

    def test_exact_optimum_has_exactly_K_teams(self):
        for s in scenario_names(self.cfg):
            p = build_params(self.cfg, s)
            # global minimum of the penalised QUBO over ALL 1,024 plans
            energies = qubo_energy_table(build_quadratic_program(p))
            best = all_bitstrings(p.n)[int(np.argmin(energies))]
            self.assertEqual(int(best.sum()), p.K)
            # and it matches the constrained brute force
            self.assertTrue(np.array_equal(best, brute_force(p)["x"]))

    def test_config_penalty_has_headroom_over_the_minimum_needed(self):
        # min_penalty = smallest A whose global optimum has exactly K teams.
        for s in scenario_names(self.cfg):
            p = build_params(self.cfg, s)
            self.assertGreaterEqual(p.A, 1.3 * min_penalty(p))

    def test_greedy_respects_K_and_is_not_better_than_exact(self):
        for s in scenario_names(self.cfg):
            p = build_params(self.cfg, s)
            g, bf = greedy(p), brute_force(p)
            self.assertEqual(int(g["x"].sum()), p.K)
            self.assertGreaterEqual(g["cost"], bf["cost"] - 1e-9)


if __name__ == "__main__":
    unittest.main()
