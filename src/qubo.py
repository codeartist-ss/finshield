"""Build the containment QUBO with qiskit_optimization and convert to Ising.

Objective (minimise), x_i = 1 means node i gets a containment team:
    C(x) = sum_i r_i (1-x_i)                      residual risk
         + lam * sum_{i<j} w_ij (1-x_i)(1-x_j)    contagion between unprotected neighbours
         + mu  * sum_i d_i x_i                    disruption cost of containing
         + A   * (sum_i x_i - K)^2                team-limit penalty

Expansion used (x^2 = x for binary variables):
    r_i (1-x_i)               = r_i - r_i x_i
    w (1-x_i)(1-x_j)          = w - w x_i - w x_j + w x_i x_j
    A (sum x - K)^2           = A K^2 + A (1-2K) sum x_i + 2A sum_{i<j} x_i x_j
"""
import itertools

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising


def original_objective(p, x, with_penalty=True):
    """Direct, un-expanded formula. Used to check the QUBO expansion is right."""
    x = np.asarray(x, dtype=float)
    total = float(np.sum(p.r * (1 - x)))
    for i, j, w in p.edges:
        total += p.lam * w * (1 - x[i]) * (1 - x[j])
    total += p.mu * float(np.sum(p.d * x))
    if with_penalty:
        total += p.A * (x.sum() - p.K) ** 2
    return total


def build_quadratic_program(p):
    """Return the QUBO as a qiskit_optimization QuadraticProgram."""
    n = p.n
    constant = float(p.r.sum() + p.A * p.K ** 2)
    linear = -p.r + p.mu * p.d + p.A * (1 - 2 * p.K)
    quadratic = {}
    for i, j, w in p.edges:
        constant += p.lam * w
        linear[i] -= p.lam * w
        linear[j] -= p.lam * w
        quadratic[(i, j)] = p.lam * w
    for i, j in itertools.combinations(range(n), 2):
        quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 2 * p.A

    qp = QuadraticProgram("finshield_containment")
    for i in range(n):
        qp.binary_var(f"x{i}")
    qp.minimize(constant=constant, linear=list(linear), quadratic=quadratic)
    return qp


def all_bitstrings(n):
    """Array (2^n, n); row k holds the bits of integer k, bit i = x_i (qubit i)."""
    k = np.arange(2 ** n)
    return ((k[:, None] >> np.arange(n)) & 1).astype(int)


def qubo_energy_table(qp):
    """QUBO energy of every bitstring, indexed by integer (bit i = x_i)."""
    n = qp.get_num_vars()
    X = all_bitstrings(n).astype(float)
    const = qp.objective.constant
    lin = qp.objective.linear.to_array()
    Q = qp.objective.quadratic.to_array()
    return const + X @ lin + np.einsum("ni,ij,nj->n", X, Q, X)


def to_ising_hamiltonian(qp):
    """QUBO -> (SparsePauliOp, offset). Energy = <H> + offset, with x = (1 - Z)/2."""
    return to_ising(qp)


def safe_penalty(p, margin=1.05):
    """A provably large-enough penalty.

    Flipping one bit changes the penalty-free cost C by at most
    b_i = max(r_i + lam*sum_j w_ij, mu*d_i). If an assignment has m too many or
    too few teams, m flips repair it and the penalty drops by A*m^2 >= A*m, so
    A > max_i b_i guarantees the optimum has exactly K teams.
    """
    b = np.maximum(p.r + p.lam * p.W.sum(axis=1), p.mu * p.d)
    return float(b.max() * margin)


def min_penalty(p, grid=None):
    """Smallest A (on a 1-unit grid) whose global QUBO minimum has exactly K teams.

    Found by brute force, so it is exact for this instance, but it is only
    cheap because n = 10. safe_penalty() is the provable (larger) bound.
    """
    X = all_bitstrings(p.n)
    cnt = X.sum(axis=1)
    C = np.array([original_objective(p, x, with_penalty=False) for x in X])
    for A in (grid if grid is not None else np.arange(1.0, 1000.0, 1.0)):
        if cnt[np.argmin(C + A * (cnt - p.K) ** 2)] == p.K:
            return float(A)
    return float("inf")
