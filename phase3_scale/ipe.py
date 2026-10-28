"""
Phase C: Iterative Phase Estimation (IPEA)
============================================
Standard QPE needs n_system + k counting qubits (2^(n+k) amplitudes).
IPEA uses ONE ancilla + classical feedback, run k times -> n_system + 1
qubits total, cutting memory by 2^(k-1) at equal precision.

Bit order: least-significant bit first. Round m extracts bit b_m with
controlled-U^(2^(k-1-m)); previously measured lower bits (l < m) are
cancelled by an RZ feedback so the X-basis measurement is deterministic.

Contract: `round_fn(sim, j)` receives a fresh StateVectorSimulator with
the ancilla already in |+> on qubit 0. It must (1) prepare the system
eigenstate on qubits 1..n and (2) apply U^(2**j) controlled on qubit 0.
See `diagonal_unitary_round_fn` for the common diagonal-U case.
"""

from typing import Callable, Tuple
import numpy as np


def diagonal_unitary_round_fn(phi: float, eigen_one: bool = True) -> Callable:
    """round_fn for U = diag(1, e^{i 2 pi phi}), eigenstate |1>
    (or |0> when eigen_one=False)."""
    def round_fn(sim, j: int):
        if eigen_one:
            sim.x(1)
        Uk = np.exp(2j * np.pi * phi * (2 ** j))
        CU = np.diag([1.0, 1.0, 1.0, Uk]).astype(complex)  # (anc, sys)
        sim._apply_2q(CU, 0, 1)
    return round_fn


def run_ipea(n_system: int, round_fn: Callable, k_bits: int) -> Tuple[float, list]:
    """
    Returns (phase in [0,1), bits). Each round uses a fresh n_system+1
    qubit simulator -> memory 2^(n_system+1) regardless of k.
    """
    from phase1_basic.simulators import StateVectorSimulator

    bits = []
    for m in range(k_bits):
        sim = StateVectorSimulator(n_system + 1)
        sim.h(0)                                  # ancilla |+>
        round_fn(sim, k_bits - 1 - m)             # eigenstate + ctrl-U^(2^j)
        # cancel phases from previously measured lower bits (l < m)
        phi_prev = sum(b * 2.0 ** (l - m - 1) for l, b in enumerate(bits))
        sim.rz(-2 * np.pi * phi_prev, 0)
        sim.h(0)
        bits.append(sim.measure(0))
    phase = sum(b * 2.0 ** (m - k_bits) for m, b in enumerate(bits))
    return phase, bits
