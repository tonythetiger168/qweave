"""
Phase C: Sparse Hamiltonians + parameter-shift gradients for VQE
=================================================================
`SparsePauliHamiltonian` stores only non-identity Pauli terms, so
memory scales with the number of terms instead of 2^n.

Expectation is evaluated directly on state-vector amplitudes with
bit-twiddling (vectorized numpy, O(2^n * #terms) per evaluation).

Gradients of <H> w.r.t. rotation angles use the parameter-shift rule:
    df/dθ = ( f(θ + π/2) - f(θ - π/2) ) / 2        (exact for Pauli rot)
"""

from typing import Dict, List, Tuple
import numpy as np


def _popcount_parity(idx: np.ndarray, mask: int) -> np.ndarray:
    """(-1)^{popcount(idx & mask)} as float array."""
    p = np.zeros(len(idx), dtype=np.int8)
    m = mask
    while m:
        low = m & (-m)
        p ^= (idx & low != 0).astype(np.int8)
        m ^= low
    return np.where(p, -1.0, 1.0)


class SparsePauliHamiltonian:
    def __init__(self, terms: Dict[str, complex]):
        self.terms = {p: c for p, c in terms.items()
                      if abs(c) > 0 and any(ch != 'I' for ch in p)}
        self.n = len(next(iter(self.terms))) if self.terms else 0

    @classmethod
    def from_list(cls, pairs: List[Tuple[str, complex]]):
        return cls(dict(pairs))

    def expectation(self, amplitudes: np.ndarray, n_qubits: int) -> float:
        N = 2 ** n_qubits
        psi = amplitudes.reshape(N)
        idx = np.arange(N)
        conj = np.conj(psi)
        total = 0.0 + 0.0j
        for pauli, coeff in self.terms.items():
            zmask = fmask = ymask = 0
            for q, ch in enumerate(pauli):
                if ch in 'ZY':
                    zmask |= 1 << q
                if ch in 'XY':
                    fmask |= 1 << q
                if ch == 'Y':
                    ymask |= 1 << q
            flipped = psi[idx ^ fmask]
            # Pauli action: P|x> = i^|Y| (-1)^{popcount(x & (z|y))} |x ^ f>
            phase = (1j ** bin(ymask).count('1')) * _popcount_parity(idx, zmask | ymask)
            total += coeff * np.dot(conj, phase * flipped)
        return float(np.real(total))


def vqe_gradient(circuit_fn, params: np.ndarray,
                 hamiltonian: SparsePauliHamiltonian,
                 n_qubits: int) -> Tuple[np.ndarray, float]:
    """Parameter-shift gradient + energy at params."""
    grads = np.zeros(len(params))
    for i in range(len(params)):
        p = params.copy()
        p[i] += np.pi / 2
        fp = hamiltonian.expectation(circuit_fn(p), n_qubits)
        p[i] -= np.pi
        fm = hamiltonian.expectation(circuit_fn(p), n_qubits)
        grads[i] = (fp - fm) / 2
    return grads, hamiltonian.expectation(circuit_fn(params), n_qubits)


def vqe_minimize(circuit_fn, params0: np.ndarray,
                 hamiltonian: SparsePauliHamiltonian,
                 n_qubits: int, iters: int = 200,
                 lr: float = 0.1) -> Tuple[np.ndarray, float]:
    """Gradient descent with parameter-shift gradients."""
    p = params0.copy()
    best_e, best_p = np.inf, p.copy()
    for _ in range(iters):
        g, e = vqe_gradient(circuit_fn, p, hamiltonian, n_qubits)
        p = p - lr * g
        if e < best_e:
            best_e, best_p = e, p.copy()
    return best_p, best_e
