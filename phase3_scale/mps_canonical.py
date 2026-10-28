"""
Phase C: Canonical-form MPS with adaptive bond dimension
=========================================================
Upgrades over MPSSimulator (P1.3):
  * two-site updates keep a genuine mixed canonical form (QR sweep)
  * ADAPTIVE truncation: keep singular values up to a cumulative
    discarded-weight threshold eps (target accuracy) with chi_max cap
  * entanglement entropy from the Schmidt spectrum at any bond
  * state_vector() export for cross-validation against state-vector sim
"""

from typing import List, Optional
import numpy as np


class MPSCanonical:
    """Canonical MPS for 1D nearest-neighbour circuits."""

    def __init__(self, n_qubits: int, chi_max: int = 128, eps: float = 1e-12):
        self.n = n_qubits
        self.chi_max = chi_max
        self.eps = eps                 # adaptive truncation threshold
        self.tensors: List[np.ndarray] = []
        for _ in range(n_qubits):
            A = np.zeros((1, 2, 1), dtype=complex)
            A[0, 0, 0] = 1.0
            self.tensors.append(A)
        self.gate_count = 0

    # ---------------- gate applications ----------------
    def _apply_1q(self, gate: np.ndarray, site: int):
        A = self.tensors[site]
        self.tensors[site] = np.einsum('ab,ibr->iar', gate, A)
        self.gate_count += 1

    def _apply_2q(self, gate: np.ndarray, site: int):
        A, B = self.tensors[site], self.tensors[site + 1]
        theta = np.einsum('iam,mbr->iabr', A, B)          # (Dl,2,2,Dr)
        # contract BOTH physical indices with the gate matrix
        out = np.einsum('cdab,iabr->cdir', gate.reshape(2, 2, 2, 2), theta)
        Dl, Dr = out.shape[2], out.shape[3]
        M = np.transpose(out, (2, 0, 1, 3)).reshape(Dl * 2, 2 * Dr)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)

        # ---- adaptive truncation by cumulative discarded weight ----
        w = S ** 2
        total = w.sum()
        csum = np.cumsum(w)
        keep = np.searchsorted(csum, (1 - self.eps) * total) + 1
        chi = int(min(keep, len(S), self.chi_max))

        U, S, Vh = U[:, :chi], S[:chi], Vh[:chi, :]
        # U rows ordered (bond, phys): direct reshape is left-canonical
        self.tensors[site] = U.reshape(Dl, 2, chi)
        self.tensors[site + 1] = (np.diag(S) @ Vh).reshape(chi, 2, Dr)
        self.gate_count += 1
        self._center = site + 1

    # ---------------- gate library ----------------
    def h(self, q):
        self._apply_1q(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), q)
    def x(self, q):
        self._apply_1q(np.array([[0, 1], [1, 0]], dtype=complex), q)
    def y(self, q):
        self._apply_1q(np.array([[0, -1j], [1j, 0]], dtype=complex), q)
    def z(self, q):
        self._apply_1q(np.array([[1, 0], [0, -1]], dtype=complex), q)
    def s(self, q):
        self._apply_1q(np.array([[1, 0], [0, 1j]], dtype=complex), q)
    def t(self, q):
        self._apply_1q(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]],
                                dtype=complex), q)
    def rx(self, theta, q):
        self._apply_1q(np.array(
            [[np.cos(theta/2), -1j*np.sin(theta/2)],
             [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex), q)
    def ry(self, theta, q):
        self._apply_1q(np.array(
            [[np.cos(theta/2), -np.sin(theta/2)],
             [np.sin(theta/2), np.cos(theta/2)]], dtype=complex), q)
    def rz(self, theta, q):
        self._apply_1q(np.array(
            [[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex), q)
    def cnot(self, c, t):
        if abs(c - t) != 1:
            raise ValueError("nearest neighbours only")
        if c < t:   # control = left tensor = high index of the matrix
            gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        else:       # control = right tensor = low index
            gate = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
        self._apply_2q(gate.astype(complex), min(c, t))
    def cz(self, c, t):
        if abs(c - t) != 1:
            raise ValueError("nearest neighbours only")
        self._apply_2q(np.diag([1, 1, 1, -1]).astype(complex), min(c, t))

    # ---------------- canonical form maintenance ----------------
    def canonicalize(self, center: Optional[int] = None):
        """Left-to-right then right-to-left QR sweep so that all sites
        except `center` are isometries; orthogonality center at `center`
        (default: last two-site-update position)."""
        if center is None:
            center = getattr(self, '_center', self.n - 1)
        # left sweep
        for i in range(center):
            Dl, d, Dr = self.tensors[i].shape
            M = self.tensors[i].reshape(Dl * d, Dr)
            Q, R = np.linalg.qr(M)
            self.tensors[i] = Q.reshape(Dl, d, Q.shape[1])
            self.tensors[i + 1] = np.einsum('ab,bsr->asr', R, self.tensors[i + 1])
        # right sweep
        for i in range(self.n - 1, center, -1):
            Dl, d, Dr = self.tensors[i].shape
            M = self.tensors[i].reshape(Dl, d * Dr)
            Q, R = np.linalg.qr(M.T)
            self.tensors[i] = Q.T.reshape(Q.shape[1], d, Dr)
            self.tensors[i - 1] = np.einsum('asb,br->asr',
                                            self.tensors[i - 1], R.T)
        self._center = center

    def schmidt_spectrum(self, bond: int) -> np.ndarray:
        """Entanglement spectrum across cut after `bond` sites (0-based,
        sites <= bond on the left). Requires canonical form; calls
        canonicalize(bond) internally."""
        self.canonicalize(bond)
        M = self.tensors[bond]              # center: right bond IS the cut
        S = np.linalg.svd(M.reshape(M.shape[0] * M.shape[1], M.shape[2]),
                          compute_uv=False)
        return S / np.linalg.norm(S)

    def entanglement_entropy(self, bond: int) -> float:
        s = self.schmidt_spectrum(bond)
        s = s[s > 1e-15]
        return float(-np.sum(s ** 2 * np.log2(s ** 2)))

    # ---------------- utilities ----------------
    def max_bond(self) -> int:
        return max(t.shape[2] for t in self.tensors[:-1])

    def state_vector(self) -> np.ndarray:
        """Full 2^n amplitudes (validation / small systems only)."""
        v = self.tensors[0][0, :, :]                        # (2, D)
        for i in range(1, self.n):
            v = np.einsum('...r,ras->...as', v, self.tensors[i])
            D = v.shape[-1]
            v = v.reshape(-1, D)
        v = v.reshape(-1)
        # contraction puts qubit 0 on the most-significant axis; the
        # state-vector convention here is little-endian (qubit 0 = bit 0)
        perm = np.array([int(format(i, f'0{self.n}b')[::-1], 2)
                         for i in range(2 ** self.n)])
        return v[perm]

    def fidelity(self, other_state: np.ndarray) -> float:
        return float(np.abs(np.vdot(self.state_vector(), other_state)) ** 2)
