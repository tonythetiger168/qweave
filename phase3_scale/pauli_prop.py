"""
Phase E2: Pauli propagation (Heisenberg-picture simulation)
===========================================================
An observable O is written as a sum of Pauli strings and pushed backwards
through the circuit, O <- U_g^dagger O U_g, from the last gate to the first.
The expectation value on |0...0> is then the sum of the coefficients of the
strings that contain only I and Z.

For a Pauli rotation U = exp(-i theta/2 G) and a string P:
    commuting P:      P                                  (unchanged)
    anticommuting P:  cos(theta) P + sin(theta) (i G P)  (one branch)
so Clifford angles (theta = k pi/2) never branch. Truncation drops strings
with |coefficient| < ``min_abs`` or with Pauli weight > ``max_weight``.
This is the method used to simulate IBM's 127-qubit kicked-Ising "utility"
experiment classically (Begusic & Chan 2023; Rudolph et al. 2025).

Representation: N strings stored as packed bit arrays x, z of shape (N, W)
with W = ceil(n/64) uint64 words, in the Hermitian convention
    P(x, z) = prod_q i^(x_q z_q) X^x_q Z^z_q        (so Y = iXZ),
and real coefficients (the gates are real Pauli rotations, O is Hermitian).
"""

from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np


_HASH_MULT = np.random.default_rng(0x9E3779B9).integers(
    1, 2**63, size=64, dtype=np.uint64) | np.uint64(1)


def _popcount_rows(a: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a).sum(axis=1, dtype=np.int64)


class PauliSum:
    """A real linear combination of n-qubit Pauli strings."""

    def __init__(self, n: int, x: np.ndarray, z: np.ndarray, coef: np.ndarray):
        self.n = n
        self.x, self.z, self.coef = x, z, coef

    @property
    def words(self) -> int:
        return (self.n + 63) // 64

    def __len__(self):
        return len(self.coef)

    @classmethod
    def from_label(cls, n: int, ops: dict, coef: float = 1.0) -> "PauliSum":
        """``ops`` maps qubit -> 'X' | 'Y' | 'Z', e.g. {62: 'Z'}."""
        W = (n + 63) // 64
        x = np.zeros((1, W), np.uint64)
        z = np.zeros((1, W), np.uint64)
        for q, p in ops.items():
            w, b = divmod(q, 64)
            bit = np.uint64(1) << np.uint64(b)
            if p in "XY":
                x[0, w] |= bit
            if p in "ZY":
                z[0, w] |= bit
        return cls(n, x, z, np.array([coef], float))

    def weights(self) -> np.ndarray:
        return _popcount_rows(self.x | self.z)

    def expectation_zero_state(self) -> float:
        """<0...0| O |0...0>: only strings without X/Y contribute, each with +1."""
        diag = ~np.any(self.x, axis=1)
        return float(self.coef[diag].sum())

    def merge(self):
        """Add up the coefficients of identical strings."""
        if len(self) < 2:
            return
        # 64-bit hash of each string, sorted in 1-D (much faster than sorting
        # the packed rows); a full-key comparison guards against collisions
        key = np.hstack([self.x, self.z])
        h = np.zeros(len(key), np.uint64)
        for j in range(key.shape[1]):
            h ^= key[:, j] * _HASH_MULT[j]
            h ^= h >> np.uint64(29)
        order = np.argsort(h, kind="stable")
        hs = h[order]
        start = np.ones(len(hs), bool)
        start[1:] = hs[1:] != hs[:-1]
        grp = np.cumsum(start) - 1
        first = order[start]
        if not np.array_equal(key[order], key[first][grp]):       # hash collision
            v = np.ascontiguousarray(key).view(
                np.dtype((np.void, key.dtype.itemsize * key.shape[1]))).ravel()
            _, first, inv = np.unique(v, return_index=True, return_inverse=True)
            self.coef = np.bincount(inv.ravel(), weights=self.coef, minlength=len(first))
        else:
            self.coef = np.bincount(grp, weights=self.coef[order], minlength=len(first))
        self.x, self.z = self.x[first], self.z[first]

    def truncate(self, min_abs: float = 0.0, max_weight: Optional[int] = None):
        keep = np.abs(self.coef) >= min_abs if min_abs > 0 else np.ones(len(self), bool)
        if max_weight is not None:
            keep &= self.weights() <= max_weight
        if not keep.all():
            self.x, self.z, self.coef = self.x[keep], self.z[keep], self.coef[keep]


def _mask(n: int, ops: dict) -> Tuple[np.ndarray, np.ndarray]:
    p = PauliSum.from_label(n, ops)
    return p.x[0], p.z[0]


def apply_pauli_rotation(ps: PauliSum, gx: np.ndarray, gz: np.ndarray, theta: float,
                         min_abs: float = 0.0, max_weight: Optional[int] = None,
                         merge: bool = True):
    """In place: ps <- U^dagger ps U with U = exp(-i theta/2 G), G = P(gx, gz)."""
    anti = (_popcount_rows((ps.x & gz) ^ (ps.z & gx)) & 1).astype(bool)
    if not anti.any():
        return
    ax, az, ac = ps.x[anti], ps.z[anti], ps.coef[anti]
    rx, rz = ax ^ gx, az ^ gz
    # phase of i G P in the Hermitian convention: i^e with e in {0, 2}
    e = (1 + int(np.bitwise_count(gx & gz).sum())
         + _popcount_rows(ax & az) + 2 * _popcount_rows(ax & gz)
         - _popcount_rows(rx & rz)) % 4
    sign = 1.0 - e                         # 0 -> +1, 2 -> -1
    c, s = np.cos(theta), np.sin(theta)
    if abs(c) < 1e-15:                     # Clifford: replace, no branching
        ps.x[anti], ps.z[anti] = rx, rz
        ps.coef[anti] = ac * s * sign
        if merge:
            ps.merge()
    else:
        ps.coef[anti] = ac * c
        ps.x = np.concatenate([ps.x, rx])
        ps.z = np.concatenate([ps.z, rz])
        ps.coef = np.concatenate([ps.coef, ac * s * sign])
        if merge:
            ps.merge()
    ps.truncate(min_abs, max_weight)


# ----------------------------------------------------------------------
# IBM Eagle (127 qubits, heavy-hex) and the kicked-Ising circuit
# ----------------------------------------------------------------------

def eagle_edges() -> List[Tuple[int, int]]:
    """The 144 couplers of a 127-qubit IBM Eagle processor (ibm_kyiv etc.)."""
    rows = [list(range(0, 14))]
    start = 18
    for _ in range(5):
        rows.append(list(range(start, start + 15)))
        start += 19
    rows.append(list(range(113, 127)))
    edges = [(r[i], r[i + 1]) for r in rows for i in range(len(r) - 1)]
    bridges = [(14, (0, 18)), (15, (4, 22)), (16, (8, 26)), (17, (12, 30)),
               (33, (20, 39)), (34, (24, 43)), (35, (28, 47)), (36, (32, 51)),
               (52, (37, 56)), (53, (41, 60)), (54, (45, 64)), (55, (49, 68)),
               (71, (58, 77)), (72, (62, 81)), (73, (66, 85)), (74, (70, 89)),
               (90, (75, 94)), (91, (79, 98)), (92, (83, 102)), (93, (87, 106)),
               (109, (96, 114)), (110, (100, 118)), (111, (104, 122)), (112, (108, 126))]
    for b, (u, v) in bridges:
        edges += [(min(b, u), max(b, u)), (min(b, v), max(b, v))]
    return sorted(edges)


def kicked_ising_expectation(observable: PauliSum, edges: Sequence[Tuple[int, int]],
                             steps: int, theta_h: float, theta_j: float = -np.pi / 2,
                             min_abs: float = 1e-4, max_weight: Optional[int] = None,
                             final_rx: bool = False, stats: Optional[dict] = None) -> float:
    """<O> after ``steps`` Trotter steps of Kim et al. (Nature 618, 500, 2023).

    One step = RX(theta_h) on every qubit, then RZZ(theta_j) on every edge.
    With ``final_rx`` one more RX layer follows (their weight-10/17 observables).
    """
    n = observable.n
    ps = PauliSum(n, observable.x.copy(), observable.z.copy(), observable.coef.copy())
    zz = [_mask(n, {a: 'Z', b: 'Z'}) for a, b in edges]
    xs = [_mask(n, {q: 'X'}) for q in range(n)]
    layers = []
    for _ in range(steps):
        layers.append(("rx", theta_h))
        layers.append(("zz", theta_j))
    if final_rx:
        layers.append(("rx", theta_h))
    peak = len(ps)
    for kind, th in reversed(layers):       # Heisenberg: last layer first
        gates = zz if kind == "zz" else xs
        for gx, gz in gates:
            # the ZZ layer is Clifford at theta_j = -pi/2: no duplicates can form
            apply_pauli_rotation(ps, gx, gz, th, min_abs, max_weight,
                                 merge=not (kind == "zz" and abs(np.cos(th)) < 1e-15))
            peak = max(peak, len(ps))
    if stats is not None:
        stats.update(final_terms=len(ps), peak_terms=peak)
    return ps.expectation_zero_state()


# ----------------------------------------------------------------------
# Exact reference: statevector on the backward light cone
# ----------------------------------------------------------------------

def light_cone(edges: Sequence[Tuple[int, int]], qubits: Iterable[int], radius: int) -> List[int]:
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen = set(qubits)
    frontier = set(qubits)
    for _ in range(radius):
        frontier = {v for u in frontier for v in adj.get(u, []) if v not in seen}
        seen |= frontier
    return sorted(seen)


def kicked_ising_statevector(n_sites: Sequence[int], edges: Sequence[Tuple[int, int]],
                             ops: dict, steps: int, theta_h: float,
                             theta_j: float = -np.pi / 2, final_rx: bool = False) -> float:
    """Exact <O> by statevector on the qubits ``n_sites`` (gates outside are dropped).

    Exact for the full device when ``n_sites`` contains the backward light cone of O.
    """
    sites = list(n_sites)
    idx = {q: i for i, q in enumerate(sites)}
    n = len(sites)
    loc = [(idx[a], idx[b]) for a, b in edges if a in idx and b in idx]
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None]) & 1).astype(np.int8)
    zsign = 1 - 2 * bits                                   # +1 for |0>, -1 for |1>
    zzsum = np.zeros(2 ** n)
    for a, b in loc:
        zzsum += zsign[:, a] * zsign[:, b]
    phase = np.exp(-0.5j * theta_j * zzsum)
    c, s = np.cos(theta_h / 2), -1j * np.sin(theta_h / 2)
    psi = np.zeros(2 ** n, complex)
    psi[0] = 1.0

    def rx_layer(v):
        v = v.reshape([2] * n)                              # axis k <-> qubit n-1-k
        for q in range(n):
            ax = n - 1 - q
            v = np.moveaxis(v, ax, 0)
            a0, a1 = v[0].copy(), v[1].copy()
            v[0], v[1] = c * a0 + s * a1, s * a0 + c * a1
            v = np.moveaxis(v, 0, ax)
        return v.reshape(-1)

    for _ in range(steps):
        psi = rx_layer(psi)
        psi = psi * phase
    if final_rx:
        psi = rx_layer(psi)
    # apply O and take <psi|O|psi>
    phi = psi.reshape([2] * n).copy()
    for q, p in ops.items():
        ax = n - 1 - idx[q]
        phi = np.moveaxis(phi, ax, 0)
        a0, a1 = phi[0].copy(), phi[1].copy()
        if p == 'Z':
            phi[1] = -a1
        elif p == 'X':
            phi[0], phi[1] = a1, a0
        else:                                               # Y
            phi[0], phi[1] = -1j * a1, 1j * a0
        phi = np.moveaxis(phi, 0, ax)
    return float(np.real(np.vdot(psi, phi.reshape(-1))))
