"""
Phase E2: compiled Pauli propagation for kicked-Ising circuits (numba)
=====================================================================
Same physics and conventions as ``pauli_prop`` (Heisenberg picture, Hermitian
Pauli convention, coefficient truncation), specialised to RX(theta_h) layers
and Clifford RZZ(-pi/2) layers, which is the circuit of IBM's 127-qubit
"utility" experiment. About 50-100x faster than the numpy engine.

Why it is fast:
- Strings live in an open-addressing hash table (keys + coefficients arrays).
- RX(theta) on qubit q only mixes the pair (Z_q ..., Y_q ...) of strings that
  differ in x_q:  a_Z' = c a_Z - s a_Y,  a_Y' = c a_Y + s a_Z.  The update is
  done in place; a missing partner is inserted only if it survives truncation.
- A Clifford RZZ(-pi/2) layer never changes x and never merges strings, so it
  is one bitwise pass over the table followed by a rehash.
"""

from typing import Dict, Optional, Sequence, Tuple
import numpy as np

try:
    import numba as nb
    HAVE_NUMBA = True
except ImportError:          # pragma: no cover
    HAVE_NUMBA = False


def available() -> bool:
    return HAVE_NUMBA


if HAVE_NUMBA:
    _ONE = np.uint64(1)

    @nb.njit(cache=True, inline="always")
    def _hash(keys, i, mask):
        h = np.uint64(14695981039346656037)
        for j in range(keys.shape[1]):
            h ^= keys[i, j]
            h *= np.uint64(1099511628211)
            h ^= h >> np.uint64(29)
        return h & mask

    @nb.njit(cache=True)
    def _find(keys, table, mask, cand):
        """Index of the row equal to row ``cand`` (or -1), and the probe slot."""
        h = _hash(keys, cand, mask)
        m = keys.shape[1]
        while True:
            idx = table[h]
            if idx == -1:
                return -1, h
            same = True
            for j in range(m):
                if keys[idx, j] != keys[cand, j]:
                    same = False
                    break
            if same:
                return idx, h
            h = (h + _ONE) & mask

    @nb.njit(cache=True)
    def _rebuild(keys, coef, n, table, mask, max_weight=-1):
        """Drop zero coefficients (and strings heavier than ``max_weight`` if >= 0),
        compact, and rehash. Returns the new size."""
        W = keys.shape[1] // 2
        m = 0
        for i in range(n):
            keep = coef[i] != 0.0
            if keep and max_weight >= 0:
                wt = 0
                for j in range(W):
                    v = keys[i, j] | keys[i, W + j]
                    while v:
                        v &= v - _ONE
                        wt += 1
                keep = wt <= max_weight
            if keep:
                if m != i:
                    keys[m, :] = keys[i, :]
                    coef[m] = coef[i]
                m += 1
        table[:] = -1
        for i in range(m):
            h = _hash(keys, i, mask)
            while table[h] != -1:
                h = (h + _ONE) & mask
            table[h] = i
        return m

    @nb.njit(cache=True)
    def _rx(keys, coef, table, mask, n, q, c, s, eps):
        """U^dagger O U for U = exp(-i theta/2 X_q); returns the new row count."""
        W = keys.shape[1] // 2
        w = q // 64
        b = np.uint64(q % 64)
        bit = _ONE << b
        n0 = n
        for i in range(n0):
            if (keys[i, W + w] >> b) & _ONE == 0:
                continue                      # I or X at q: commutes
            xq = (keys[i, w] >> b) & _ONE
            keys[n, :] = keys[i, :]           # partner key in scratch row n
            keys[n, w] ^= bit
            p, slot = _find(keys, table, mask, n)
            if xq == 1:                       # Y at q
                if p != -1:
                    continue                  # handled from its Z partner
                aY = coef[i]
                nZ = -s * aY
                coef[i] = c * aY if abs(c * aY) >= eps else 0.0
                if abs(nZ) >= eps:
                    table[slot] = n
                    coef[n] = nZ
                    n += 1
            else:                             # Z at q
                aZ = coef[i]
                aY = coef[p] if p != -1 else 0.0
                nZ = c * aZ - s * aY
                nY = c * aY + s * aZ
                coef[i] = nZ if abs(nZ) >= eps else 0.0
                if p != -1:
                    coef[p] = nY if abs(nY) >= eps else 0.0
                elif abs(nY) >= eps:
                    table[slot] = n
                    coef[n] = nY
                    n += 1
        return n

    @nb.njit(cache=True)
    def _zz_layer(keys, coef, n, ea, eb):
        """All RZZ(-pi/2) gates of one layer (they commute), in place."""
        W = keys.shape[1] // 2
        for i in range(n):
            for k in range(ea.shape[0]):
                a = ea[k]
                bq = eb[k]
                wa = a // 64
                ba = np.uint64(a % 64)
                wb = bq // 64
                bb = np.uint64(bq % 64)
                xa = (keys[i, wa] >> ba) & _ONE
                xb = (keys[i, wb] >> bb) & _ONE
                if xa != xb:
                    if xa == 1:
                        zs = (keys[i, W + wa] >> ba) & _ONE
                    else:
                        zs = (keys[i, W + wb] >> bb) & _ONE
                    if zs == 1:
                        coef[i] = -coef[i]
                    keys[i, W + wa] ^= _ONE << ba
                    keys[i, W + wb] ^= _ONE << bb


class _Table:
    def __init__(self, n_qubits: int, cap: int = 1 << 12):
        self.W = (n_qubits + 63) // 64
        self.n = 0
        self._alloc(cap)

    def _alloc(self, cap):
        self.cap = cap
        self.keys = np.zeros((cap, 2 * self.W), np.uint64)
        self.coef = np.zeros(cap, np.float64)
        size = 1 << int(np.ceil(np.log2(2 * cap)))
        self.table = np.full(size, -1, np.int64)
        self.mask = np.uint64(size - 1)

    def ensure(self, need):
        if need <= self.cap:
            return
        cap = self.cap
        while cap < need:
            cap *= 2
        keys, coef, n = self.keys, self.coef, self.n
        self._alloc(cap)
        self.keys[:n] = keys[:n]
        self.coef[:n] = coef[:n]
        self.n = _rebuild(self.keys, self.coef, n, self.table, self.mask)


def kicked_ising_expectation_fast(n_qubits: int, ops: Dict[int, str],
                                  edges: Sequence[Tuple[int, int]], steps: int,
                                  theta_h: float, min_abs: float = 1e-4,
                                  final_rx: bool = False,
                                  max_weight: Optional[int] = None,
                                  stats: Optional[dict] = None) -> float:
    """Same result as ``pauli_prop.kicked_ising_expectation`` at theta_J = -pi/2.

    ``max_weight`` drops strings with more than that many non-identity sites
    after every ZZ layer (the only place the weight can grow)."""
    if not HAVE_NUMBA:
        raise ImportError("numba is required for the fast engine")
    from phase3_scale.pauli_prop import PauliSum
    t = _Table(n_qubits)
    p = PauliSum.from_label(n_qubits, ops)
    t.keys[0, :t.W] = p.x[0]
    t.keys[0, t.W:] = p.z[0]
    t.coef[0] = 1.0
    t.n = _rebuild(t.keys, t.coef, 1, t.table, t.mask)
    ea = np.array([a for a, _ in edges], np.int64)
    eb = np.array([b for _, b in edges], np.int64)
    c, s = float(np.cos(theta_h)), float(np.sin(theta_h))
    layers = []
    for _ in range(steps):
        layers += ["rx", "zz"]
    if final_rx:
        layers.append("rx")
    peak = t.n
    for kind in reversed(layers):
        if kind == "zz":
            _zz_layer(t.keys, t.coef, t.n, ea, eb)
            t.n = _rebuild(t.keys, t.coef, t.n, t.table, t.mask,
                           -1 if max_weight is None else int(max_weight))
        else:
            for q in range(n_qubits):
                t.ensure(2 * t.n + 2)
                t.n = _rx(t.keys, t.coef, t.table, t.mask, t.n, q, c, s, min_abs)
                peak = max(peak, t.n)
            t.n = _rebuild(t.keys, t.coef, t.n, t.table, t.mask)
    if stats is not None:
        stats.update(final_terms=int(t.n), peak_terms=int(peak))
    k = t.keys[:t.n]
    diag = ~np.any(k[:, :t.W], axis=1)
    return float(t.coef[:t.n][diag].sum())
