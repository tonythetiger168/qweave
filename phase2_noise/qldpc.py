"""
Phase E1-b: Bivariate Bicycle qLDPC Codes + BP+OSD Decoder
===========================================================
IBM's bivariate bicycle (BB) codes (Bravyi et al., Nature 627, 778,
2024), with the [[144,12,12]] "gross code" as the flagship, decoded by
our own numpy belief propagation + ordered-statistics decoder (BP+OSD-0,
Panteleev & Kalachev 2021; Roffe et al. 2020).

  A = sum of monomials x^a y^b,  B likewise, with
      x = S_l (x) I_m,  y = I_l (x) S_m   (S = cyclic shift)
  H_X = [A | B],  H_Z = [B^T | A^T]   ->  H_X H_Z^T = AB + BA = 0

Everything here is pure numpy: GF(2) linear algebra, logical operator
extraction, a batched min-sum BP, and OSD-0 post-processing. The
``ldpc`` package (Roffe) is only used as an optional cross-check.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


# ----------------------------------------------------------------------
# GF(2) linear algebra
# ----------------------------------------------------------------------
def gf2_row_reduce(M: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Reduced row echelon form over GF(2); returns (R, pivot columns)."""
    R = (np.array(M, dtype=np.uint8) & 1).copy()
    rows, cols = R.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        hit = np.nonzero(R[r:, c])[0]
        if len(hit) == 0:
            continue
        p = r + hit[0]
        if p != r:
            R[[r, p]] = R[[p, r]]
        others = np.nonzero(R[:, c])[0]
        others = others[others != r]
        R[others] ^= R[r]
        pivots.append(c)
        r += 1
    return R[:r], pivots


def gf2_rank(M: np.ndarray) -> int:
    return len(gf2_row_reduce(M)[1])


def gf2_nullspace(M: np.ndarray) -> np.ndarray:
    """Basis of {v : M v = 0} as rows."""
    R, pivots = gf2_row_reduce(M)
    n = M.shape[1]
    free = [c for c in range(n) if c not in set(pivots)]
    basis = np.zeros((len(free), n), np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, p in enumerate(pivots):
            basis[i, p] = R[r, f]
    return basis


def logical_operators(hx: np.ndarray, hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    (L_X, L_Z): k independent X- and Z-type logicals.
    L_Z spans ker(H_X) / rowspace(H_Z), and L_X spans ker(H_Z) / rowspace(H_X).
    """
    def quotient(kernel_of, stabs):
        ker = gf2_nullspace(kernel_of)
        base = gf2_rank(stabs)
        picked = []
        cur = stabs.copy()
        for v in ker:
            trial = np.vstack([cur, v])
            if gf2_rank(trial) > base + len(picked):
                picked.append(v)
                cur = trial
        return np.array(picked, np.uint8)
    return quotient(hz, hx), quotient(hx, hz)


# ----------------------------------------------------------------------
# Bivariate bicycle codes
# ----------------------------------------------------------------------
def _shift(n: int) -> np.ndarray:
    return np.roll(np.eye(n, dtype=np.uint8), 1, axis=1)


def bivariate_bicycle(l: int, m: int,
                      a_terms: Sequence[Tuple[str, int]],
                      b_terms: Sequence[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    (H_X, H_Z) for A = sum of a_terms, B = sum of b_terms, where a term
    ('x', 3) means x^3 and ('y', 2) means y^2.
    """
    x = np.kron(_shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), _shift(m))

    def poly(terms):
        out = np.zeros((l * m, l * m), np.uint8)
        for var, power in terms:
            base = x if var == 'x' else y
            out ^= np.linalg.matrix_power(base.astype(np.int64), power).astype(np.uint8) & 1
        return out

    A, B = poly(a_terms), poly(b_terms)
    hx = np.hstack([A, B])
    hz = np.hstack([B.T, A.T])
    return hx, hz


def gross_code() -> Tuple[np.ndarray, np.ndarray]:
    """[[144,12,12]]: l=12, m=6, A = x^3 + y + y^2, B = y^3 + x + x^2."""
    return bivariate_bicycle(12, 6,
                             [('x', 3), ('y', 1), ('y', 2)],
                             [('y', 3), ('x', 1), ('x', 2)])


def code_parameters(hx: np.ndarray, hz: np.ndarray) -> Dict:
    n = hx.shape[1]
    k = n - gf2_rank(hx) - gf2_rank(hz)
    return {'n': n, 'k': k,
            'commute': not np.any((hx.astype(np.int64) @ hz.T.astype(np.int64)) % 2),
            'row_weights': sorted(set(hx.sum(1).tolist()) | set(hz.sum(1).tolist())),
            'col_weights': sorted(set(hx.sum(0).tolist()) | set(hz.sum(0).tolist()))}


# ----------------------------------------------------------------------
# BP + OSD-0 decoder
# ----------------------------------------------------------------------
class BpOsdDecoder:
    """
    Batched normalized min-sum BP, then OSD-0 on shots where BP did not
    reach the syndrome.

    BP runs on all shots at once: messages live on the edges of H, padded
    to the max row/column weight so check and variable updates are plain
    numpy reductions over a fixed axis.
    """

    def __init__(self, H: np.ndarray, error_rate: float,
                 max_iter: int = 100, ms_scaling: float = 0.625):
        self.H = (np.array(H, np.uint8) & 1)
        self.m, self.n = self.H.shape
        self.max_iter = max_iter
        self.alpha = ms_scaling
        self.prior = np.log((1 - error_rate) / error_rate)

        rows, cols = np.nonzero(self.H)
        self.E = len(rows)
        self.edge_row, self.edge_col = rows, cols
        # check-side padded layout: (m, wr) of edge ids, -1 = padding
        wr = int(self.H.sum(1).max())
        wc = int(self.H.sum(0).max())
        self.chk = -np.ones((self.m, wr), np.int64)
        self.var = -np.ones((self.n, wc), np.int64)
        fill_c = np.zeros(self.m, np.int64)
        fill_v = np.zeros(self.n, np.int64)
        for e, (r, c) in enumerate(zip(rows, cols)):
            self.chk[r, fill_c[r]] = e
            fill_c[r] += 1
            self.var[c, fill_v[c]] = e
            fill_v[c] += 1
        self.chk_mask = self.chk >= 0
        self.var_mask = self.var >= 0
        self.rank = gf2_rank(self.H)

    # -- belief propagation -------------------------------------------
    def bp(self, syndromes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (hard decisions, posterior LLRs, converged mask)."""
        S = syndromes.shape[0]
        s_sign = 1.0 - 2.0 * syndromes.astype(np.float64)          # (S, m)
        v2c = np.full((S, self.E), self.prior)
        post = np.full((S, self.n), self.prior)
        hard = np.zeros((S, self.n), np.uint8)
        final_post = post.copy()
        cur = hard.copy()
        done = np.zeros(S, bool)
        chk_idx = np.where(self.chk_mask, self.chk, 0)
        var_idx = np.where(self.var_mask, self.var, 0)

        for _ in range(self.max_iter):
            # check update: sign product and min |.| excluding self
            msg = v2c[:, chk_idx]                                     # (S, m, wr)
            mag = np.where(self.chk_mask, np.abs(msg), np.inf)
            sgn = np.where(self.chk_mask, np.where(msg < 0, -1.0, 1.0), 1.0)
            total_sign = np.prod(sgn, axis=2, keepdims=True) * s_sign[:, :, None]
            order = np.argsort(mag, axis=2)
            min1 = np.take_along_axis(mag, order[:, :, :1], axis=2)
            min2 = np.take_along_axis(mag, order[:, :, 1:2], axis=2)
            is_min = np.arange(mag.shape[2])[None, None, :] == order[:, :, :1]
            out_mag = np.where(is_min, min2, min1) * self.alpha
            c2v_pad = total_sign * sgn * out_mag
            c2v = np.zeros((S, self.E))
            c2v[:, self.chk[self.chk_mask]] = c2v_pad[:, self.chk_mask]

            # variable update
            inc = np.where(self.var_mask, c2v[:, var_idx], 0.0)        # (S, n, wc)
            post = self.prior + inc.sum(axis=2)
            v2c_pad = post[:, :, None] - inc
            v2c = np.zeros((S, self.E))
            v2c[:, self.var[self.var_mask]] = v2c_pad[:, self.var_mask]

            cur = (post < 0).astype(np.uint8)
            ok = np.all(((cur.astype(np.int64) @ self.H.T) % 2) == syndromes, axis=1)
            # freeze each shot at its first converged iteration (as ldpc does)
            new = ok & ~done
            hard[new], final_post[new] = cur[new], post[new]
            done |= ok
            if done.all():
                break
        hard[~done], final_post[~done] = cur[~done], post[~done]
        return hard, final_post, done

    # -- ordered statistics decoding ----------------------------------
    def osd0(self, syndrome: np.ndarray, posterior: np.ndarray) -> np.ndarray:
        """Solve H e = s on the rank(H) most-likely-flipped columns."""
        order = np.argsort(posterior, kind='stable')   # most likely error first
        A = np.hstack([self.H[:, order], syndrome[:, None].astype(np.uint8)])
        R, pivots = gf2_row_reduce(A)
        e = np.zeros(self.n, np.uint8)
        for r, pc in enumerate(pivots):
            if pc == self.n:
                raise ValueError("syndrome not in the column space of H")
            e[order[pc]] = R[r, -1]
        return e

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        syndromes = np.asarray(syndromes, np.uint8)
        hard, post, done = self.bp(syndromes)
        for i in np.nonzero(~done)[0]:
            hard[i] = self.osd0(syndromes[i], post[i])
        return hard


# ----------------------------------------------------------------------
# Code-capacity experiment
# ----------------------------------------------------------------------
def code_capacity_ler(hx: np.ndarray, hz: np.ndarray, p: float, shots: int,
                      decoder=None, seed: Optional[int] = None) -> Dict:
    """
    Independent X errors with probability p on every data qubit, decoded
    from the Z-check syndrome. A shot fails if the residual flips any
    logical Z (i.e. anticommutes with a row of L_Z).
    """
    rng = np.random.default_rng(seed)
    _, lz = logical_operators(hx, hz)
    errors = (rng.random((shots, hx.shape[1])) < p).astype(np.uint8)
    synd = (errors.astype(np.int64) @ hz.T % 2).astype(np.uint8)
    if decoder is None:
        decoder = BpOsdDecoder(hz, p)
    corr = decoder.decode_batch(synd)
    resid = errors ^ corr
    assert not np.any(resid.astype(np.int64) @ hz.T % 2), "correction misses syndrome"
    fail = np.any(resid.astype(np.int64) @ lz.T % 2, axis=1)
    return {'p': p, 'shots': shots, 'errors': int(fail.sum()),
            'ler': float(fail.mean()), 'k': lz.shape[0]}


class LdpcReference:
    """Adapter so ``ldpc.BpOsdDecoder`` fits ``code_capacity_ler``."""

    def __init__(self, H: np.ndarray, error_rate: float,
                 max_iter: int = 100, ms_scaling: float = 0.625):
        from ldpc import BpOsdDecoder as _Ref
        self.dec = _Ref(H, error_rate=error_rate, bp_method='minimum_sum',
                        ms_scaling_factor=ms_scaling, max_iter=max_iter,
                        osd_method='osd0', osd_order=0)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        return np.array([self.dec.decode(s) for s in syndromes], np.uint8)


if __name__ == "__main__":
    import time
    hx, hz = gross_code()
    print("gross code:", code_parameters(hx, hz))
    for p in (0.02, 0.04, 0.06, 0.08):
        t0 = time.time()
        r = code_capacity_ler(hx, hz, p, shots=2000, seed=1)
        print(f"p={p:.2f}  LER={r['ler']:.4f} ({r['errors']}/{r['shots']})  "
              f"{time.time() - t0:.1f}s")


def make_decoder(H: np.ndarray, error_rate: float, backend: str = "auto", **kw):
    """
    BP+OSD-0 decoder with a ``decode_batch(syndromes)`` method.

    backend: "torch-cuda" (batched BP on GPU), "ldpc" (Roffe's C++ BP+OSD,
    fastest on CPU), "torch-cpu", "numpy", or "auto" = the first of
    torch-cuda / ldpc / numpy that is available. All make the same
    decisions up to floating-point ties.
    """
    def has_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def has_ldpc():
        try:
            import ldpc  # noqa: F401
            return True
        except ImportError:
            return False

    if backend == "auto":
        backend = "torch-cuda" if has_cuda() else "ldpc" if has_ldpc() else "numpy"
    if backend in ("torch-cuda", "torch-cpu"):
        from phase2_noise.qldpc_gpu import TorchBpOsdDecoder
        return TorchBpOsdDecoder(H, error_rate, device=backend.split("-")[1], **kw)
    if backend == "ldpc":
        return LdpcReference(H, error_rate, **kw)
    if backend == "numpy":
        return BpOsdDecoder(H, error_rate, **kw)
    raise ValueError(f"unknown decoder backend {backend!r}")
