"""
Phase E1-c: Circuit-level syndrome extraction for bivariate bicycle codes
==========================================================================
The depth-8 syndrome cycle of Bravyi et al. (Nature 627, 778, 2024) for
BB codes, as a stim circuit with Z-basis memory detectors and the 12
logical Z observables of the gross code.

Registers (n = number of data qubits, n/2 checks of each type):
    X checks  0 .. n/2-1        (|+>, CNOT controls)
    L data    n/2 .. n-1
    R data    n .. 3n/2-1
    Z checks  3n/2 .. 2n-1      (|0>, CNOT targets)

One cycle (A = A1+A2+A3, B = B1+B2+B3 as permutation matrices):
    round 1: Z <- R[A1^T]                        (X checks prepared)
    round 2: X -> L[A2],  Z <- R[A3^T]
    round 3: X -> R[B2],  Z <- L[B1^T]
    round 4: X -> R[B1],  Z <- L[B2^T]
    round 5: X -> R[B3],  Z <- L[B3^T]
    round 6: X -> L[A1],  Z <- R[A2^T]
    round 7: X -> L[A3],  measure+reset Z checks
    round 8: measure+reset X checks
Every qubit takes part in at most one gate per round. The order follows
the open implementation in gongaa/SlidingWindowDecoder (build_circuit.py),
which reproduces the paper's schedule.

Noise (same knobs as the surface-code module, all equal to p):
DEPOLARIZE2 after every CNOT, X/Z flip after reset and before measurement,
DEPOLARIZE1 on every data qubit once per cycle.
"""

from typing import List, Sequence, Tuple
import numpy as np

from phase2_noise.qldpc import _shift, logical_operators


def bb_monomials(l: int, m: int,
                 a_terms: Sequence[Tuple[str, int]],
                 b_terms: Sequence[Tuple[str, int]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """The three permutation matrices making up A and B."""
    x = np.kron(_shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), _shift(m))

    def mono(var, power):
        base = x if var == 'x' else y
        return np.linalg.matrix_power(base.astype(np.int64), power).astype(np.uint8) & 1

    return [mono(*t) for t in a_terms], [mono(*t) for t in b_terms]


GROSS = (12, 6, [('x', 3), ('y', 1), ('y', 2)], [('y', 3), ('x', 1), ('x', 2)])


def bb_memory_circuit(p: float, cycles: int, code=GROSS):
    """Z-basis memory experiment: ``cycles`` syndrome cycles, then data readout."""
    import stim

    l, m, a_terms, b_terms = code
    (a1, a2, a3), (b1, b2, b3) = bb_monomials(l, m, a_terms, b_terms)
    A = (a1 ^ a2 ^ a3)
    B = (b1 ^ b2 ^ b3)
    hx = np.hstack([A, B])
    hz = np.hstack([B.T, A.T])
    _, lz = logical_operators(hx, hz)

    half = l * m
    n = 2 * half
    XC, L, R, ZC = 0, half, n, n + half

    def col(mat):      # column index of the single 1 in each row
        return np.argmax(mat, axis=1)

    A1, A2, A3 = col(a1), col(a2), col(a3)
    B1, B2, B3 = col(b1), col(b2), col(b3)
    A1T, A2T, A3T = col(a1.T), col(a2.T), col(a3.T)
    B1T, B2T, B3T = col(b1.T), col(b2.T), col(b3.T)
    I = np.arange(half)
    xc, zc = XC + I, ZC + I

    def cx(c, controls, targets):
        pairs = np.stack([controls, targets], 1).reshape(-1).tolist()
        c.append("CX", pairs)
        if p > 0:
            c.append("DEPOLARIZE2", pairs, p)

    def cycle(first: bool):
        c = stim.Circuit()
        # round 1
        if p > 0:
            c.append("DEPOLARIZE1", (L + I).tolist() + (R + I).tolist(), p)
        if first:
            c.append("H", xc.tolist())
            if p > 0:
                c.append("DEPOLARIZE1", xc.tolist(), p)
        cx(c, R + A1T, zc)
        c.append("TICK")
        cx(c, np.concatenate([xc, R + A3T]), np.concatenate([L + A2, zc])); c.append("TICK")
        cx(c, np.concatenate([xc, L + B1T]), np.concatenate([R + B2, zc])); c.append("TICK")
        cx(c, np.concatenate([xc, L + B2T]), np.concatenate([R + B1, zc])); c.append("TICK")
        cx(c, np.concatenate([xc, L + B3T]), np.concatenate([R + B3, zc])); c.append("TICK")
        cx(c, np.concatenate([xc, R + A2T]), np.concatenate([L + A1, zc])); c.append("TICK")
        # round 7: last X CNOTs, measure+reset Z checks
        cx(c, xc, L + A3)
        if p > 0:
            c.append("X_ERROR", zc.tolist(), p)
        c.append("MR", zc.tolist())
        if p > 0:
            c.append("X_ERROR", zc.tolist(), p)
        for i in range(half):
            t = [stim.target_rec(-half + i)]
            if not first:
                t.append(stim.target_rec(-3 * half + i))     # same Z check, previous cycle
            c.append("DETECTOR", t, [i, 0])
        c.append("TICK")
        # round 8: measure+reset X checks (X basis)
        if p > 0:
            c.append("Z_ERROR", xc.tolist(), p)
        c.append("MRX", xc.tolist())
        if p > 0:
            c.append("Z_ERROR", xc.tolist(), p)
        c.append("SHIFT_COORDS", [], [0, 1])
        c.append("TICK")
        return c

    c = stim.Circuit()
    c.append("R", list(range(2 * n)))
    if p > 0:
        c.append("X_ERROR", list(range(2 * n)), p)
    c.append("TICK")
    c += cycle(first=True)
    if cycles > 1:
        c.append(stim.CircuitRepeatBlock(cycles - 1, cycle(first=False)))
    # data readout in Z
    data = list(range(L, L + n))
    if p > 0:
        c.append("X_ERROR", data, p)
    c.append("M", data)
    # last Z-check record: before the final X-check MRX block
    for i in range(half):
        t = [stim.target_rec(-n + int(q)) for q in np.nonzero(hz[i])[0]]
        t.append(stim.target_rec(-n - 2 * half + i))
        c.append("DETECTOR", t, [i, 1])
    for k, row in enumerate(lz):
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-n + int(q)) for q in np.nonzero(row)[0]], k)
    return c


def dem_matrices(circuit):
    """(H, L, priors) of the circuit's detector error model, one column per error mechanism."""
    import stim
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    cols_d, cols_o, priors = [], [], []
    for inst in dem:
        if inst.type != "error":
            continue
        d, o = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                d.append(t.val)
            elif t.is_logical_observable_id():
                o.append(t.val)
        cols_d.append(d); cols_o.append(o); priors.append(inst.args_copy()[0])
    H = np.zeros((dem.num_detectors, len(priors)), np.uint8)
    Lm = np.zeros((dem.num_observables, len(priors)), np.uint8)
    for j, (d, o) in enumerate(zip(cols_d, cols_o)):
        H[d, j] = 1
        Lm[o, j] = 1
    return H, Lm, np.array(priors)
