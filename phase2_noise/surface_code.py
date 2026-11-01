"""
Phase E1-a: Rotated Surface Code Memory + Threshold
====================================================
Distance-d rotated surface code, Z-basis memory, circuit-level noise.

  layout + hook-safe CNOT schedule (our own generator)
    -> stim circuit with DETECTOR / OBSERVABLE annotations
    -> detector error model (stim) -> PyMatching MWPM
    -> logical error rate per round, d = 3/5/7 threshold crossing

The noise model matches stim's ``surface_code:rotated_memory_z``
generator (after_clifford_depolarization, after_reset_flip_probability,
before_measure_flip_probability, before_round_data_depolarization all
equal to p), so our circuits can be checked shot-for-shot in
distribution against that reference.

Layout (d=3, coordinates are (x, y)):
    data qubits    at odd  (x, y), 1 <= x, y <= 2d-1
    measure qubits at even (x, y), bulk ones touch 4 data qubits,
    boundary ones touch 2 (X type on top/bottom, Z type on left/right).
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


Coord = Tuple[int, int]

# CNOT orders (offsets from the measure qubit). X and Z use mirrored
# "N" / "Z" shapes so hook errors run perpendicular to the logical
# operator they could shorten.
_X_ORDER = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
_Z_ORDER = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


class RotatedSurfaceCode:
    """Geometry of a distance-d rotated surface code."""

    def __init__(self, distance: int):
        if distance < 2:
            raise ValueError("distance must be >= 2")
        self.d = d = distance
        self.data: List[Coord] = [(x, y) for y in range(1, 2 * d, 2)
                                  for x in range(1, 2 * d, 2)]
        data_set = set(self.data)
        self.x_meas: List[Coord] = []
        self.z_meas: List[Coord] = []
        for y in range(0, 2 * d + 1, 2):
            for x in range(0, 2 * d + 1, 2):
                nbrs = [(x + dx, y + dy) for dx, dy in _X_ORDER
                        if (x + dx, y + dy) in data_set]
                is_x = ((x + y) // 2) % 2 == 1
                if len(nbrs) == 4:
                    (self.x_meas if is_x else self.z_meas).append((x, y))
                elif len(nbrs) == 2:
                    top_bottom = y in (0, 2 * d)
                    left_right = x in (0, 2 * d)
                    if is_x and top_bottom and not left_right:
                        self.x_meas.append((x, y))
                    elif not is_x and left_right and not top_bottom:
                        self.z_meas.append((x, y))
        # qubit indices: data first, then X, then Z measure qubits
        self.coords: List[Coord] = self.data + self.x_meas + self.z_meas
        self.index: Dict[Coord, int] = {c: i for i, c in enumerate(self.coords)}
        # logical Z: row y=1 of Z operators (every X check, bulk or on
        # the top boundary, touches that row in 0 or 2 places)
        self.logical_z: List[Coord] = [(x, 1) for x in range(1, 2 * d, 2)]

    @property
    def n_qubits(self) -> int:
        return len(self.coords)

    def stabilizer_support(self, m: Coord, kind: str) -> List[Coord]:
        order = _X_ORDER if kind == 'X' else _Z_ORDER
        data_set = set(self.data)
        return [(m[0] + dx, m[1] + dy) for dx, dy in order
                if (m[0] + dx, m[1] + dy) in data_set]

    def check_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """(H_X, H_Z) over the data qubits."""
        col = {c: i for i, c in enumerate(self.data)}
        hx = np.zeros((len(self.x_meas), len(self.data)), np.uint8)
        hz = np.zeros((len(self.z_meas), len(self.data)), np.uint8)
        for r, m in enumerate(self.x_meas):
            for c in self.stabilizer_support(m, 'X'):
                hx[r, col[c]] = 1
        for r, m in enumerate(self.z_meas):
            for c in self.stabilizer_support(m, 'Z'):
                hz[r, col[c]] = 1
        return hx, hz


def rotated_memory_z_circuit(distance: int, rounds: int, p: float):
    """
    Build a noisy Z-memory experiment as a ``stim.Circuit``.

    Each round: data depolarization, H on X-measure qubits, four CNOT
    layers, H, measure-and-reset of all measure qubits. Detectors compare
    each Z check with the previous round (first round: with the |0>
    initialization); the final round reconstructs Z checks from data
    measurements. Observable 0 = logical Z.
    """
    import stim

    code = RotatedSurfaceCode(distance)
    idx = code.index
    data = [idx[c] for c in code.data]
    xm = [idx[c] for c in code.x_meas]
    zm = [idx[c] for c in code.z_meas]
    meas = xm + zm
    n_meas = len(meas)

    c = stim.Circuit()
    for q, (x, y) in enumerate(code.coords):
        c.append("QUBIT_COORDS", [q], [x, y])

    def noisy(name, targets, p1=None, p2=None):
        c.append(name, targets)
        if p > 0 and p1 is not None:
            c.append("DEPOLARIZE1", targets, p)
        if p > 0 and p2 is not None:
            c.append("DEPOLARIZE2", targets, p)

    c.append("R", data + meas)
    if p > 0:
        c.append("X_ERROR", data + meas, p)

    # CNOT layers: X checks control the data qubit, Z checks are targets
    layers = []
    for k in range(4):
        pairs = []
        for m in code.x_meas:
            dq = (m[0] + _X_ORDER[k][0], m[1] + _X_ORDER[k][1])
            if dq in idx and dq in set(code.data):
                pairs += [idx[m], idx[dq]]
        for m in code.z_meas:
            dq = (m[0] + _Z_ORDER[k][0], m[1] + _Z_ORDER[k][1])
            if dq in idx and dq in set(code.data):
                pairs += [idx[dq], idx[m]]
        layers.append(pairs)

    def one_round(first: bool):
        if p > 0:
            c.append("DEPOLARIZE1", data, p)
        noisy("H", xm, p1=True)
        for pairs in layers:
            noisy("CX", pairs, p2=True)
        noisy("H", xm, p1=True)
        if p > 0:
            c.append("X_ERROR", meas, p)
        c.append("MR", meas)
        if p > 0:
            c.append("X_ERROR", meas, p)
        # Z checks sit at the end of the measurement record
        for j, m in enumerate(code.z_meas):
            rec = -n_meas + len(xm) + j
            targets = [stim.target_rec(rec)]
            if not first:
                targets.append(stim.target_rec(rec - n_meas))
            c.append("DETECTOR", targets, [m[0], m[1], 0])
        if not first:
            for j, m in enumerate(code.x_meas):
                rec = -n_meas + j
                c.append("DETECTOR",
                         [stim.target_rec(rec), stim.target_rec(rec - n_meas)],
                         [m[0], m[1], 0])
        c.append("SHIFT_COORDS", [], [0, 0, 1])

    one_round(first=True)
    if rounds > 1:
        body = stim.Circuit()
        saved, c = c, body
        one_round(first=False)
        c = saved
        c.append(stim.CircuitRepeatBlock(rounds - 1, body))

    if p > 0:
        c.append("X_ERROR", data, p)
    c.append("M", data)
    n_data = len(data)
    data_pos = {coord: i for i, coord in enumerate(code.data)}
    for j, m in enumerate(code.z_meas):
        targets = [stim.target_rec(-n_data + data_pos[dq])
                   for dq in code.stabilizer_support(m, 'Z')]
        targets.append(stim.target_rec(-n_data - n_meas + len(xm) + j))
        c.append("DETECTOR", targets, [m[0], m[1], 1])
    c.append("OBSERVABLE_INCLUDE",
             [stim.target_rec(-n_data + data_pos[dq]) for dq in code.logical_z],
             0)
    return c


def stim_reference_circuit(distance: int, rounds: int, p: float):
    """stim's own generator with the same noise model (cross-check)."""
    import stim
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=rounds,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def logical_error_rate(circuit, shots: int, seed: Optional[int] = None) -> Dict:
    """Sample, decode with PyMatching, return per-shot logical error rate."""
    import pymatching
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    pred = matcher.decode_batch(det)
    errors = int(np.sum(np.any(pred != obs, axis=1)))
    return {'shots': shots, 'errors': errors, 'ler': errors / shots}


def per_round(ler: float, rounds: int) -> float:
    """Convert a per-shot logical error rate to a per-round rate."""
    ler = min(ler, 0.4999)
    return 0.5 * (1 - (1 - 2 * ler) ** (1.0 / rounds))


def threshold_sweep(distances: Sequence[int], ps: Sequence[float],
                    shots: int, seed: int = 0) -> List[Dict]:
    """Logical error per round for each (d, p), rounds = d."""
    rows = []
    for d in distances:
        for i, p in enumerate(ps):
            circ = rotated_memory_z_circuit(d, d, p)
            r = logical_error_rate(circ, shots, seed=seed + 1000 * d + i)
            r.update(d=d, p=p, rounds=d, ler_round=per_round(r['ler'], d))
            rows.append(r)
    return rows


def estimate_crossing(rows: List[Dict], d_small: int, d_large: int) -> Optional[float]:
    """
    Physical error rate where the d_large curve crosses d_small
    (log-log linear interpolation between grid points). None if the
    curves do not cross inside the grid.
    """
    def curve(d):
        pts = sorted((r['p'], r['ler_round']) for r in rows if r['d'] == d)
        return np.array(pts)
    a, b = curve(d_small), curve(d_large)
    diff = np.log(np.maximum(b[:, 1], 1e-12)) - np.log(np.maximum(a[:, 1], 1e-12))
    for i in range(len(diff) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            lp0, lp1 = np.log(a[i, 0]), np.log(a[i + 1, 0])
            t = -diff[i] / (diff[i + 1] - diff[i])
            return float(np.exp(lp0 + t * (lp1 - lp0)))
    return None


if __name__ == "__main__":
    ps = [0.004, 0.006, 0.008, 0.010, 0.012]
    rows = threshold_sweep([3, 5, 7], ps, shots=20000, seed=7)
    for r in rows:
        print(f"d={r['d']} p={r['p']:.3f} errors={r['errors']:5d}/{r['shots']} "
              f"per-round={r['ler_round']:.2e}")
    for a, b in ((3, 5), (5, 7), (3, 7)):
        print(f"crossing d={a} vs d={b}: {estimate_crossing(rows, a, b)}")
