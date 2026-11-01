"""
Phase B-3: Detector Error Model + PyMatching Decoder Pipeline
==============================================================
End-to-end QEC workflow, all on our own engines:

  Circuit IR (B-1)
    -> noise model -> Clifford affine sampler (P0)   [shot generation]
    -> detector construction (syndrome differences)
    -> fault enumeration by Pauli propagation        [DEM construction]
    -> PyMatching MWPM decode                        [correction]
    -> logical error rate vs physical error rate

Demonstrated code: distance-d Z-basis repetition memory (X errors),
the standard MWPM tutorial setting. The pipeline is code-agnostic:
any stabilizer code built through Circuit works the same way.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from phase1_basic.circuit import Circuit
from phase1_basic.simulators import CliffordSimulator


# ----------------------------------------------------------------------
# Repetition-code memory experiment
# ----------------------------------------------------------------------
class RepetitionCodeMemory:
    """
    Z-basis repetition code: |0_L> = |0...0>, logical Z = Z^⊗d.
    Rounds of ZZ parity checks on d-1 ancillas, then data measurement.

    Qubit map: data 0..d-1, ancilla d..2d-2.
    """

    def __init__(self, distance: int, rounds: int):
        self.d = distance
        self.R = rounds
        self.n_data = distance
        self.n_anc = distance - 1
        self.n = 2 * distance - 1

    def build_circuit(self) -> Circuit:
        circ = Circuit(self.n, name=f"rep_d{self.d}_R{self.R}")
        cb = 0
        for _r in range(self.R):
            for i in range(self.n_anc):
                a = self.n_data + i
                circ.cnot(i, a)
                circ.cnot(i + 1, a)
                circ.measure(a, cbit=cb)
                cb += 1
        for q in range(self.n_data):
            circ.measure(q, cbit=cb)
            cb += 1
        return circ

    def syndrome_index(self, rnd: int, check: int) -> int:
        return rnd * self.n_anc + check

    def detector_count(self) -> int:
        # D(r,i) = s_r(i) xor s_{r-1}(i); round 0 compares to implicit 0;
        # final round R: parity recomputed from data measurements xor s_{R-1}
        return (self.R + 1) * self.n_anc

    def detector_matrix(self, outcomes: np.ndarray) -> np.ndarray:
        """(shots, n_meas) outcomes -> (shots, n_detector) detection events."""
        shots = outcomes.shape[0]
        raw = outcomes[:, :self.R * self.n_anc].reshape(shots, self.R, self.n_anc)
        # ancillas are not reset between rounds: m_r = parity_r xor m_{r-1}
        anc = raw.copy()
        anc[:, 1:, :] = raw[:, 1:, :] ^ raw[:, :-1, :]
        data = outcomes[:, self.R * self.n_anc:]
        det = np.zeros((shots, self.R + 1, self.n_anc), dtype=np.uint8)
        det[:, 0, :] = anc[:, 0, :]
        det[:, 1:self.R, :] = anc[:, 1:, :] ^ anc[:, :-1, :]
        det[:, self.R, :] = (data[:, :-1] ^ data[:, 1:]) ^ anc[:, -1, :]
        return det.reshape(shots, self.detector_count())

    def actual_logical(self, outcomes: np.ndarray) -> np.ndarray:
        """Logical Z = Z on data qubit 0 (1 = logical error for |0_L>)."""
        return outcomes[:, self.R * self.n_anc].astype(np.uint8)


# ----------------------------------------------------------------------
# Fault enumeration: build the DEM by Pauli propagation
# ----------------------------------------------------------------------
class DemBuilder:
    """
    Enumerate every single-fault location (X error after each gate, on
    each qubit the gate touches), propagate through the noiseless
    circuit with the Clifford simulator, and record which detectors
    flip and whether the logical observable flips.
    """

    def __init__(self, experiment: RepetitionCodeMemory,
                 circuit: Circuit):
        self.exp = experiment
        self.circ = circuit
        self.ops = circuit.to_sampler_ops()

    def _run_with_fault(self, fault_pos: int, fault_qubit: int,
                        mz_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Run once with an X fault inserted; return (all outcomes, ...)."""
        sim = CliffordSimulator(self.exp.n)
        m = 0
        for gi, op in enumerate(self.ops):
            tag = op[0]
            if gi == fault_pos:
                sim.pauli_x(fault_qubit)
            if tag == 'cnot':
                sim.cnot(op[1], op[2])
            elif tag == 'mz':
                sim.measure(op[1])
                m += 1
            elif tag in ('h', 's'):
                getattr(sim, tag)(op[1])
        # collect all measurement outcomes: rerun recording (deterministic
        # outcomes are identical; only random ones need the same stream).
        # Simpler: measure() above already consumed outcomes but discarded
        # them; redo with recording.
        return self._record_run(fault_pos, fault_qubit)

    def _record_run(self, fault_pos: int, fault_qubit: int) -> np.ndarray:
        sim = CliffordSimulator(self.exp.n)
        outs = []
        for gi, op in enumerate(self.ops):
            if gi == fault_pos:
                sim.pauli_x(fault_qubit)
            if op[0] == 'cnot':
                sim.cnot(op[1], op[2])
            elif op[0] == 'mz':
                outs.append(sim.measure(op[1]))
            elif op[0] in ('h', 's'):
                getattr(sim, op[0])(op[1])
        return np.array(outs, dtype=np.uint8)

    def enumerate_faults(self, p_gate: float = 0.0, p_meas: float = 0.0
                         ) -> List[Tuple[str, Tuple[int, ...], bool, float]]:
        """
        Returns list of (label, detector_tuple, logical_flip, prob).
        Detector indices are flat (round * n_anc + check).
        prob: X-type fault probability at this location (X or Y component
        of gate depolarizing = 2p_gate/3 on data, p_meas on ancilla).
        """
        ref = self._record_run(-1, -1)
        ref_det = self.exp.detector_matrix(ref[None, :])[0]
        ref_log = self.exp.actual_logical(ref[None, :])[0]

        faults = []
        for gi, op in enumerate(self.ops):
            if op[0] == 'mz':
                qs = [op[1]]
            else:
                qs = list(op[1:])
            for q in qs:
                out = self._record_run(gi, q)
                det = self.exp.detector_matrix(out[None, :])[0]
                log = self.exp.actual_logical(out[None, :])[0]
                flipped = tuple(int(i) for i in np.where(det ^ ref_det)[0])
                logical = bool(log ^ ref_log)
                prob = (2 * p_gate / 3) if q < self.exp.n_data else p_meas
                faults.append((f"X after op{gi}({op[0]}) q{q}",
                               flipped, logical, prob))
        return faults


# ----------------------------------------------------------------------
# Decode with PyMatching
# ----------------------------------------------------------------------
class Matcher:
    """Build a PyMatching graph from enumerated faults and decode shots.

    PyMatching `fault_ids` are OBSERVABLE indices: an edge carries {0}
    when its fault flips the logical observable, {} otherwise; decoding
    returns the predicted observable flips directly.
    """

    def __init__(self, faults: List[Tuple[str, Tuple[int, ...], bool, float]],
                 n_detectors: int):
        import pymatching
        self.faults = faults
        self.m = pymatching.Matching()

        # Several single-fault locations can produce the SAME detector
        # pattern (e.g. measurement error vs late boundary data error) --
        # indistinguishable by syndrome, so they share one graph edge.
        groups: Dict[Tuple[int, ...], List[int]] = {}
        for fid, (_, dets, _log, _p) in enumerate(faults):
            if dets:
                groups.setdefault(tuple(sorted(dets)), []).append(fid)

        def merged(fids):
            ps = np.array([max(faults[f][3], 0.0) for f in fids])
            flips = np.array([faults[f][2] for f in fids])
            if ps.sum() > 0:
                p = float(ps.sum())
                logical = bool(ps[flips].sum() > ps[~flips].sum())
                w = float(np.log((1 - p) / p)) if p < 0.5 else 0.0
            else:                       # no rates: uniform weights, majority flag
                logical = bool(flips.sum() * 2 > len(flips))
                w = 1.0
            return logical, w

        graphlike = {k: merged(v) for k, v in groups.items() if len(k) <= 2}
        for key, fids in groups.items():
            if len(key) <= 2:
                continue
            # hyperedge: if it splits into existing graph-like edges MWPM
            # already covers it; otherwise add it as consecutive pairs
            if not self._decomposable(key, graphlike):
                logical, w = merged(fids)
                pairs = [key[i:i + 2] for i in range(0, len(key), 2)]
                for k, pr in enumerate(pairs):
                    cur = graphlike.get(pr)
                    if cur is None or cur[1] > w:
                        graphlike[pr] = (logical if k == 0 else False, w)

        for key, (logical, w) in graphlike.items():
            obs = {0} if logical else set()
            if len(key) == 1:
                self.m.add_boundary_edge(key[0], weight=w, fault_ids=obs,
                                         merge_strategy="smallest-weight")
            else:
                self.m.add_edge(key[0], key[1], weight=w, fault_ids=obs,
                                merge_strategy="smallest-weight")
        self.n_detectors = n_detectors

    @staticmethod
    def _decomposable(key, edges) -> bool:
        if not key:
            return True
        a = key[0]
        if (a,) in edges and Matcher._decomposable(key[1:], edges):
            return True
        for b in key[1:]:
            if (a, b) in edges:
                rest = tuple(x for x in key[1:] if x != b)
                if Matcher._decomposable(rest, edges):
                    return True
        return False

    def decode(self, detection_events: np.ndarray) -> np.ndarray:
        """Returns predicted logical flips (shots,)."""
        width = self.m.num_detectors
        if detection_events.shape[1] < width:
            pad = np.zeros((detection_events.shape[0],
                            width - detection_events.shape[1]), np.uint8)
            detection_events = np.concatenate([detection_events, pad], axis=1)
        elif detection_events.shape[1] > width:
            detection_events = detection_events[:, :width]
        pred = self.m.decode_batch(detection_events)
        if pred.ndim == 1 or pred.shape[1] == 0:
            return np.zeros(len(detection_events), dtype=np.uint8)
        return pred[:, 0].astype(np.uint8)


# ----------------------------------------------------------------------
# Full experiment driver
# ----------------------------------------------------------------------
def run_memory_experiment(distance: int, rounds: int, shots: int,
                          p_gate: float, p_meas: float,
                          seed: Optional[int] = None) -> Dict:
    """
    Returns dict with logical error rates (decoded / undecoded) and the
    enumerated fault model.
    """
    exp = RepetitionCodeMemory(distance, rounds)
    circ = exp.build_circuit()

    # sample noisy shots
    noise = {'after_2q': ('depolarize', p_gate), 'after_1q': ('depolarize', p_gate)}
    # measurement errors: model as X on ancilla before measurement is hard
    # via this noise hook; keep p_meas for API symmetry (0 default use)
    ops = circ.to_sampler_ops(noise=noise)
    if p_meas > 0:
        noisy_ops = []
        for op in ops:
            if op[0] == 'mz' and op[1] >= exp.n_data:   # ancilla measurement
                noisy_ops.append(('x_error', op[1], p_meas))
            noisy_ops.append(op)
        ops = noisy_ops
    outcomes = CliffordSimulator(exp.n).sample(ops, shots=shots, seed=seed)

    det = exp.detector_matrix(outcomes)
    logical = exp.actual_logical(outcomes)

    builder = DemBuilder(exp, circ)
    faults = builder.enumerate_faults(p_gate=p_gate, p_meas=p_meas)
    matcher = Matcher(faults, exp.detector_count())
    pred_logical = matcher.decode(det)

    decoded_err = float(np.mean(pred_logical ^ logical))
    undecoded_err = float(np.mean(logical))
    return {
        'distance': distance, 'rounds': rounds, 'shots': shots,
        'p_gate': p_gate, 'p_meas': p_meas,
        'logical_error_decoded': decoded_err,
        'logical_error_undecoded': undecoded_err,
        'n_faults': len(faults),
        'mean_detection_events': float(det.mean()),
    }


if __name__ == "__main__":
    for p in (0.001, 0.005, 0.02):
        r = run_memory_experiment(distance=3, rounds=3, shots=2000,
                                  p_gate=p, p_meas=0.0, seed=11)
        print(f"p={p:g}  decoded={r['logical_error_decoded']:.4f}  "
              f"undecoded={r['logical_error_undecoded']:.4f}  "
              f"faults={r['n_faults']}  det/shots={r['mean_detection_events']:.3f}")
