"""
Phase B-2: Monte-Carlo Quantum Trajectory Simulator
====================================================
Replaces the O(4^n) global density-matrix evolution with pure-state
trajectories: each shot evolves a ket of size 2^n, sampling channel
outcomes as they occur. Pauli channels (depolarizing) are pre-sampled
vectorized over shots; general channels (amplitude/phase damping) sample
their Kraus index on the fly from state-dependent probabilities.

Memory per shot: O(2^n)  vs  O(4^n) for density matrix
=> noisy simulation scales from ~15 qubits to ~28+ qubits at equal memory.

Ops vocabulary (same style as the Clifford sampler):
    ('h',q) ('x',q) ('y',q) ('z',q) ('s',q) ('t',q)
    ('rx',theta,q) ('ry',theta,q) ('rz',theta,q)
    ('cnot',a,b) ('mz',q)
    ('depolarize',q,p)   ('amp_damp',q,gamma)   ('phase_damp',q,lam)
"""

from typing import List, Optional
import numpy as np


class TrajectorySimulator:
    """Pure-state Monte-Carlo trajectory sampler for noisy circuits."""

    def __init__(self, n_qubits: int):
        self.n = n_qubits

    # ---- channel primitives on a strided view of the state ----
    def _amp_damp(self, state: np.ndarray, q: int, gamma: float,
                  u: float) -> None:
        k = self.n - 1 - q
        st = state.reshape(2 ** k, 2, 2 ** q)
        p_jump = gamma * float(np.sum(np.abs(st[:, 1, :]) ** 2))
        if u < p_jump:
            # K1|psi> / sqrt(p): |1> component moves to |0>, amp0 discarded
            st[:, 0, :] = st[:, 1, :]
            st[:, 1, :] = 0.0
            st[:, :, :] *= np.sqrt(gamma / p_jump) if p_jump > 0 else 0.0
        else:
            sc = 1.0 / np.sqrt(1 - p_jump) if p_jump < 1 else 1.0
            st[:, 0, :] *= sc                      # amp0 also renormalized
            st[:, 1, :] *= np.sqrt(1 - gamma) * sc

    def _phase_damp(self, state: np.ndarray, q: int, lam: float,
                    u: float) -> None:
        k = self.n - 1 - q
        st = state.reshape(2 ** k, 2, 2 ** q)
        p = lam * float(np.sum(np.abs(st[:, 1, :]) ** 2))
        if u < p:
            # K1 = diag(0, sqrt(lam)): |1> -> |0> keeping phase, amp0 discarded
            st[:, 0, :] = st[:, 1, :]
            st[:, 1, :] = 0.0
            st[:, :, :] *= np.sqrt(lam / p) if p > 0 else 0.0
        else:
            sc = 1.0 / np.sqrt(1 - p) if p < 1 else 1.0
            st[:, 0, :] *= sc                      # amp0 also renormalized
            st[:, 1, :] *= np.sqrt(1 - lam) * sc

    # ---- main entry ----
    def sample(self, ops: List[tuple], shots: int,
               seed: Optional[int] = None,
               keep_states: bool = False):
        """
        Returns (shots, n_measurements) uint8 outcomes.
        With keep_states=True also returns the final kets (shots, 2^n).
        """
        from phase1_basic.simulators import StateVectorSimulator

        rng = np.random.default_rng(seed)
        n_meas = sum(1 for o in ops if o[0] == 'mz')

        # pre-sample Pauli (depolarizing) events vectorized over shots
        depol = [o for o in ops if o[0] == 'depolarize']
        # per depolarizing op, 3 Bernoulli(p/3) bit-vectors over shots
        depol_code = {}
        for idx, o in enumerate(depol):
            u = rng.random(shots)
            p3 = o[2] / 3.0
            # exactly one of I/X/Y/Z with probs 1-p, p/3, p/3, p/3
            depol_code[id(o)] = np.where(u < p3, 1, np.where(u < 2 * p3, 2,
                                          np.where(u < o[2], 3, 0)))

        # pre-draw uniforms for the on-the-fly channels
        n_ad = sum(1 for o in ops if o[0] == 'amp_damp')
        n_pd = sum(1 for o in ops if o[0] == 'phase_damp')
        u_ad = rng.random((n_ad, shots)) if n_ad else None
        u_pd = rng.random((n_pd, shots)) if n_pd else None

        outcomes = np.zeros((shots, n_meas), dtype=np.uint8)
        states = np.zeros((shots, 2 ** self.n), dtype=complex) if keep_states else None
        ad_i = pd_i = 0

        for s in range(shots):
            sim = StateVectorSimulator(self.n)
            m = 0
            for o in ops:
                tag = o[0]
                if tag == 'mz':
                    outcomes[s, m] = sim.measure(o[1])
                    m += 1
                elif tag == 'depolarize':
                    code = depol_code[id(o)][s]
                    if code == 1:
                        sim.x(o[1])
                    elif code == 2:
                        sim.y(o[1])
                    elif code == 3:
                        sim.z(o[1])
                elif tag == 'amp_damp':
                    sim._flush_all()   # channels need the flushed state
                    self._amp_damp(sim.state, o[1], o[2], u_ad[ad_i, s])
                    ad_i += 1
                elif tag == 'phase_damp':
                    sim._flush_all()
                    self._phase_damp(sim.state, o[1], o[2], u_pd[pd_i, s])
                    pd_i += 1
                elif tag in ('rx', 'ry', 'rz'):
                    getattr(sim, tag)(o[1], o[2])
                else:
                    getattr(sim, tag)(*o[1:])
            sim._flush_all()          # flush buffered fused 1q gates
            if keep_states:
                states[s] = sim.state
            ad_i = pd_i = 0

        if keep_states:
            return outcomes, states
        return outcomes


def exact_noisy_expectation(ops: List[tuple], pauli: str,
                            n_qubits: int) -> float:
    """
    Exact reference: evolve |0...0><0...0| under the noisy ops with the
    density-matrix channel library and return <pauli>.
    Only channels with existing exact implementations are supported.
    """
    from phase2_noise.noise_and_ec import NoiseChannels
    import numpy as np
    from phase1_basic.simulators import StateVectorSimulator

    rho = np.zeros((2 ** n_qubits,) * 2, dtype=complex)
    rho[0, 0] = 1.0
    for o in ops:
        tag = o[0]
        if tag == 'h':
            # conjugate by H
            rho = _conj(rho, np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), o[1], n_qubits)
        elif tag == 'x':
            rho = _conj(rho, np.array([[0, 1], [1, 0]], dtype=complex), o[1], n_qubits)
        elif tag == 's':
            rho = _conj(rho, np.array([[1, 0], [0, 1j]], dtype=complex), o[1], n_qubits)
        elif tag == 'cnot':
            rho = _conj2(rho, o[1], o[2], n_qubits)
        elif tag == 'depolarize':
            rho = NoiseChannels.depolarizing(rho, o[2], n_qubits, o[1])
        elif tag == 'amp_damp':
            rho = NoiseChannels.amplitude_damping(rho, o[2], n_qubits, o[1])
        elif tag == 'phase_damp':
            rho = NoiseChannels.phase_damping(rho, o[2], n_qubits, o[1])
        elif tag == 'rx':
            th = o[1]
            g = np.array([[np.cos(th/2), -1j*np.sin(th/2)],
                          [-1j*np.sin(th/2), np.cos(th/2)]], dtype=complex)
            rho = _conj(rho, g, o[2], n_qubits)
        else:
            raise ValueError(f"exact ref: unsupported op {o}")

    # expectation of pauli string
    sim = StateVectorSimulator(n_qubits)
    sim.state = None  # unused
    P = np.array([[1.0]])
    for ch in pauli:
        M = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]),
             'Y': np.array([[0, -1j], [1j, 0]]),
             'Z': np.array([[1, 0], [0, -1]])}[ch]
        P = np.kron(P, M)
    return float(np.real(np.trace(P @ rho)))


def _conj(rho, U, q, n):
    from phase2_noise.noise_and_ec import NoiseChannels
    K = U
    tmp = NoiseChannels._apply_kraus(rho, K, n, q)
    # _apply_kraus does K rho K^dagger
    return tmp


def _conj2(rho, a, b, n):
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    U = np.eye(2 ** n, dtype=complex)
    # build CNOT via kron with qubit order: bit positions
    base = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(2 ** n):
        j = i
        if (i >> a) & 1:
            j ^= 1 << b
        base[j, i] = 1.0
    return base @ rho @ base.conj().T
