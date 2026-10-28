"""
Phase D: Torch-accelerated state-vector backend
=================================================
Device-agnostic state-vector engine written in PyTorch: runs on CUDA
when available (torch.cuda.is_available()) and on CPU otherwise, so the
exact same code path is validated here and accelerated on GPU machines.

Rationale: this environment has no NVIDIA GPU visible, but ships
torch 2.8.0+cu128 -- torch gives us (a) a single code path for both
devices, (b) future batched-shot vectorization, (c) easy interop with
the existing numpy ecosystem.

Expected GPU behaviour (from the survey, arXiv:2308.01999): gate
application is memory-bandwidth bound; a single modern GPU raises the
state-vector ceiling from ~2^27 (CPU) to ~2^33-2^35 complex128
amplitudes.
"""

from typing import Dict, Optional
import numpy as np
import torch


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchSVSimulator:
    """State-vector simulator with the same conventions as the numpy
    engine (little-endian qubit 0, fused single-qubit gates)."""

    def __init__(self, n_qubits: int, device: Optional[torch.device] = None):
        self.n = n_qubits
        self.N = 2 ** n_qubits
        self.device = device or default_device()
        self.state = torch.zeros(self.N, dtype=torch.complex128,
                                 device=self.device)
        self.state[0] = 1.0
        self._fused = [None] * n_qubits
        self.gate_count = 0

    # ---- fusion (same strategy as the numpy engine) ----
    def _buffer_1q(self, gate, target):
        g = torch.as_tensor(gate, dtype=torch.complex128, device=self.device)
        p = self._fused[target]
        self._fused[target] = g if p is None else g @ p
        self.gate_count += 1

    def _flush_1q(self, target=None):
        if target is None:
            for t in range(self.n):
                if self._fused[t] is not None:
                    self._flush_1q(t)
            return
        G = self._fused[target]
        if G is None:
            return
        k = self.n - 1 - target
        st = self.state.reshape(2 ** k, 2, 2 ** target)
        self.state = torch.einsum('ab,kbj->kaj', G, st).reshape(self.N)
        self._fused[target] = None

    def _flush_all(self):
        self._flush_1q()

    # ---- gates ----
    def h(self, q): self._buffer_1q(np.array([[1, 1], [1, -1]]) / np.sqrt(2), q)
    def y(self, q): self._buffer_1q(np.array([[0, -1j], [1j, 0]]), q)
    def s(self, q): self._buffer_1q(np.array([[1, 0], [0, 1j]]), q)
    def t(self, q): self._buffer_1q(np.array([[1, 0], [0, np.exp(1j*np.pi/4)]]), q)
    def rx(self, th, q):
        self._buffer_1q(np.array([[np.cos(th/2), -1j*np.sin(th/2)],
                                  [-1j*np.sin(th/2), np.cos(th/2)]]), q)
    def ry(self, th, q):
        self._buffer_1q(np.array([[np.cos(th/2), -np.sin(th/2)],
                                  [np.sin(th/2), np.cos(th/2)]]), q)

    def x(self, q):
        if self._fused[q] is not None: self._flush_1q(q)
        k = self.n - 1 - q
        st = self.state.reshape(2 ** k, 2, 2 ** q)
        st[:, 0, :], st[:, 1, :] = st[:, 1, :].clone(), st[:, 0, :].clone()
        self.gate_count += 1

    def z(self, q):
        if self._fused[q] is not None: self._flush_1q(q)
        k = self.n - 1 - q
        st = self.state.reshape(2 ** k, 2, 2 ** q)
        st[:, 1, :] *= -1
        self.gate_count += 1

    def rz(self, th, q):
        if self._fused[q] is not None: self._flush_1q(q)
        k = self.n - 1 - q
        st = self.state.reshape(2 ** k, 2, 2 ** q)
        st[:, 0, :] *= np.exp(-1j * th / 2)
        st[:, 1, :] *= np.exp(1j * th / 2)
        self.gate_count += 1

    def _apply_2q(self, gate, q1, q2):
        if q1 < q2:
            P = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            gate = P @ gate @ P
            q1, q2 = q2, q1
        self._flush_all()
        g = torch.as_tensor(gate, dtype=torch.complex128, device=self.device)
        st = self.state.reshape(2 ** (self.n - 1 - q1), 2,
                                2 ** (q1 - 1 - q2), 2, 2 ** q2)
        T = st.permute(1, 3, 0, 2, 4).reshape(4, -1)
        out = (g.reshape(4, 4) @ T).reshape(2, 2, st.shape[0], st.shape[2],
                                            st.shape[4])
        self.state = out.permute(2, 0, 3, 1, 4).reshape(self.N).contiguous()
        self.gate_count += 1

    def cnot(self, c, t):
        g = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        self._apply_2q(g, c, t)

    def cz(self, c, t):
        self._apply_2q(np.diag([1, 1, 1, -1]).astype(complex), c, t)

    # ---- measurement / analysis ----
    def measure(self, target: int, shots: int = 1):
        self._flush_all()
        k = self.n - 1 - target
        st = self.state.reshape(2 ** k, 2, 2 ** target)
        p0 = float(torch.sum(torch.abs(st[:, 0, :]) ** 2).cpu())
        r = int(torch.rand(1, device=self.device).item() < (1 - p0))
        if r == 0:
            st[:, 1, :] = 0
            st[:, 0, :] /= np.sqrt(p0) if p0 > 0 else 1
        else:
            st[:, 0, :] = 0
            st[:, 1, :] /= np.sqrt(1 - p0) if p0 < 1 else 1
        return r

    def get_probabilities(self) -> np.ndarray:
        self._flush_all()
        return (torch.abs(self.state) ** 2).cpu().numpy()

    def fidelity(self, other_state: np.ndarray) -> float:
        self._flush_all()
        o = torch.as_tensor(other_state, dtype=torch.complex128,
                            device=self.device)
        return float(torch.abs(torch.dot(torch.conj(self.state), o)) ** 2)

    def benchmark(self) -> Dict:
        self._flush_all()
        return {
            'n_qubits': self.n,
            'device': str(self.device),
            'memory_mb': self.state.nbytes / (1024 ** 2),
            'gate_count': self.gate_count,
        }


class TorchSVBackend:
    """Adapter for the backends registry."""
    name = "torch-statevector"

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device

    def run(self, circuit):
        from phase1_basic.circuit import Circuit
        sim = TorchSVSimulator(circuit.n_qubits, device=self.device)
        for g in circuit.gates:
            if g.name == 'measure':
                sim.measure(g.targets[0])
            elif g.name == 'cx':
                sim.cnot(*g.targets)
            elif g.name in ('rx', 'ry', 'rz'):
                getattr(sim, g.name)(g.params[0], g.targets[0])
            elif g.name == 'u3':
                th, ph, lm = g.params
                sim.rz(lm, g.targets[0])
                sim.ry(th, g.targets[0])
                sim.rz(ph, g.targets[0])
            else:
                getattr(sim, g.name)(*g.targets)
        return sim.state.cpu().numpy()
