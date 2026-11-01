"""
Phase E5-c: NVIDIA cuStateVec state-vector backend
===================================================
Same API and qubit convention as ``TorchSVSimulator`` (little-endian,
fused single-qubit gates), but every gate goes through cuStateVec's
``apply_matrix`` kernel, which runs near peak memory bandwidth and works
in place (no second state-sized buffer, so one more qubit fits).

The state still lives in a torch CUDA tensor, so everything else in the
torch backend (measurement, probabilities, fidelity) keeps working.
Requires ``pip install cuquantum-python-cu12``; ``available()`` reports
whether it can be used.
"""

from typing import Optional
import numpy as np
import torch

from phase1_basic.gpu_backend import TorchSVSimulator

try:                                                  # cuquantum >= 25
    from cuquantum.bindings import custatevec as cusv
except ImportError:                                   # older layouts
    try:
        from cuquantum import custatevec as cusv
    except ImportError:
        cusv = None

# enum values from the CUDA / cuStateVec headers (stable across releases)
CUDA_C_64F = 5
COMPUTE_64F = 1 << 4
LAYOUT_ROW = 1

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def available() -> bool:
    return cusv is not None and torch.cuda.is_available()


class CuStateVecSimulator(TorchSVSimulator):
    """State-vector simulator whose gate kernels are cuStateVec."""

    def __init__(self, n_qubits: int, device: Optional[torch.device] = None):
        if not available():
            raise RuntimeError("cuStateVec needs cuquantum-python and a CUDA GPU")
        super().__init__(n_qubits, device or torch.device("cuda"))
        self.handle = cusv.create()
        self._ws = None
        self._ws_size = 0

    def __del__(self):
        h = getattr(self, "handle", None)
        if h is not None and cusv is not None:
            cusv.destroy(h)
            self.handle = None

    # ---- the one kernel everything goes through ----
    def _apply(self, matrix: np.ndarray, targets, controls=()):
        mat = np.ascontiguousarray(matrix, dtype=np.complex128)
        nt, nc = len(targets), len(controls)
        need = cusv.apply_matrix_get_workspace_size(
            self.handle, CUDA_C_64F, self.n, mat.ctypes.data, CUDA_C_64F,
            LAYOUT_ROW, 0, nt, nc, COMPUTE_64F)
        if need > self._ws_size:
            self._ws = torch.empty(need, dtype=torch.uint8, device=self.device)
            self._ws_size = need
        cusv.apply_matrix(
            self.handle, self.state.data_ptr(), CUDA_C_64F, self.n,
            mat.ctypes.data, CUDA_C_64F, LAYOUT_ROW, 0,
            list(targets), nt, list(controls), [1] * nc, nc, COMPUTE_64F,
            self._ws.data_ptr() if need else 0, need)
        self.gate_count += 1

    # ---- overrides of the torch engine's state updates ----
    def _flush_1q(self, target=None):
        if target is None:
            for t in range(self.n):
                if self._fused[t] is not None:
                    self._flush_1q(t)
            return
        G = self._fused[target]
        if G is None:
            return
        self._fused[target] = None
        self.gate_count -= 1                      # counted when buffered
        self._apply(G.detach().cpu().numpy(), [target])

    def rz(self, th, q):                          # fuse instead of a pass
        self._buffer_1q(np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2)]), q)

    def x(self, q):
        self._buffer_1q(_X, q)

    def z(self, q):
        self._buffer_1q(_Z, q)

    def cnot(self, c, t):
        self._flush_1q(c); self._flush_1q(t)
        self._apply(_X, [t], [c])

    def cz(self, c, t):
        self._flush_1q(c); self._flush_1q(t)
        self._apply(_Z, [t], [c])

    def _apply_2q(self, gate, q1, q2):
        # matrix basis index = b(q2) + 2 b(q1) for targets [q2, q1]
        self._flush_1q(q1); self._flush_1q(q2)
        self._apply(np.asarray(gate, np.complex128), [q2, q1])
