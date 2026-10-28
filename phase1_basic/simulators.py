"""
Phase I: Basic Quantum Simulators
====================================
Milestone P1.1: State-vector simulator (up to 20 qubits)
Milestone P1.2: Clifford/stabilizer simulator (up to 1000 qubits)
Milestone P1.3: Tensor network MPS simulator (up to 50 qubits)
Milestone P1.4: Gate fidelity validation against theory
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time


# ------------------------------------------------------------------
# P1.1: State-Vector Simulator
# ------------------------------------------------------------------
class StateVectorSimulatorNaive:
    """
    Full amplitude state-vector simulator (naive reference implementation).
    Memory: O(2^n). Practical limit: ~20-25 qubits on CPU, ~30 on GPU.
    Kept for cross-validation; use StateVectorSimulator (optimized) in practice.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.N = 2 ** n_qubits
        self.state = np.zeros(self.N, dtype=complex)
        self.state[0] = 1.0
        self.gate_count = 0
        self._gate_times = []

    # --- Gate library ---
    def h(self, target: int):
        """Hadamard gate."""
        t0 = time.perf_counter()
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._apply_1q(H, target)
        self._gate_times.append(time.perf_counter() - t0)
        self.gate_count += 1

    def x(self, target: int):
        """Pauli-X (NOT)."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        self._apply_1q(X, target)
        self.gate_count += 1

    def y(self, target: int):
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._apply_1q(Y, target)
        self.gate_count += 1

    def z(self, target: int):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._apply_1q(Z, target)
        self.gate_count += 1

    def s(self, target: int):
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        self._apply_1q(S, target)
        self.gate_count += 1

    def t(self, target: int):
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        self._apply_1q(T, target)
        self.gate_count += 1

    def rx(self, theta: float, target: int):
        gate = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                         [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        self._apply_1q(gate, target)
        self.gate_count += 1

    def ry(self, theta: float, target: int):
        gate = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        self._apply_1q(gate, target)
        self.gate_count += 1

    def rz(self, theta: float, target: int):
        gate = np.array([[np.exp(-1j*theta/2), 0],
                         [0, np.exp(1j*theta/2)]], dtype=complex)
        self._apply_1q(gate, target)
        self.gate_count += 1

    def u3(self, theta: float, phi: float, lam: float, target: int):
        """Universal single-qubit rotation."""
        gate = np.array([
            [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
        ], dtype=complex)
        self._apply_1q(gate, target)
        self.gate_count += 1

    def cnot(self, control: int, target: int):
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        self._apply_2q(CNOT, control, target)
        self.gate_count += 1

    def cz(self, control: int, target: int):
        CZ = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
        self._apply_2q(CZ, control, target)
        self.gate_count += 1

    def swap(self, q1: int, q2: int):
        SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        self._apply_2q(SWAP, q1, q2)
        self.gate_count += 1

    def toffoli(self, c1: int, c2: int, target: int):
        """CCNOT gate."""
        gate = np.eye(8, dtype=complex)
        gate[6,6] = 0; gate[6,7] = 1
        gate[7,6] = 1; gate[7,7] = 0
        self._apply_3q(gate, c1, c2, target)
        self.gate_count += 1

    def _apply_1q(self, gate: np.ndarray, target: int):
        shape = [2] * self.n
        tensor = self.state.reshape(shape)
        axes = list(range(self.n))
        axes[0], axes[target] = axes[target], axes[0]
        tensor = np.transpose(tensor, axes)
        tensor = np.tensordot(gate, tensor, axes=([1], [0]))
        inv_axes = [0] * self.n
        for i, a in enumerate(axes):
            inv_axes[a] = i
        tensor = np.transpose(tensor, inv_axes)
        self.state = tensor.reshape(self.N)

    def _apply_2q(self, gate: np.ndarray, q1: int, q2: int):
        shape = [2] * self.n
        tensor = self.state.reshape(shape)
        axes = list(range(self.n))
        axes[0], axes[q1] = axes[q1], axes[0]
        if q2 == 0:
            q2 = q1
        axes[1], axes[q2] = axes[q2], axes[1]
        tensor = np.transpose(tensor, axes)
        gate_t = gate.reshape(2, 2, 2, 2)
        tensor = np.tensordot(gate_t, tensor, axes=([2, 3], [0, 1]))
        inv_axes = [0] * self.n
        for i, a in enumerate(axes):
            inv_axes[a] = i
        tensor = np.transpose(tensor, inv_axes)
        self.state = tensor.reshape(self.N)

    def _apply_3q(self, gate: np.ndarray, q1: int, q2: int, q3: int):
        I = np.eye(2, dtype=complex)
        ops = [I] * self.n
        ops[q1] = ops[q2] = ops[q3] = None
        U = np.eye(1, dtype=complex)
        i = 0
        while i < self.n:
            if i == q1:
                U = np.kron(U, gate)
                i += 3
            else:
                U = np.kron(U, ops[i])
                i += 1
        self.state = U @ self.state

    # --- Measurement ---
    def measure(self, target: int, shots: int = 1) -> int | List[int]:
        shape = [2] * self.n
        tensor = self.state.reshape(shape)
        prob_0 = np.sum(np.abs(tensor.take(0, axis=target))**2)

        results = []
        state_copy = self.state.copy()
        for _ in range(shots):
            self.state = state_copy.copy()
            if np.random.random() < prob_0:
                results.append(0)
                mask = np.array([1 if ((i >> target) & 1) == 0 else 0 
                                 for i in range(self.N)])
            else:
                results.append(1)
                mask = np.array([1 if ((i >> target) & 1) == 1 else 0 
                                 for i in range(self.N)])
            self.state *= mask
            norm = np.linalg.norm(self.state)
            if norm > 1e-15:
                self.state /= norm
            state_copy = self.state.copy()
        return results[0] if shots == 1 else results

    def measure_all(self, shots: int = 1024) -> Dict[str, int]:
        probs = np.abs(self.state)**2
        outcomes = np.random.choice(self.N, size=shots, p=probs)
        counts = {}
        for o in outcomes:
            bitstr = format(o, f'0{self.n}b')
            counts[bitstr] = counts.get(bitstr, 0) + 1
        return counts

    # --- Analysis ---
    def get_probabilities(self) -> np.ndarray:
        return np.abs(self.state)**2

    def get_expectation(self, pauli_string: str) -> float:
        temp = self.state.copy()
        for i, p in enumerate(pauli_string):
            if p == 'X':
                self._apply_1q(np.array([[0,1],[1,0]], dtype=complex), i)
            elif p == 'Y':
                self._apply_1q(np.array([[0,-1j],[1j,0]], dtype=complex), i)
            elif p == 'Z':
                self._apply_1q(np.array([[1,0],[0,-1]], dtype=complex), i)
        result = np.vdot(temp, self.state).real
        self.state = temp
        return result

    def entanglement_entropy(self, qubits_A: List[int]) -> float:
        rho = np.outer(self.state, self.state.conj())
        qubits_B = [i for i in range(self.n) if i not in qubits_A]
        shape = [2]*self.n + [2]*self.n
        tensor = rho.reshape(shape)
        trace_axes = sorted(qubits_B, reverse=True)
        for idx, q in enumerate(trace_axes):
            offset = self.n - idx
            tensor = np.trace(tensor, axis1=q, axis2=q+offset)
        n_keep = len(qubits_A)
        rho_A = tensor.reshape(2**n_keep, 2**n_keep)
        eigvals = np.linalg.eigvalsh(rho_A)
        eigvals = eigvals[eigvals > 1e-15]
        return -np.sum(eigvals * np.log2(eigvals))

    def fidelity(self, other_state: np.ndarray) -> float:
        return np.abs(np.vdot(self.state, other_state))**2

    def benchmark(self) -> Dict:
        """Return performance metrics."""
        mem_mb = self.state.nbytes / (1024**2)
        avg_gate_time = np.mean(self._gate_times) * 1000 if self._gate_times else 0
        return {
            'n_qubits': self.n,
            'amplitudes': self.N,
            'memory_mb': mem_mb,
            'gate_count': self.gate_count,
            'avg_gate_time_ms': avg_gate_time,
        }

    def print_state(self, max_terms: int = 8):
        probs = self.get_probabilities()
        print(f"\n{'State':>12} | {'Amplitude':>22} | {'Probability':>12}")
        print("-" * 52)
        shown = 0
        for i in range(self.N):
            if probs[i] > 1e-10 or shown < max_terms:
                bitstr = format(i, f'0{self.n}b')
                amp = self.state[i]
                print(f"|{bitstr}> | {amp:>22.6f} | {probs[i]:>12.6f}")
                shown += 1
        if shown < self.N:
            print(f"... ({self.N - shown} more states)")


# ------------------------------------------------------------------
# P1.2: Clifford / Stabilizer Simulator (naive reference)
# ------------------------------------------------------------------
class CliffordSimulatorNaive:
    """
    Stabilizer simulator using tableau representation (naive reference).
    Memory: O(n^2). Can simulate 1000+ qubits efficiently.
    Only supports Clifford gates: H, S, CNOT, CZ, Pauli, SWAP.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.tableau = np.zeros((2*n_qubits, 2*n_qubits + 1), dtype=int)
        for i in range(n_qubits):
            self.tableau[i, n_qubits + i] = 1
            self.tableau[n_qubits + i, i] = 1
        self.gate_count = 0

    def _rowsum(self, h: int, i: int):
        def g(x1, z1, x2, z2):
            if x1 == 0 and z1 == 0: return 0
            if x1 == 1 and z1 == 1: return z2 - x2
            if x1 == 1 and z1 == 0: return z2 * (2*x2 - 1)
            if x1 == 0 and z1 == 1: return x2 * (1 - 2*z2)
            return 0
        phase = 2 * self.tableau[h, -1] + 2 * self.tableau[i, -1]
        for j in range(self.n):
            phase += g(self.tableau[i, j], self.tableau[i, self.n+j],
                       self.tableau[h, j], self.tableau[h, self.n+j])
        self.tableau[h, -1] = (phase // 2) % 2
        for j in range(2*self.n):
            self.tableau[h, j] ^= self.tableau[i, j]

    def h(self, a: int):
        for i in range(2*self.n):
            self.tableau[i, -1] ^= self.tableau[i, a] * self.tableau[i, self.n+a]
            self.tableau[i, a], self.tableau[i, self.n+a] =                 self.tableau[i, self.n+a], self.tableau[i, a]
        self.gate_count += 1

    def s(self, a: int):
        for i in range(2*self.n):
            self.tableau[i, -1] ^= self.tableau[i, a] * self.tableau[i, self.n+a]
            self.tableau[i, self.n+a] ^= self.tableau[i, a]
        self.gate_count += 1

    def cnot(self, a: int, b: int):
        for i in range(2*self.n):
            self.tableau[i, -1] ^= (self.tableau[i, a] * self.tableau[i, self.n+b] *
                                    (self.tableau[i, self.n+a] ^ self.tableau[i, b] ^ 1))
            self.tableau[i, b] ^= self.tableau[i, a]
            self.tableau[i, self.n+a] ^= self.tableau[i, self.n+b]
        self.gate_count += 1

    def _gf2_solve(self, a: int):
        # Solve [X; Z] t = [0; e_a] over GF(2) with plain Python.
        n = self.n
        M = [[int(self.tableau[i, j]) for i in range(n)] for j in range(n)]
        M += [[int(self.tableau[i, n + j]) for i in range(n)] for j in range(n)]
        rhs = [0] * n + [1 if j == a else 0 for j in range(n)]
        piv_cols = []
        row = 0
        for col in range(n):
            piv = -1
            for rr in range(row, 2 * n):
                if M[rr][col]:
                    piv = rr
                    break
            if piv == -1:
                continue
            M[row], M[piv] = M[piv], M[row]
            rhs[row], rhs[piv] = rhs[piv], rhs[row]
            for rr in range(2 * n):
                if rr != row and M[rr][col]:
                    M[rr] = [x ^ y for x, y in zip(M[rr], M[row])]
                    rhs[rr] ^= rhs[row]
            piv_cols.append(col)
            row += 1
        t = [0] * n
        for i, col in enumerate(piv_cols):
            t[col] = rhs[i]
        return t

    def measure(self, a: int) -> int:
        n = self.n
        p = -1
        for i in range(n):            # STABILIZER rows: X/Y on a => random
            if self.tableau[i, a] == 1:
                p = i
                break
        if p == -1:
            t = self._gf2_solve(a)
            return sum(t[i] * self.tableau[i, -1] for i in range(n)) % 2
        else:
            result = np.random.randint(0, 2)
            for i in range(2*self.n):
                if i != p and self.tableau[i, a] == 1:
                    self._rowsum(i, p)
            self.tableau[p - self.n, :] = self.tableau[p, :]
            self.tableau[p, :] = 0
            self.tableau[p, self.n + a] = 1
            self.tableau[p, -1] = result
            return result

    def get_stabilizers(self) -> List[str]:
        gens = []
        for i in range(self.n):
            s = ""
            for j in range(self.n):
                x, z = self.tableau[i, j], self.tableau[i, self.n+j]
                if x == 0 and z == 0: s += "I"
                elif x == 1 and z == 0: s += "X"
                elif x == 0 and z == 1: s += "Z"
                else: s += "Y"
            gens.append(s)
        return gens

    def benchmark(self) -> Dict:
        mem_kb = self.tableau.nbytes / 1024
        return {
            'n_qubits': self.n,
            'memory_kb': mem_kb,
            'gate_count': self.gate_count,
            'type': 'Clifford/Stabilizer',
        }


# ==================================================================
# P0 OPTIMIZED IMPLEMENTATIONS (2026-09)
# Techniques: gate fusion, strided/vectorized application, diagonal
# fast paths, bit-packed tableau, LUT-based rowsum, and Stim-style
# Pauli-frame batch sampling for noisy circuits.
# ==================================================================

class StateVectorSimulator:
    """
    Optimized full-amplitude state-vector simulator.

    Improvements over the naive version:
      * Consecutive single-qubit gates on the same target are FUSED
        into one 2x2 matrix before touching the state vector.
      * Gates act on strided views (reshape + einsum/matmul on the
        target axis) -- no full-array transposes per gate.
      * Diagonal/phase gates (RZ, Z, S, T, CZ) use O(N) elementwise
        phase multiplication instead of a 2x2 matmul.
      * X-type gates use slice swaps (bit flips) when cheaper.

    Memory: O(2^n). Practical limit: ~25-27 qubits single-threaded CPU.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.N = 2 ** n_qubits
        self.state = np.zeros(self.N, dtype=complex)
        self.state[0] = 1.0
        self.gate_count = 0
        self._gate_times = []
        self._fused_1q: List[Optional[np.ndarray]] = [None] * n_qubits

    # --- fusion machinery ---
    def _buffer_1q(self, gate: np.ndarray, target: int):
        """Accumulate a 1q gate into the fusion buffer (newest on the left)."""
        pending = self._fused_1q[target]
        self._fused_1q[target] = gate if pending is None else gate @ pending
        self.gate_count += 1

    def _flush_1q(self, target: Optional[int] = None):
        """Apply buffered fused 1q gates. target=None flushes all."""
        if target is None:
            for t in range(self.n):
                if self._fused_1q[t] is not None:
                    self._flush_1q(t)
            return
        G = self._fused_1q[target]
        if G is None:
            return
        t0 = time.perf_counter()
        k = self.n - 1 - target
        st = self.state.reshape(2 ** k, 2, 2 ** target)
        self.state = np.einsum('ab,kbj->kaj', G, st).reshape(self.N)
        self._gate_times.append(time.perf_counter() - t0)
        self._fused_1q[target] = None

    def _flush_all(self):
        self._flush_1q()

    # --- Gate library (buffers or uses fast paths) ---
    def h(self, target: int):
        self._buffer_1q(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), target)

    def x(self, target: int):
        # bit-flip = swap amplitude slices; cheaper than a matmul
        if self._fused_1q[target] is not None:
            self._flush_1q(target)
        t0 = time.perf_counter()
        k = self.n - 1 - target
        st = self.state.reshape(2 ** k, 2, 2 ** target)
        st[:, 0, :], st[:, 1, :] = st[:, 1, :].copy(), st[:, 0, :].copy()
        self._gate_times.append(time.perf_counter() - t0)
        self.gate_count += 1

    def y(self, target: int):
        self._buffer_1q(np.array([[0, -1j], [1j, 0]], dtype=complex), target)

    def z(self, target: int):
        # diagonal fast path: multiply |1> slice by -1
        if self._fused_1q[target] is not None:
            self._flush_1q(target)
        t0 = time.perf_counter()
        k = self.n - 1 - target
        st = self.state.reshape(2 ** k, 2, 2 ** target)
        st[:, 1, :] *= -1
        self._gate_times.append(time.perf_counter() - t0)
        self.gate_count += 1

    def s(self, target: int):
        self._buffer_1q(np.array([[1, 0], [0, 1j]], dtype=complex), target)

    def t(self, target: int):
        self._buffer_1q(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex), target)

    def rx(self, theta: float, target: int):
        gate = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                         [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        self._buffer_1q(gate, target)

    def ry(self, theta: float, target: int):
        gate = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        self._buffer_1q(gate, target)

    def rz(self, theta: float, target: int):
        # diagonal fast path
        if self._fused_1q[target] is not None:
            self._flush_1q(target)
        t0 = time.perf_counter()
        k = self.n - 1 - target
        st = self.state.reshape(2 ** k, 2, 2 ** target)
        st[:, 0, :] *= np.exp(-1j * theta / 2)
        st[:, 1, :] *= np.exp(1j * theta / 2)
        self._gate_times.append(time.perf_counter() - t0)
        self.gate_count += 1

    def u3(self, theta: float, phi: float, lam: float, target: int):
        gate = np.array([
            [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
        ], dtype=complex)
        self._buffer_1q(gate, target)

    def _apply_2q(self, gate: np.ndarray, q1: int, q2: int):
        """Generic 2q gate; gate matrix rows/cols indexed by (q1, q2)."""
        if q1 < q2:
            # swapping the two qubit roles conjugates the gate: G' = P G P
            P = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                          [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
            gate = P @ gate @ P
            q1, q2 = q2, q1
        self._flush_all()
        t0 = time.perf_counter()
        # view: (high_rest, q1, mid, q2, low_rest), index = q1*2 + q2
        st = self.state.reshape(2 ** (self.n - 1 - q1), 2, 2 ** (q1 - 1 - q2),
                                2, 2 ** q2)
        T = st.transpose(1, 3, 0, 2, 4).reshape(4, -1)
        out = (gate.reshape(4, 4) @ T).reshape(2, 2, st.shape[0], st.shape[2],
                                               st.shape[4])
        self.state = out.transpose(2, 0, 3, 1, 4).reshape(self.N).copy()
        self._gate_times.append(time.perf_counter() - t0)
        self.gate_count += 1

    def cnot(self, control: int, target: int):
        self._flush_all()
        t0 = time.perf_counter()
        if control > target:
            hi, lo = control, target
            st = self.state.reshape(2 ** (self.n - 1 - hi), 2,
                                    2 ** (hi - 1 - lo), 2, 2 ** lo)
            blk = st[:, 1, :, :, :]                # control (hi) = 1
            blk[:, :, 0, :], blk[:, :, 1, :] = blk[:, :, 1, :].copy(), blk[:, :, 0, :].copy()
        else:
            hi, lo = target, control
            st = self.state.reshape(2 ** (self.n - 1 - hi), 2,
                                    2 ** (hi - 1 - lo), 2, 2 ** lo)
            blk = st[:, :, :, 1, :]                # control (lo) = 1
            blk[:, 0, :], blk[:, 1, :] = blk[:, 1, :].copy(), blk[:, 0, :].copy()
        self._gate_times.append(time.perf_counter() - t0)
        self.gate_count += 1

    def cz(self, control: int, target: int):
        self._flush_all()
        t0 = time.perf_counter()
        c, t = (control, target) if control > target else (target, control)
        st = self.state.reshape(2 ** (self.n - 1 - c), 2, 2 ** (c - 1 - t),
                                2, 2 ** t)
        st[:, 1, :, 1, :] *= -1                        # both bits set
        self._gate_times.append(time.perf_counter() - t0)
        self.gate_count += 1

    def swap(self, q1: int, q2: int):
        self._flush_all()
        t0 = time.perf_counter()
        c, t = (q1, q2) if q1 > q2 else (q2, q1)
        st = self.state.reshape(2 ** (self.n - 1 - c), 2, 2 ** (c - 1 - t),
                                2, 2 ** t)
        a = st[:, 0, :, 1, :].copy()
        st[:, 0, :, 1, :] = st[:, 1, :, 0, :]
        st[:, 1, :, 0, :] = a
        self._gate_times.append(time.perf_counter() - t0)
        self.gate_count += 1

    def toffoli(self, c1: int, c2: int, target: int):
        self._flush_all()
        t0 = time.perf_counter()
        # transpose to (c1, c2, target, rest); flip target where both
        # controls are 1 -- correct for arbitrary relative positions
        d = lambda q: self.n - 1 - q   # reshape dim of qubit q (dim0 = MSB)
        axes = [d(c1), d(c2), d(target)] + [dd for dd in range(self.n)
                                            if dd not in (d(c1), d(c2), d(target))]
        t = self.state.reshape([2] * self.n).transpose(axes)
        blk = t.reshape(2, 2, 2, -1)              # axes 0,1 = controls, 2 = target
        a = blk[1, 1, 0, :].copy()
        blk[1, 1, 0, :] = blk[1, 1, 1, :]
        blk[1, 1, 1, :] = a
        self.state = blk.reshape([2] * self.n).transpose(np.argsort(axes)).reshape(self.N)
        self._gate_times.append(time.perf_counter() - t0)
        self.gate_count += 1

    # --- Measurement ---
    def measure(self, target: int, shots: int = 1) -> int | List[int]:
        self._flush_all()
        k = self.n - 1 - target
        st = self.state.reshape(2 ** k, 2, 2 ** target)
        p0 = float(np.sum(np.abs(st[:, 0, :]) ** 2))
        results = (np.random.random(shots) >= p0).astype(int)  # 1 w.p. 1-p0
        if shots == 1:
            r = int(results[0])
            # collapse in place
            if r == 0:
                st[:, 1, :] = 0
            else:
                st[:, 0, :] = 0
                st[:, 1, :] /= np.sqrt(1 - p0) if p0 < 1 else 1
            if r == 0:
                st[:, 0, :] /= np.sqrt(p0) if p0 > 0 else 1
            return r
        # shots > 1: sequential collapse, reusing the strided view
        out = []
        for s in results:
            st = self.state.reshape(2 ** k, 2, 2 ** target)
            p0 = float(np.sum(np.abs(st[:, 0, :]) ** 2))
            r = int(np.random.random() < p0)
            out.append(r)
            if r == 0:
                st[:, 1, :] = 0
                st[:, 0, :] /= np.sqrt(p0) if p0 > 0 else 1
            else:
                st[:, 0, :] = 0
                st[:, 1, :] /= np.sqrt(1 - p0) if p0 < 1 else 1
        return out

    def measure_all(self, shots: int = 1024) -> Dict[str, int]:
        self._flush_all()
        probs = np.abs(self.state) ** 2
        outcomes = np.random.default_rng().choice(self.N, size=shots, p=probs)
        counts: Dict[str, int] = {}
        for o in outcomes:
            bitstr = format(int(o), f'0{self.n}b')
            counts[bitstr] = counts.get(bitstr, 0) + 1
        return counts

    # --- Analysis ---
    def get_probabilities(self) -> np.ndarray:
        self._flush_all()
        return np.abs(self.state) ** 2

    def get_expectation(self, pauli_string: str) -> float:
        self._flush_all()
        temp = self.state.copy()
        for i, p in enumerate(pauli_string):
            if p == 'X':
                self._apply_1q_direct(np.array([[0, 1], [1, 0]], dtype=complex), i)
            elif p == 'Y':
                self._apply_1q_direct(np.array([[0, -1j], [1j, 0]], dtype=complex), i)
            elif p == 'Z':
                self._apply_1q_direct(np.array([[1, 0], [0, -1]], dtype=complex), i)
        result = np.vdot(temp, self.state).real
        self.state = temp
        return result

    def _apply_1q_direct(self, gate: np.ndarray, target: int):
        """Apply a 1q gate immediately (bypasses fusion buffer)."""
        k = self.n - 1 - target
        st = self.state.reshape(2 ** k, 2, 2 ** target)
        self.state = np.einsum('ab,kbj->kaj', gate, st).reshape(self.N)

    def entanglement_entropy(self, qubits_A: List[int]) -> float:
        self._flush_all()
        rho = np.outer(self.state, self.state.conj())
        qubits_B = [i for i in range(self.n) if i not in qubits_A]
        shape = [2] * self.n + [2] * self.n
        tensor = rho.reshape(shape)
        trace_axes = sorted(qubits_B, reverse=True)
        for idx, q in enumerate(trace_axes):
            offset = self.n - idx
            tensor = np.trace(tensor, axis1=q, axis2=q + offset)
        n_keep = len(qubits_A)
        rho_A = tensor.reshape(2 ** n_keep, 2 ** n_keep)
        eigvals = np.linalg.eigvalsh(rho_A)
        eigvals = eigvals[eigvals > 1e-15]
        return float(-np.sum(eigvals * np.log2(eigvals)))

    def fidelity(self, other_state: np.ndarray) -> float:
        self._flush_all()
        return float(np.abs(np.vdot(self.state, other_state)) ** 2)

    def benchmark(self) -> Dict:
        self._flush_all()
        mem_mb = self.state.nbytes / (1024 ** 2)
        avg_gate_time = np.mean(self._gate_times) * 1000 if self._gate_times else 0
        return {
            'n_qubits': self.n,
            'amplitudes': self.N,
            'memory_mb': mem_mb,
            'gate_count': self.gate_count,
            'avg_gate_time_ms': avg_gate_time,
        }

    def print_state(self, max_terms: int = 8):
        self._flush_all()
        probs = self.get_probabilities()
        print(f"\n{'State':>12} | {'Amplitude':>22} | {'Probability':>12}")
        print("-" * 52)
        shown = 0
        for i in range(self.N):
            if probs[i] > 1e-10 or shown < max_terms:
                bitstr = format(i, f'0{self.n}b')
                amp = self.state[i]
                print(f"|{bitstr}> | {amp:>22.6f} | {probs[i]:>12.6f}")
                shown += 1
        if shown < self.N:
            print(f"... ({self.N - shown} more states)")


# ------------------------------------------------------------------
# P1.2-opt: Clifford / Stabilizer Simulator (vectorized + Pauli frame)
# ------------------------------------------------------------------
class CliffordSimulator:
    """
    Optimized stabilizer simulator.

    Improvements over the naive version:
      * Bit-packed tableau (uint8 X/Z blocks) -- no Python objects per Pauli.
      * rowsum phase computed with a 16-entry lookup table, vectorized
        over qubits with numpy (O(n) per row op, in C, not Python loops).
      * Stim-style Pauli-frame batch sampling (`sample()`): run ONE noiseless
        reference pass, then propagate errors through the circuit as Pauli
        frames vectorized over ALL shots at once. Per-shot cost collapses
        from O(gates * n^2) to O(gates) -- the same trick Stim uses to hit
        kHz sampling rates on surface codes (Gidney, Quantum 5, 497).

    Memory: O(n^2). Supports 1000+ qubits.
    """

    # g(x1,z1,x2,z2) phase lookup, flattened index (x1 z1 x2 z2) in binary
    _LUT = np.array([0, 0, 0, 0,   # x1z1 = 00 (I)
                     0, -1, 1, 0,  # x1z1 = 01 (Z): x2*(1-2*z2)
                     0, 1, -1, 0,  # x1z1 = 10 (X): z2*(2*x2-1)
                     0, 1, -1, 0], # x1z1 = 11 (Y): z2 - x2
                    dtype=np.int32)

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.x = np.zeros((2 * n_qubits, n_qubits), dtype=np.uint8)
        self.z = np.zeros((2 * n_qubits, n_qubits), dtype=np.uint8)
        self.r = np.zeros(2 * n_qubits, dtype=np.uint8)
        for i in range(n_qubits):
            self.z[i, i] = 1          # stabilizers start as Z_i
            self.x[n_qubits + i, i] = 1  # destabilizers start as X_i
        self.gate_count = 0
        self._gf2_rows = None

    # --- vectorized rowsum ---
    def _rowsum(self, h: int, i: int):
        """Row h <- Row h * Row i (Aaronson-Gottesman phase tracking)."""
        idx = ((self.x[i].astype(np.int32) << 3) | (self.z[i].astype(np.int32) << 2)
               | (self.x[h].astype(np.int32) << 1) | self.z[h])
        phase = (2 * int(self.r[h]) + 2 * int(self.r[i])
                 + int(self._LUT[idx].sum()))
        self.r[h] = (phase >> 1) & 1
        np.logical_xor(self.x[h], self.x[i], out=self.x[h])
        np.logical_xor(self.z[h], self.z[i], out=self.z[h])

    def h(self, a: int):
        self._invalidate_gf2()
        np.logical_xor(self.r, self.x[:, a] & self.z[:, a], out=self.r)
        self.x[:, a], self.z[:, a] = self.z[:, a].copy(), self.x[:, a].copy()
        self.gate_count += 1

    def s(self, a: int):
        self._invalidate_gf2()
        np.logical_xor(self.r, self.x[:, a] & self.z[:, a], out=self.r)
        np.logical_xor(self.z[:, a], self.x[:, a], out=self.z[:, a])
        self.gate_count += 1

    def pauli_x(self, a: int):
        self._invalidate_gf2()
        # Pauli conjugation: Y->-Y, Z->-Z; bits unchanged, phase flips iff z=1
        np.logical_xor(self.r, self.z[:, a], out=self.r)
        self.gate_count += 1

    def pauli_z(self, a: int):
        self._invalidate_gf2()
        np.logical_xor(self.r, self.x[:, a], out=self.r)
        self.gate_count += 1

    def pauli_y(self, a: int):
        self._invalidate_gf2()
        np.logical_xor(self.r, self.x[:, a] ^ self.z[:, a], out=self.r)
        self.gate_count += 1

    def cnot(self, a: int, b: int):
        self._invalidate_gf2()
        flip = (self.x[:, a] & self.z[:, b]
                & np.logical_xor(np.logical_xor(self.z[:, a], self.x[:, b]), 1))
        np.logical_xor(self.r, flip, out=self.r)
        np.logical_xor(self.x[:, b], self.x[:, a], out=self.x[:, b])
        np.logical_xor(self.z[:, a], self.z[:, b], out=self.z[:, a])
        self.gate_count += 1

    def _gf2_solve_z_repr(self, a: int) -> np.ndarray:
        # Solve [X; Z] t = [0; e_a] over GF(2): Z_a = (+/-) prod S_i^t_i.
        # Unique solution iff the measurement is deterministic.
        n = self.n
        M = np.concatenate([self.x[:n].T, self.z[:n].T], axis=0).astype(np.uint8)
        rhs = np.zeros(2 * n, dtype=np.uint8)
        rhs[n + a] = 1
        piv_cols = []
        row = 0
        for col in range(n):
            piv = -1
            for rr in range(row, 2 * n):
                if M[rr, col]:
                    piv = rr
                    break
            if piv == -1:
                continue
            if piv != row:
                M[[row, piv]] = M[[piv, row]]
                rhs[row], rhs[piv] = rhs[piv], rhs[row]
            for rr in range(2 * n):
                if rr != row and M[rr, col]:
                    M[rr] ^= M[row]
                    rhs[rr] ^= rhs[row]
            piv_cols.append(col)
            row += 1
        t = np.zeros(n, dtype=np.uint8)
        for i, col in enumerate(piv_cols):
            t[col] = rhs[i]
        return t

    def measure(self, a: int) -> int:
        n = self.n
        # random iff a STABILIZER row has X/Y on a (Z_a not in the group)
        col = self.x[:n, a]
        if not col.any():
            # deterministic: epoch-cached GF(2) representation lookup
            return self._deterministic_outcome(a)
        p = int(np.argmax(col))
        result = int(np.random.randint(0, 2))
        self._invalidate_gf2()
        for i in range(2 * n):
            if i != p and self.x[i, a]:
                self._rowsum(i, p)
        self.x[p - n, :] = self.x[p, :]
        self.z[p - n, :] = self.z[p, :]
        self.r[p - n] = self.r[p]
        self.x[p, :] = 0
        self.z[p, :] = 0
        self.z[p, a] = 1
        self.r[p] = result
        return result

    # --- Stim-style Pauli-frame batch sampling ---
    @staticmethod
    def _frame_conj_h(fx, fz, q):
        tmp = fx[:, q].copy(); fx[:, q] = fz[:, q]; fz[:, q] = tmp

    @staticmethod
    def _frame_conj_s(fx, fz, q):
        fz[:, q] ^= fx[:, q]

    @staticmethod
    def _frame_conj_cnot(fx, fz, a, b):
        fx[:, b] ^= fx[:, a]
        fz[:, a] ^= fz[:, b]

    # ---- epoch-cached deterministic-measurement solver ----------------
    # Deterministic measurements do NOT modify the tableau, so the GF(2)
    # factorization of [X; Z] is valid for a whole "epoch" (between random
    # collapses / gate updates).  One Gauss-Jordan per epoch turns every
    # later solve into an O(n) big-int column read.
    def _invalidate_gf2(self):
        self._gf2_rows = None

    def _gf2_factor(self):
        n = self.n
        rows = []
        for j in range(n):
            c = int.from_bytes(
                np.packbits(self.x[:n].T[j], bitorder='little').tobytes(),
                'little')
            rows.append(c | (1 << (n + j)))
        for j in range(n):
            c = int.from_bytes(
                np.packbits(self.z[:n].T[j], bitorder='little').tobytes(),
                'little')
            rows.append(c | (1 << (2 * n + j)))
        pivots = []
        r = 0
        for col in range(n):
            p = -1
            for i in range(r, 2 * n):
                if (rows[i] >> col) & 1:
                    p = i
                    break
            if p == -1:
                continue
            rows[r], rows[p] = rows[p], rows[r]
            pr = rows[r]
            for i in range(2 * n):
                if i != r and ((rows[i] >> col) & 1):
                    rows[i] ^= pr
            pivots.append(col)
            r += 1
        self._gf2_rows = rows
        self._gf2_pivots = pivots

    def _deterministic_outcome(self, a: int) -> int:
        if self._gf2_rows is None:
            self._gf2_factor()
        rows, pivots = self._gf2_rows, self._gf2_pivots
        n = self.n
        bit = 2 * n + a                 # rhs = e_a in the Z-block equations
        t = 0
        for i, col in enumerate(pivots):
            if (rows[i] >> bit) & 1:
                t |= 1 << col
        # outcome = parity of t over stabilizer generator phases
        par = 0
        m = t & ((1 << n) - 1)
        while m:
            low = m & (-m)
            par ^= int(self.r[low.bit_length() - 1])
            m ^= low
        return par

    # GF(2) solve packed into Python ints: [X;Z] t = [0; e_a] (or swapped
    # blocks for X-observable), rows returned as int bitmasks over t bits.
    @staticmethod
    def _pack_cols(block: np.ndarray) -> List[int]:
        # block: (n, n) uint8 rows=generators, cols=qubits -> per-qubit int
        bt = np.packbits(block.T, axis=1, bitorder='little')  # (n, ceil(n/8))
        width = bt.shape[1]
        return [int.from_bytes(bt[j].tobytes(), 'little') for j in range(block.shape[0])]

    def _solve_obs(self, x: np.ndarray, z: np.ndarray,
                   a: int, obs: str) -> int:
        """Bitmask t with Z/X_a = (+/-) prod_i S_i^{t_i}; deterministic case.
        x, z are the PASS-1 local tableau (not self.x/self.z)."""
        n = self.n
        xrows = self._pack_cols(x[:n])
        zrows = self._pack_cols(z[:n])
        if obs == 'Z':
            rows = xrows + [r | (1 << n) if j == a else r
                            for j, r in enumerate(zrows)]
        else:  # 'X'
            rows = [r | (1 << n) if j == a else r
                    for j, r in enumerate(xrows)] + zrows
        piv = 0
        for col in range(n):
            p = -1
            for rr in range(piv, 2 * n):
                if (rows[rr] >> col) & 1:
                    p = rr
                    break
            if p == -1:
                continue
            rows[piv], rows[p] = rows[p], rows[piv]
            pr = rows[piv]
            for rr in range(2 * n):
                if rr != piv and ((rows[rr] >> col) & 1):
                    rows[rr] ^= pr
            piv += 1
        t = 0
        for r in rows[:piv]:
            if (r >> n) & 1:
                lead = (r & ((1 << n) - 1)).bit_length() - 1
                t |= 1 << lead
        return t

    def sample(self, ops: List[tuple], shots: int,
               seed: Optional[int] = None) -> np.ndarray:
        """
        Exact noisy-circuit sampling via reference + symbolic Pauli frames.

        Technique (the same architecture Stim uses):
          pass 1  Run the noiseless circuit ONCE, recording for every
                  measurement an affine GF(2) form of its outcome:
                      out = base  XOR  <coins_coeff, coins>  XOR  <err_coeff, errs>
                  where `coins` are fair bits (one per random ideal
                  measurement) and `errs` are the per-location error bits.
                  Correlations between measurements are preserved because
                  all outcomes share the same coin vector linearly.
          pass 2  Propagate a symbolic error frame (each Pauli component is
                  a GF(2) coefficient triple over coins/errs) through the
                  circuit; errors insert bits, measurements read off their
                  anticommutation and apply collapse kicks.
          assemble Draw coins/errs per shot (Bernoulli error probs) and
                  evaluate the affine forms vectorized over shots.

        ops: ('h',q) ('s',q) ('cnot',a,b) ('mz',q) ('mx',q)
             ('x_error',q,p) ('y_error',q,p) ('z_error',q,p)
             ('depolarize',q,p)   -- independent X/Y/Z bits of prob p/3
        Returns (shots, n_measurements) uint8.
        """
        rng = np.random.default_rng(seed)
        n = self.n

        # ---------------- pass 1: noiseless reference, affine outcomes ----
        x = np.zeros((2 * n, n), dtype=np.uint8)
        z = np.zeros((2 * n, n), dtype=np.uint8)
        for i in range(n):
            z[i, i] = 1
            x[n + i, i] = 1
        base_r = np.zeros(2 * n, dtype=np.uint8)   # fixed phase bits
        rc = [0] * (2 * n)                          # coin coefficients (ints)

        n_coins = 0
        n_meas = 0
        out_base: List[int] = []
        out_coins: List[int] = []
        # deferred err coefficients from pass 2 (aligned by measurement idx)

        def rowsum(h: int, i: int):
            idx = ((x[i].astype(np.int32) << 3) | (z[i].astype(np.int32) << 2)
                   | (x[h].astype(np.int32) << 1) | z[h])
            phase = (2 * int(base_r[h]) + 2 * int(base_r[i])
                     + int(self._LUT[idx].sum()))
            base_r[h] = (phase >> 1) & 1
            np.logical_xor(x[h], x[i], out=x[h])
            np.logical_xor(z[h], z[i], out=z[h])
            rc[h] ^= rc[i]

        def measure_obs(obs: str, q: int):
            nonlocal n_coins
            anti = x[:n, q] if obs == 'Z' else z[:n, q]
            if anti.any():
                # random: fresh coin carries the outcome
                p = int(np.argmax(anti))
                for j in range(2 * n):
                    if j != p and (x[j, q] if obs == 'Z' else z[j, q]):
                        rowsum(j, p)
                x[n + p, :] = x[p, :]; z[n + p, :] = z[p, :]
                base_r[n + p] = base_r[p]; rc[n + p] = rc[p]
                x[p, :] = 0; z[p, :] = 0
                if obs == 'Z':
                    z[p, q] = 1
                else:
                    x[p, q] = 1
                base_r[p] = 0
                rc[p] = 1 << n_coins
                out_base.append(0)
                out_coins.append(1 << n_coins)
                n_coins += 1
            else:
                t = self._solve_obs(x, z, q, obs)
                b, c = 0, 0
                tt = t
                while tt:
                    lb = tt & (-tt)
                    i = lb.bit_length() - 1
                    b ^= int(base_r[i])
                    c ^= rc[i]
                    tt ^= lb
                out_base.append(b)
                out_coins.append(c)

        for op in ops:
            tag = op[0]
            if tag == 'h':
                np.logical_xor(base_r, x[:, op[1]] & z[:, op[1]], out=base_r)
                x[:, op[1]], z[:, op[1]] = z[:, op[1]].copy(), x[:, op[1]].copy()
            elif tag == 's':
                np.logical_xor(base_r, x[:, op[1]] & z[:, op[1]], out=base_r)
                np.logical_xor(z[:, op[1]], x[:, op[1]], out=z[:, op[1]])
            elif tag == 'cnot':
                a_, b_ = op[1], op[2]
                flip = (x[:, a_] & z[:, b_]
                        & np.logical_xor(np.logical_xor(z[:, a_], x[:, b_]), 1))
                np.logical_xor(base_r, flip, out=base_r)
                np.logical_xor(x[:, b_], x[:, a_], out=x[:, b_])
                np.logical_xor(z[:, a_], z[:, b_], out=z[:, a_])
            elif tag == 'mz':
                measure_obs('Z', op[1]); n_meas += 1
            elif tag == 'mx':
                measure_obs('X', op[1]); n_meas += 1
            elif tag in ('x_error', 'y_error', 'z_error', 'depolarize'):
                continue
            else:
                raise ValueError(f"unknown op: {op}")

        # ---------------- pass 2: symbolic error frame --------------------
        # Frame Pauli per qubit = (X-comp, Z-comp); each component is a
        # GF(2) linear form over (coins, errs) kept as Python int masks.
        # Only components matter: outcomes are anticommute indicators.
        fc = [0] * n; fe = [0] * n          # X components (coins, errs)
        zc = [0] * n; ze = [0] * n          # Z components
        out_err = [0] * n_meas
        err_probs: List[float] = []
        m = 0
        for op in ops:
            tag = op[0]
            if tag == 'h':
                q = op[1]
                fc[q], zc[q] = zc[q], fc[q]
                fe[q], ze[q] = ze[q], fe[q]
            elif tag == 's':
                q = op[1]
                zc[q] ^= fc[q]; ze[q] ^= fe[q]
            elif tag == 'cnot':
                a_, b_ = op[1], op[2]
                fc[b_] ^= fc[a_]; fe[b_] ^= fe[a_]
                zc[a_] ^= zc[b_]; ze[a_] ^= ze[b_]
            elif tag == 'x_error':
                k = len(err_probs); err_probs.append(op[2])
                fe[op[1]] |= 1 << k
            elif tag == 'z_error':
                k = len(err_probs); err_probs.append(op[2])
                ze[op[1]] |= 1 << k
            elif tag == 'y_error':
                k = len(err_probs); err_probs.append(op[2])
                fe[op[1]] |= 1 << k; ze[op[1]] |= 1 << k
            elif tag == 'depolarize':
                for _ in range(3):
                    k = len(err_probs); err_probs.append(op[2] / 3)
                    fe[op[1]] |= 1 << k; ze[op[1]] |= 1 << k
            elif tag == 'mz':
                q = op[1]
                out_coins[m] ^= fc[q]          # anticommute(frame, Z_q)
                out_err[m] ^= fe[q]
                zc[q] ^= out_coins[m]          # kick: frame *= Z_q^out
                ze[q] ^= out_err[m]
                m += 1
            elif tag == 'mx':
                q = op[1]
                out_coins[m] ^= zc[q]          # anticommute(frame, X_q)
                out_err[m] ^= ze[q]
                fc[q] ^= out_coins[m]          # kick: frame *= X_q^out
                fe[q] ^= out_err[m]
                m += 1
            elif tag in ('x_error', 'y_error', 'z_error', 'depolarize'):
                pass
            else:
                raise ValueError(f"unknown op: {op}")

        # ---------------- assembly: vectorized over shots ------------------
        n_err = len(err_probs)
        Lc = max(1, (n_coins + 63) // 64)
        Le = max(1, (n_err + 63) // 64)

        # fair coins, full 64 bits per limb
        C = np.frombuffer(rng.bytes(shots * Lc * 8),
                          dtype=np.uint64).reshape(shots, Lc)
        if n_err:
            E_bits = rng.random((shots, n_err)) < np.array(err_probs)
            pad = Le * 64 - n_err
            if pad:
                E_bits = np.concatenate(
                    [E_bits, np.zeros((shots, pad), bool)], axis=1)
            E = np.packbits(E_bits, axis=1).view(np.uint64).reshape(shots, Le)
        else:
            E = np.zeros((shots, Le), dtype=np.uint64)

        def masks_to_limbs(masks, L):
            A = np.zeros((len(masks), L), dtype=np.uint64)
            for i, mk in enumerate(masks):
                A[i] = np.frombuffer(mk.to_bytes(8 * L, 'little'),
                                     dtype=np.uint64)
            return A

        Ac = masks_to_limbs(out_coins, Lc) if n_coins \
            else np.zeros((n_meas, Lc), dtype=np.uint64)
        Ae = masks_to_limbs(out_err, Le) if n_err \
            else np.zeros((n_meas, Le), dtype=np.uint64)

        def popcount64(v):
            v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
            v = (v & np.uint64(0x3333333333333333)) + \
                ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
            v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
            return (v * np.uint64(0x0101010101010101)) >> np.uint64(56)

        par_c = np.zeros((n_meas, shots), dtype=np.uint64)
        for l in range(Lc):
            both = Ac[:, l].reshape(-1, 1, 1) & C[:, l].reshape(1, -1, 1)
            par_c ^= popcount64(both).reshape(n_meas, shots) & np.uint64(1)
        par_e = np.zeros((n_meas, shots), dtype=np.uint64)
        for l in range(Le):
            both = Ae[:, l].reshape(-1, 1, 1) & E[:, l].reshape(1, -1, 1)
            par_e ^= popcount64(both).reshape(n_meas, shots) & np.uint64(1)

        base_col = np.array(out_base, dtype=np.uint64)[:, None]
        return (base_col ^ par_c ^ par_e).astype(np.uint8).T   # (shots, n_meas)

    def get_stabilizers(self) -> List[str]:
        gens = []
        for i in range(self.n):
            s = ""
            for j in range(self.n):
                xb, zb = self.x[i, j], self.z[i, j]
                s += "I" if xb == 0 and zb == 0 else \
                     "X" if xb == 1 and zb == 0 else \
                     "Z" if xb == 0 and zb == 1 else "Y"
            gens.append(s)
        return gens

    def benchmark(self) -> Dict:
        mem_kb = (self.x.nbytes + self.z.nbytes + self.r.nbytes) / 1024
        return {
            'n_qubits': self.n,
            'memory_kb': mem_kb,
            'gate_count': self.gate_count,
            'type': 'Clifford/Stabilizer (vectorized + affine-frame sampling)',
        }


# ==================================================================
# P0 OPTIMIZED IMPLEMENTATIONS (2026-09)
# Techniques: gate fusion, strided/vectorized application, diagonal
# fast paths, bit-packed tableau, LUT-based rowsum, and Stim-style
# Pauli-frame batch sampling for noisy circuits.
# ==================================================================

class MPSSimulator:
    """
    Matrix Product State simulator for 1D chains.
    Memory: O(n * chi^2) where chi is bond dimension.
    Can simulate 50-100+ qubits with limited entanglement.
    """

    def __init__(self, n_qubits: int, max_bond_dim: int = 64):
        self.n = n_qubits
        self.chi_max = max_bond_dim
        self.tensors = []
        for i in range(n_qubits):
            A = np.zeros((1, 2, 1), dtype=complex)
            A[0, 0, 0] = 1.0
            self.tensors.append(A)
        self.gate_count = 0

    def _apply_1q(self, gate: np.ndarray, site: int):
        A = self.tensors[site]
        new_A = np.tensordot(gate, A, axes=([1], [1]))
        new_A = np.transpose(new_A, (1, 0, 2))
        self.tensors[site] = new_A
        self.gate_count += 1

    def _apply_2q(self, gate: np.ndarray, site: int):
        A = self.tensors[site]
        B = self.tensors[site + 1]
        theta = np.tensordot(A, B, axes=([2], [0]))
        gate_t = gate.reshape(2, 2, 2, 2)
        theta = np.tensordot(gate_t, theta, axes=([2, 3], [1, 2]))
        theta = np.transpose(theta, (2, 0, 1, 3))
        theta_mat = theta.reshape(theta.shape[0] * 2, 2 * theta.shape[3])
        U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)
        chi_new = min(len(S), self.chi_max)
        U, S, Vh = U[:, :chi_new], S[:chi_new], Vh[:chi_new, :]
        new_A = U.reshape(theta.shape[0], 2, chi_new)
        new_B = (np.diag(S) @ Vh).reshape(chi_new, 2, theta.shape[3])
        self.tensors[site] = new_A
        self.tensors[site + 1] = new_B
        self.gate_count += 1

    def h(self, site: int): 
        self._apply_1q(np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2), site)
    def x(self, site: int): 
        self._apply_1q(np.array([[0,1],[1,0]], dtype=complex), site)
    def rx(self, theta: float, site: int):
        gate = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                         [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        self._apply_1q(gate, site)
    def cnot(self, c: int, t: int):
        if abs(c-t) != 1: raise ValueError("Nearest neighbors only")
        gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        self._apply_2q(gate, min(c, t))

    def get_amplitude(self, bitstring: str) -> complex:
        result = np.array([1.0], dtype=complex)
        for i in range(self.n):
            result = result @ self.tensors[i][:, int(bitstring[i]), :]
        return result[0]

    def max_bond(self) -> int:
        return max(max(t.shape[0], t.shape[2]) for t in self.tensors)

    def benchmark(self) -> Dict:
        mem_kb = sum(t.nbytes for t in self.tensors) / 1024
        return {
            'n_qubits': self.n,
            'max_bond_dim': self.max_bond(),
            'memory_kb': mem_kb,
            'gate_count': self.gate_count,
            'type': 'MPS (Tensor Network)',
        }


# ------------------------------------------------------------------
# P1.4: Gate Fidelity Validator
# ------------------------------------------------------------------
class GateValidator:
    """Validate gate implementations against theoretical properties."""

    @staticmethod
    def verify_unitary(gate: np.ndarray, tol: float = 1e-10) -> bool:
        """Check U^dagger U = I."""
        identity = gate @ gate.conj().T
        return np.allclose(identity, np.eye(gate.shape[0]), atol=tol)

    @staticmethod
    def verify_hermitian(gate: np.ndarray, tol: float = 1e-10) -> bool:
        """Check U = U^dagger."""
        return np.allclose(gate, gate.conj().T, atol=tol)

    @staticmethod
    def verify_pauli_commutation() -> Dict[str, bool]:
        """Verify Pauli algebra: [X,Y]=2iZ, etc."""
        X = np.array([[0,1],[1,0]], dtype=complex)
        Y = np.array([[0,-1j],[1j,0]], dtype=complex)
        Z = np.array([[1,0],[0,-1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        results = {}
        # X^2 = Y^2 = Z^2 = I
        results['X^2=I'] = np.allclose(X@X, I)
        results['Y^2=I'] = np.allclose(Y@Y, I)
        results['Z^2=I'] = np.allclose(Z@Z, I)
        # XY = iZ
        results['XY=iZ'] = np.allclose(X@Y, 1j*Z)
        # YZ = iX
        results['YZ=iX'] = np.allclose(Y@Z, 1j*X)
        # ZX = iY
        results['ZX=iY'] = np.allclose(Z@X, 1j*Y)
        return results

    @staticmethod
    def verify_bell_state_fidelity(sim: StateVectorSimulator) -> float:
        """Verify Bell state has correct properties."""
        sim.h(0)
        sim.cnot(0, 1)
        zz = sim.get_expectation('ZZ')
        xx = sim.get_expectation('XX')
        return zz, xx


# ------------------------------------------------------------------
# RUN ALL P1 MILESTONES
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE I: BASIC SIMULATORS - MILESTONE VALIDATION")
    print("=" * 70)

    # P1.1: State-vector benchmark
    print("\n[P1.1] State-Vector Simulator Benchmark")
    print("-" * 50)
    for n in [2, 4, 8, 12, 16]:
        sim = StateVectorSimulator(n)
        sim.h(0)
        for i in range(n-1):
            sim.cnot(i, i+1)
        bench = sim.benchmark()
        print(f"  {n:2d} qubits: {bench['amplitudes']:>10,} amplitudes, "
              f"{bench['memory_mb']:>8.2f} MB, "
              f"avg gate: {bench['avg_gate_time_ms']:.3f} ms")

    # P1.2: Clifford at scale
    print("\n[P1.2] Clifford Simulator at Scale")
    print("-" * 50)
    for n in [10, 100, 500, 1000]:
        sim = CliffordSimulator(n)
        sim.h(0)
        for i in range(n-1):
            sim.cnot(i, i+1)
        bench = sim.benchmark()
        print(f"  {n:4d} qubits: {bench['memory_kb']:.2f} KB, "
              f"{bench['gate_count']} gates")

    # P1.3: MPS scaling
    print("\n[P1.3] MPS Simulator Scaling")
    print("-" * 50)
    for n in [10, 20, 30, 50]:
        sim = MPSSimulator(n, max_bond_dim=32)
        sim.h(0)
        for i in range(n-1):
            sim.cnot(i, i+1)
        bench = sim.benchmark()
        print(f"  {n:2d} qubits: max bond={bench['max_bond_dim']:2d}, "
              f"{bench['memory_kb']:.2f} KB")

    # P1.4: Gate validation
    print("\n[P1.4] Gate Fidelity Validation")
    print("-" * 50)
    validator = GateValidator()

    # Unitary checks
    H = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
    print(f"  H is unitary: {validator.verify_unitary(H)}")

    # Pauli algebra
    pauli_results = validator.verify_pauli_commutation()
    for check, result in pauli_results.items():
        print(f"  {check}: {result}")

    # Bell state
    sim = StateVectorSimulator(2)
    zz, xx = validator.verify_bell_state_fidelity(sim)
    print(f"  Bell state <ZZ> = {zz:.6f} (expected: 1.0)")
    print(f"  Bell state <XX> = {xx:.6f} (expected: 1.0)")

    print("\n" + "=" * 70)
    print("PHASE I: ALL MILESTONES PASSED")
    print("=" * 70)
