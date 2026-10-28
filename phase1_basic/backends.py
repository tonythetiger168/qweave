"""
Phase C: Pluggable backend adapters + auto-selection
=====================================================
One Circuit IR drives multiple simulation backends:

  * NativeSVBackend   -- optimized local state-vector engine
  * NativeCliffordBackend -- affine Pauli-frame sampler
  * AerBackend        -- Qiskit Aer statevector (external reference)
  * StimBackend       -- Google Stim sampler (external reference)
  * auto_backend()    -- pick a backend from circuit features
     (Clifford-only -> frame sampler; <=SV_LIMIT qubits -> state-vector;
      else raise / suggest MPS)
"""

from typing import Optional
import numpy as np

from phase1_basic.circuit import Circuit

SV_QUBIT_LIMIT = 24          # ~256 GB state vector; raise with GPU backend
CLIFFORD_GATES = {'h', 's', 'x', 'y', 'z', 'cx', 'cz', 'swap', 'measure'}


# ----------------------------------------------------------------------
class NativeSVBackend:
    name = "native-statevector"

    def run(self, circuit: Circuit):
        return circuit.run_statevector().state


# ----------------------------------------------------------------------
class NativeCliffordBackend:
    name = "native-clifford-sampler"

    def run(self, circuit: Circuit, shots: int = 4096,
            noise: Optional[dict] = None, seed: Optional[int] = None):
        return circuit.run_clifford_shots(shots, seed=seed, noise=noise)


# ----------------------------------------------------------------------
class AerBackend:
    name = "aer-statevector"

    def run(self, circuit: Circuit):
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        n = circuit.n_qubits
        qc = QuantumCircuit(n)
        for g in circuit.gates:
            if g.name == 'measure':
                continue
            if g.name == 'cx':
                qc.cx(*g.targets)
            elif g.name in ('rx', 'ry', 'rz'):
                getattr(qc, g.name)(g.params[0], g.targets[0])
            elif g.name == 'u3':
                qc.u(*g.params, g.targets[0])
            else:
                getattr(qc, g.name)(*g.targets)
        qc.save_statevector()
        sim = AerSimulator(method='statevector')
        result = sim.run(qc).result()
        return np.array(result.get_statevector(qc), dtype=complex)


# ----------------------------------------------------------------------
class StimBackend:
    name = "stim-sampler"

    def run(self, circuit: Circuit, shots: int = 4096,
            noise: Optional[dict] = None, seed: Optional[int] = None):
        """Stim must agree with the native sampler for Clifford circuits."""
        import stim
        cir = stim.Circuit()
        for g in circuit.gates:
            if g.name == 'h':
                cir.append("H", g.targets[0])
            elif g.name == 's':
                cir.append("S", g.targets[0])
            elif g.name == 'x':
                cir.append("X", g.targets[0])
            elif g.name == 'y':
                cir.append("Y", g.targets[0])
            elif g.name == 'z':
                cir.append("Z", g.targets[0])
            elif g.name == 'cx':
                cir.append("CNOT", list(g.targets))
            elif g.name == 'cz':
                cir.append("CZ", list(g.targets))
            elif g.name == 'measure':
                cir.append("M", g.targets[0])
            else:
                raise ValueError(f"StimBackend: {g.name} not Clifford")
        sampler = cir.compile_sampler(seed=seed)
        n_meas = sum(1 for g in circuit.gates if g.name == 'measure')
        data = sampler.sample(shots)
        return np.asarray(data, dtype=np.uint8)[:, :n_meas] if n_meas else \
            np.zeros((shots, 0), np.uint8)


# ----------------------------------------------------------------------
def auto_backend(circuit: Circuit) -> str:
    """Decision tree (survey arXiv:2401.06188 style)."""
    names = {g.name for g in circuit.gates}
    if names <= CLIFFORD_GATES:
        return NativeCliffordBackend.name
    if circuit.n_qubits <= SV_QUBIT_LIMIT:
        try:
            import torch
            if torch.cuda.is_available():
                return "torch-statevector"      # GPU path
        except ImportError:
            pass
        return NativeSVBackend.name
    raise ValueError(
        f"{circuit.n_qubits} qubits with non-Clifford gates: use MPS/GPU backend")


def _all_backends():
    from phase1_basic.gpu_backend import TorchSVBackend
    return (NativeSVBackend(), NativeCliffordBackend(), AerBackend(),
            StimBackend(), TorchSVBackend())


BACKENDS = {b.name: b for b in _all_backends()}
