"""
Phase B-1: Circuit Intermediate Representation (IR) + OpenQASM I/O
==================================================================
Defines a gate-level Circuit IR shared by all simulators, with:

  * programmatic gate builders (h, cnot, rz, u3, measure, ...)
  * OpenQASM 2.0 export / import (qelib1 gate subset)
  * conversion to the Clifford affine-sampler op list (``to_sampler_ops``)
  * convenience runners (state-vector / Clifford shots)

The IR is the interoperability layer planned in the survey: one circuit
description drives every backend.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import re
import numpy as np


@dataclass
class Gate:
    name: str          # canonical name: h x y z s t rx ry rz u3 cx cz swap ccx measure
    targets: tuple
    params: tuple = ()     # e.g. rx: (theta,)
    cbit: Optional[int] = None   # for measure

    def __repr__(self):
        p = f"({', '.join(f'{v:.6g}' for v in self.params)})" if self.params else ""
        return f"{self.name}{p}{self.targets}"


class Circuit:
    """Gate-level quantum circuit IR."""

    QASM_NAME = {'cx': 'cx', 'cnot': 'cx'}

    def __init__(self, n_qubits: int, name: str = "circuit"):
        self.n_qubits = n_qubits
        self.name = name
        self.gates: List[Gate] = []
        self.n_clbits = 0

    # ---------------- builders ----------------
    def _add(self, name, targets, params=(), cbit=None):
        self.gates.append(Gate(name, tuple(targets), tuple(params), cbit))
        if cbit is not None:
            self.n_clbits = max(self.n_clbits, cbit + 1)
        return self

    def h(self, q): return self._add('h', (q,))
    def x(self, q): return self._add('x', (q,))
    def y(self, q): return self._add('y', (q,))
    def z(self, q): return self._add('z', (q,))
    def s(self, q): return self._add('s', (q,))
    def t(self, q): return self._add('t', (q,))
    def rx(self, theta, q): return self._add('rx', (q,), (theta,))
    def ry(self, theta, q): return self._add('ry', (q,), (theta,))
    def rz(self, theta, q): return self._add('rz', (q,), (theta,))
    def u3(self, theta, phi, lam, q): return self._add('u3', (q,), (theta, phi, lam))
    def cnot(self, c, t): return self._add('cx', (c, t))
    def cz(self, c, t): return self._add('cz', (c, t))
    def swap(self, a, b): return self._add('swap', (a, b))
    def ccx(self, c1, c2, t): return self._add('ccx', (c1, c2, t))
    def measure(self, q, cbit=None):
        return self._add('measure', (q,), cbit=self.n_clbits if cbit is None else cbit)

    # ---------------- QASM export ----------------
    @staticmethod
    def _fmt_num(v: float) -> str:
        for k in (1, 2, 3, 4, 6, 8):
            r = v / (np.pi / k)
            if abs(r - round(r)) < 1e-9:
                rr = int(round(r))
                if k == 1:
                    return f"{rr}*pi" if rr != 1 else "pi"
                return f"{rr}*pi/{k}" if rr != 1 else f"pi/{k}"
        return f"{v:.10g}"

    def to_qasm(self, version: int = 2) -> str:
        L = [f"OPENQASM {version}.0;"]
        if version == 2:
            L.append('include "qelib1.inc";')
            L.append(f"qreg q[{self.n_qubits}];")
            if self.n_clbits:
                L.append(f"creg c[{self.n_clbits}];")
        else:  # OpenQASM 3
            L.append(f"qubit[{self.n_qubits}] q;")
            if self.n_clbits:
                L.append(f"bit[{self.n_clbits}] c;")
        for g in self.gates:
            if g.name == 'measure':
                if version == 2:
                    L.append(f"measure q[{g.targets[0]}] -> c[{g.cbit}];")
                else:
                    L.append(f"c[{g.cbit}] = measure(q[{g.targets[0]}]);")
                continue
            qasm_name = 'cx' if g.name == 'cnot' else g.name
            tgt = ", ".join(f"q[{t}]" for t in g.targets)
            if g.params:
                prm = ", ".join(self._fmt_num(p) for p in g.params)
                L.append(f"{qasm_name}({prm}) {tgt};")
            else:
                L.append(f"{qasm_name} {tgt};")
        return "\n".join(L) + "\n"

    # ---------------- QASM import ----------------
    _GATE_RE = re.compile(
        r"^\s*([a-zA-Z][a-zA-Z0-9_]*)\s*(?:\(([^)]*)\))?\s+(.+?)\s*;\s*$")
    _MEAS_RE = re.compile(
        r"^\s*measure\s+q\[(\d+)\]\s*->\s*c\[(\d+)\]\s*;\s*$")
    _QREG_RE = re.compile(r"^\s*qreg\s+q\[(\d+)\]\s*;\s*$")
    _CREG_RE = re.compile(r"^\s*creg\s+c\[(\d+)\]\s*;\s*$")
    _QUBIT_RE = re.compile(r"^\s*qubit(?:\[(\d+)\])?\s+q\s*;\s*$")
    _BIT_RE = re.compile(r"^\s*bit(?:\[(\d+)\])?\s+c\s*;\s*$")
    _MEAS3_RE = re.compile(
        r"^\s*c\[(\d+)\]\s*=\s*measure\s*\(\s*q\[(\d+)\]\s*\)\s*;\s*$")
    _TGT_RE = re.compile(r"q\[(\d+)\]")

    _ALIASES = {'id': 'h', 'u': 'u3', 'CX': 'cx', 'toffoli': 'ccx'}

    @classmethod
    def from_qasm(cls, text: str) -> "Circuit":
        lines = []
        for raw in text.splitlines():
            raw = raw.split("//")[0].strip()
            if raw:
                lines.append(raw)
        if not lines or not lines[0].startswith("OPENQASM"):
            raise ValueError("missing OPENQASM header")
        version = int(lines[0].split()[1].rstrip(';').split('.')[0])

        n_q = n_c = None
        circ = None
        for ln in lines[1:]:
            m = cls._QREG_RE.match(ln) or cls._QUBIT_RE.match(ln)
            if m:
                n_q = int(m.group(1) or 1)
                circ = cls(n_q)
                continue
            m = cls._CREG_RE.match(ln) or cls._BIT_RE.match(ln)
            if m:
                n_c = int(m.group(1) or 1)
                continue
            m = cls._MEAS3_RE.match(ln)
            if m:
                circ.measure(int(m.group(2)), int(m.group(1)))
                continue
            if ln.startswith("include") or ln.startswith("gate ") or ln == "":
                continue
            if circ is None:
                raise ValueError("qreg must precede gate statements")
            m = cls._MEAS_RE.match(ln)
            if m:
                circ.measure(int(m.group(1)), int(m.group(2)))
                continue
            m = cls._GATE_RE.match(ln)
            if not m:
                raise ValueError(f"cannot parse line: {ln}")
            name, params_s, tgt_s = m.groups()
            name = cls._ALIASES.get(name, name.lower())
            tgts = tuple(int(t) for t in cls._TGT_RE.findall(tgt_s))
            params = ()
            if params_s:
                allowed = {"pi": np.pi, "sin": np.sin, "cos": np.cos,
                           "exp": np.exp, "sqrt": np.sqrt}
                params = tuple(float(eval(p, {"__builtins__": {}}, allowed))
                               for p in params_s.split(","))
            if name in ('rx', 'ry', 'rz'):
                circ._add(name, (tgts[0],), params)
            elif name == 'u3':
                circ._add('u3', (tgts[0],), params)
            elif name in ('h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg'):
                if name == 'sdg':
                    circ._add('s', (tgts[0],)); circ._add('s', (tgts[0],))
                    circ._add('z', (tgts[0],))
                elif name == 'tdg':
                    circ._add('t', (tgts[0],)); circ._add('t', (tgts[0],))
                    circ._add('t', (tgts[0],)); circ._add('z', (tgts[0],))
                else:
                    circ._add(name, (tgts[0],))
            elif name in ('cx', 'cz', 'swap', 'ccx'):
                circ._add(name, tgts)
            elif name == 'barrier':
                continue
            else:
                raise ValueError(f"unsupported gate: {name}")
        if circ is None:
            raise ValueError("no qreg found")
        return circ

    # ---------------- backend drivers ----------------
    def to_sampler_ops(self, noise: Optional[dict] = None) -> List[tuple]:
        """
        Convert to CliffordSimulator.sample op list.
        noise: optional dict with keys 'after_1q' / 'after_2q' ->
               ('depolarize', p) | ('x_error', p) | ('z_error', p)
        """
        ops = []
        noise = noise or {}
        for g in self.gates:
            if g.name == 'measure':
                ops.append(('mz', g.targets[0]))
                continue
            if g.name == 'cx':
                ops.append(('cnot', g.targets[0], g.targets[1]))
                kind = 'after_2q'
            elif g.name == 'cnot':
                ops.append(('cnot', g.targets[0], g.targets[1]))
                kind = 'after_2q'
            elif g.name in ('cz', 'swap', 'ccx'):
                raise ValueError(f"{g.name} not supported by Clifford sampler")
            else:
                if g.name not in ('h', 's'):
                    raise ValueError(
                        f"non-Clifford gate {g.name} cannot run on Clifford backend")
                ops.append((g.name, g.targets[0]))
                kind = 'after_1q'
            ch = noise.get(kind)
            if ch:
                for t in g.targets:
                    ops.append((ch[0], t, ch[1]))
        return ops

    def run_statevector(self):
        from phase1_basic.simulators import StateVectorSimulator
        sim = StateVectorSimulator(self.n_qubits)
        for g in self.gates:
            if g.name == 'measure':
                sim.measure(g.targets[0])
            elif g.name == 'cx':
                sim.cnot(*g.targets)
            elif g.name in ('rx', 'ry', 'rz'):
                getattr(sim, g.name)(g.params[0], g.targets[0])
            elif g.name == 'u3':
                sim.u3(*g.params, g.targets[0])
            else:
                getattr(sim, g.name)(*g.targets)
        return sim

    def run_clifford_shots(self, shots: int, seed: Optional[int] = None,
                           noise: Optional[dict] = None) -> np.ndarray:
        from phase1_basic.simulators import CliffordSimulator
        ops = self.to_sampler_ops(noise=noise)
        return CliffordSimulator(self.n_qubits).sample(ops, shots=shots, seed=seed)

    def __repr__(self):
        return f"Circuit({self.n_qubits}q, {len(self.gates)} gates)"


def _smoke():
    # smoke: GHZ -> QASM -> parse back -> compare
    c = Circuit(3)
    c.h(0).cnot(0, 1).cnot(1, 2).measure(0).measure(1).measure(2)
    qasm = c.to_qasm()
    print(qasm)
    c2 = Circuit.from_qasm(qasm)
    assert [g.name for g in c.gates] == [g.name for g in c2.gates]
    assert [g.targets for g in c.gates] == [g.targets for g in c2.gates]
    print("roundtrip OK:", c2)


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    _smoke()
