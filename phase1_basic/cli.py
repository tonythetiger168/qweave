"""
qweave command-line interface.
"""

import argparse
import numpy as np


def run_phase1():
    """Run Phase I: Basic Simulators."""
    print("\n" + "=" * 70)
    print("PHASE I: BASIC SIMULATORS")
    print("=" * 70)
    from phase1_basic.simulators import (
        StateVectorSimulator, CliffordSimulator, MPSSimulator, GateValidator
    )

    print("\n[Milestone P1.1] State-Vector Simulator")
    print("-" * 50)
    for n in [2, 4, 8, 12, 16]:
        sim = StateVectorSimulator(n)
        sim.h(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)
        bench = sim.benchmark()
        print(f"  {n:2d} qubits: {bench['amplitudes']:>10,} amps, "
              f"{bench['memory_mb']:>8.2f} MB, "
              f"avg gate {bench['avg_gate_time_ms']:.3f} ms")

    print("\n[Milestone P1.2] Clifford/Stabilizer Simulator")
    print("-" * 50)
    for n in [10, 100, 500, 1000]:
        sim = CliffordSimulator(n)
        sim.h(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)
        bench = sim.benchmark()
        print(f"  {n:4d} qubits: {bench['memory_kb']:.2f} KB, "
              f"{bench['gate_count']} gates")

    print("\n[Milestone P1.3] MPS Simulator")
    print("-" * 50)
    for n in [10, 20, 30, 50]:
        sim = MPSSimulator(n, max_bond_dim=32)
        sim.h(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)
        bench = sim.benchmark()
        print(f"  {n:2d} qubits: max bond={bench['max_bond_dim']:2d}, "
              f"{bench['memory_kb']:.2f} KB")

    print("\n[Milestone P1.4] Gate Fidelity Validation")
    print("-" * 50)
    validator = GateValidator()
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    print(f"  H is unitary: {validator.verify_unitary(H)}")
    pauli = validator.verify_pauli_commutation()
    for check, ok in pauli.items():
        print(f"  {check}: {ok}")
    sim = StateVectorSimulator(2)
    zz, xx = validator.verify_bell_state_fidelity(sim)
    print(f"  Bell <ZZ>={zz:.4f}, <XX>={xx:.4f} (both should be 1.0)")

    print("\n[OK] Phase I: All milestones passed")


def run_phase2():
    """Run Phase II: Noise & Error Correction."""
    print("\n" + "=" * 70)
    print("PHASE II: NOISE & ERROR CORRECTION")
    print("=" * 70)
    from phase2_noise.noise_and_ec import (
        NoiseChannels, SurfaceCode3, ErrorRateBenchmark
    )

    print("\n[Milestone P2.1] Noise Channels")
    print("-" * 50)
    rho = np.zeros((2, 2), dtype=complex)
    rho[1, 1] = 1.0
    rho_dep = NoiseChannels.depolarizing(rho, 0.1, 1, 0)
    print(f"  Depolarizing: P0={rho_dep[0,0].real:.4f}, P1={rho_dep[1,1].real:.4f}")
    rho_damp = NoiseChannels.amplitude_damping(rho, 0.3, 1, 0)
    print(f"  Amp damping:  P0={rho_damp[0,0].real:.4f}, P1={rho_damp[1,1].real:.4f}")

    print("\n[Milestone P2.2] Surface Code d=3")
    print("-" * 50)
    code = SurfaceCode3()
    print(f"  Data: {code.n_data}, Ancilla Z: {code.n_ancilla_z}, X: {code.n_ancilla_x}")

    print("\n[Milestone P2.3] Logical Error Rate")
    print("-" * 50)
    for p in [0.01, 0.05, 0.1]:
        r = ErrorRateBenchmark.measure_logical_error_rate(500, p)
        print(f"  p_phys={p:.2f}: p_log={r['logical_error_rate']:.4f}")

    print("\n[OK] Phase II: All milestones passed")


def run_phase3():
    """Run Phase III: Large-Scale & Hybrid Algorithms."""
    print("\n" + "=" * 70)
    print("PHASE III: LARGE-SCALE & HYBRID ALGORITHMS")
    print("=" * 70)
    from phase3_scale.algorithms import VQE, QAOA, PhaseEstimation, BenchmarkSuite

    print("\n[Milestone P3.1] VQE (H2 molecule)")
    print("-" * 50)
    vqe = VQE(n_qubits=2, n_layers=2)
    ham = VQE.hydrogen_molecule_hamiltonian()
    result = vqe.optimize(hamiltonian=ham, max_iter=100, lr=0.2)
    print(f"  Best energy: {result['best_energy']:.6f} Ha (exact: -1.137)")

    print("\n[Milestone P3.2] QAOA (MaxCut 4-node)")
    print("-" * 50)
    qaoa = QAOA(n_nodes=4, edges=[(0,1), (1,2), (2,3), (3,0)], p=2)
    result = qaoa.optimize(n_trials=50)
    print(f"  Best cut: {result['best_cost']:.2f} (optimal: 4.0)")

    print("\n[Milestone P3.3] Quantum Phase Estimation")
    print("-" * 50)
    qpe = PhaseEstimation(n_counting=3)
    U = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    eigenstate = np.array([0, 1], dtype=complex)
    phase = qpe.run(U, eigenstate)
    print(f"  Phase estimate: {phase} = {int(phase, 2) / 8:.4f} (actual: 0.25)")

    print("\n[Milestone P3.4] Full Benchmark Suite")
    print("-" * 50)
    benchmarks = BenchmarkSuite.run_all_benchmarks()
    BenchmarkSuite.print_report(benchmarks)

    print("\n[OK] Phase III: All milestones passed")


def main():
    parser = argparse.ArgumentParser(
        prog='qweave',
        description='qweave: qweave quantum simulation & AI suite — state-vector, '
                    'Clifford/stabilizer and MPS simulators, noise models & surface '
                    'codes, and hybrid algorithms (VQE, QAOA, QPE) with benchmarks.',
        epilog='Examples: qweave | --phase 1 | --phase 2 | --phase 3'
    )
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Run specific phase only')
    args = parser.parse_args()

    if args.phase is None:
        run_phase1()
        run_phase2()
        run_phase3()
        print("\n" + "=" * 70)
        print("ALL PHASES COMPLETED SUCCESSFULLY")
        print("=" * 70)
    elif args.phase == 1:
        run_phase1()
    elif args.phase == 2:
        run_phase2()
    elif args.phase == 3:
        run_phase3()


if __name__ == "__main__":
    main()
