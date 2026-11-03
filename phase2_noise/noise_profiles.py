"""
Phase E3: per-platform noise profiles
=====================================
Circuit-level noise models built from published median error rates, for
estimating what a given processor would do with a QEC code before running it.

A profile gives separate rates for single-qubit gates, two-qubit gates,
measurement, reset, and idling (from T1/T2 over the measure+reset window).
Idling uses the Pauli-twirled amplitude/phase damping channel
    pX = pY = (1 - exp(-t/T1)) / 4
    pZ = (1 - exp(-t/T2)) / 2 - (1 - exp(-t/T1)) / 4

The numbers are public medians as of late 2025, rounded. They are for
comparison between platforms, not calibrations of any one device; see each
profile's ``source`` and check before quoting.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import math


@dataclass(frozen=True)
class NoiseProfile:
    name: str
    p1: float                 # single-qubit gate error (depolarizing)
    p2: float                 # two-qubit gate error (depolarizing)
    p_meas: float             # measurement flip
    p_reset: float            # reset flip
    t1_us: Optional[float] = None
    t2_us: Optional[float] = None
    t_measure_us: float = 0.0  # measure + reset window in which data qubits idle
    cycle_us: Optional[float] = None
    source: str = ""

    def idle_pauli(self) -> Optional[tuple]:
        """(pX, pY, pZ) for a data qubit idling through measurement and reset."""
        if not self.t1_us or not self.t2_us or self.t_measure_us <= 0:
            return None
        t = self.t_measure_us
        a = 1 - math.exp(-t / self.t1_us)
        b = 1 - math.exp(-t / self.t2_us)
        px = a / 4
        pz = max(b / 2 - a / 4, 0.0)
        return (px, px, pz)

    @classmethod
    def uniform(cls, p: float) -> "NoiseProfile":
        """Standard uniform circuit noise: every location fails with probability p."""
        return cls(name=f"uniform p={p}", p1=p, p2=p, p_meas=p, p_reset=p)

    def scaled(self, factor: float) -> "NoiseProfile":
        """Every error rate (and 1/T1, 1/T2) multiplied by ``factor``."""
        return NoiseProfile(
            name=f"{self.name} x{factor:g}", p1=self.p1 * factor, p2=self.p2 * factor,
            p_meas=self.p_meas * factor, p_reset=self.p_reset * factor,
            t1_us=None if self.t1_us is None else self.t1_us / factor,
            t2_us=None if self.t2_us is None else self.t2_us / factor,
            t_measure_us=self.t_measure_us, cycle_us=self.cycle_us, source=self.source)


PROFILES: Dict[str, NoiseProfile] = {
    "google_willow": NoiseProfile(
        name="Google Willow (105q superconducting)",
        p1=0.00035, p2=0.0033, p_meas=0.0077, p_reset=0.001,
        t1_us=68.0, t2_us=89.0, t_measure_us=0.88, cycle_us=1.1,
        source="Google Quantum AI, Nature 638, 920 (2025): mean simultaneous 1q 0.035%, "
               "CZ 0.33%, readout 0.77%, T1 68 us, T2 (CPMG) 89 us, 1.1 us cycle"),
    "ibm_heron": NoiseProfile(
        name="IBM Heron r2 (156q superconducting)",
        p1=0.00025, p2=0.003, p_meas=0.012, p_reset=0.002,
        t1_us=250.0, t2_us=150.0, t_measure_us=1.6, cycle_us=2.5,
        source="IBM Quantum platform medians for Heron r2 (2025): 1q ~2.5e-4, CZ ~3e-3, "
               "readout ~1.2%, T1 ~250 us, T2 ~150 us"),
    "quantinuum_h2": NoiseProfile(
        name="Quantinuum H2 (56q trapped ion)",
        p1=0.00003, p2=0.001, p_meas=0.0015, p_reset=0.0005,
        source="Quantinuum H2 spec sheet (2025): 1q 3e-5, 2q 1e-3, SPAM ~1.5e-3; "
               "idle dephasing ignored (seconds-scale coherence)"),
    "ionq_forte": NoiseProfile(
        name="IonQ Forte (36q trapped ion)",
        p1=0.0002, p2=0.004, p_meas=0.005, p_reset=0.001,
        source="IonQ Forte published medians (2025): 1q ~2e-4, 2q ~4e-3, SPAM ~5e-3"),
    "neutral_atom": NoiseProfile(
        name="Neutral atoms (Harvard/QuEra, Rb)",
        p1=0.0005, p2=0.005, p_meas=0.01, p_reset=0.003,
        source="Bluvstein et al., Nature 626, 58 (2024) and follow-ups: CZ 99.5%, "
               "1q ~99.95%, readout ~99%"),
}


def get_profile(name: str) -> NoiseProfile:
    return PROFILES[name]
