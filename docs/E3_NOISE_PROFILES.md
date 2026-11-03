# E3: Per-platform noise profiles (2026-11-03)

Code:
- `phase2_noise/noise_profiles.py`: the `NoiseProfile` class and 5 presets.
- `phase2_noise/surface_code.py`: `rotated_memory_z_circuit(d, rounds, p)` now accepts a profile as well as a float.
- `scripts/e3_platform_lambda.py`: the platform sweep.

Checks: `phase2_noise/validate_e3.py`, which runs in CI in about 12 s. Everything ran on CPU.

## Model
A profile has separate rates for:
- single-qubit gates (depolarizing)
- two-qubit gates (depolarizing)
- measurement flips
- reset flips
- data-qubit idling during the measure+reset window, from T1/T2 as a Pauli-twirled damping channel

The presets are public medians as of late 2025, rounded, with a source string on each. They are for comparing platforms, not calibrations of any one device.
A float `p` still builds exactly the same circuit as before, and validate_e3 [1] checks that.

## Check against a published result: Google Willow
I used Willow's published component numbers with no fitting: 1q 0.035%, CZ 0.33%, readout 0.77%, T1 68 µs, T2 89 µs, 0.88 µs measure+reset.

| | qweave (this profile) | Google, Nature 638, 920 (2025) |
|---|---|---|
| Λ (d = 3 → 5 → 7) | **2.11** | **2.14 ± 0.02** |
| ε7, logical error per cycle | 0.125% | 0.143% |

Λ agrees to within 2%. ε7 is about 13% lower than measured. That is expected, because this model leaves out leakage, crosstalk and correlated errors, which the real device has.

## All platforms
These results come from a rotated surface-code memory decoded with PyMatching, with 3d rounds and 200k, 400k and 1M shots for d = 3, 5 and 7:

| platform | ε3 | ε5 | ε7 | Λ |
|---|---|---|---|---|
| Google Willow | 5.6e-3 | 2.6e-3 | 1.25e-3 | 2.11 |
| IBM Heron r2 | 3.8e-3 | 1.4e-3 | 5.4e-4 | 2.64 |
| Quantinuum H2 | 1.3e-4 | 1.2e-5 | 1.1e-6 | 11.3 |
| IonQ Forte | 1.9e-3 | 7.1e-4 | 2.4e-4 | 2.86 |
| Neutral atoms | 3.5e-3 | 1.7e-3 | 7.4e-4 | 2.18 |

Reading this table:
- **Only Willow's number is a check against a measurement.** The others are what each platform's published gate numbers imply for this code. No platform except Google has published a surface-code Λ to compare against.
- **IBM Heron:** the chip has heavy-hex connectivity, not the square grid a surface code needs. The real circuit needs extra SWAPs or a different code, such as the gross code from E1, so 2.64 is an upper bound.
- **Trapped ions (H2, Forte):** the model ignores their slow cycles (milliseconds) and shuttling errors. The high Λ shows how low their 2-qubit error is, not how fast they can run.
- **Neutral atoms:** atom loss and erasure aren't modelled. Erasure conversion would raise Λ.
- **Sensitivity:** doubling every Willow rate drops Λ from 2.12 to 1.17 (validate_e3 [4]), so the platforms sit close to threshold.

## Next steps
- Add leakage and correlated (crosstalk) channels, then refit against Willow's ε7.
- Run each profile through the E1 gross-code circuit to compare qLDPC and surface code per platform.
- Feed the profiles into E2 Pauli propagation as Pauli noise, to compare against IBM's noisy kicked-Ising data.
