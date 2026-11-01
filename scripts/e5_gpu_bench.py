"""
Phase E5: GPU validation + decoder latency benchmark (run on a CUDA pod).

[1] Torch state-vector backend on CUDA: correctness vs numpy, then
    per-layer time and effective memory bandwidth from n=20 up to the
    largest n that fits in GPU memory (complex128).
[2] Same random layers on the numpy engine (CPU baseline).
[3] Reference: Qiskit Aer GPU (cuStateVec) on the same circuit, if installed.
[4] Gross code [[144,12,12]] decoding, code capacity p=4%:
    own numpy BP+OSD-0 vs ldpc (CPU) vs CUDA-Q QEC nv-qldpc-decoder (GPU),
    throughput (batched) and single-syndrome latency.

Every section is independent; results go to $OUT/e5_report.json after
each one so a crash or timeout keeps what finished.
"""

import json, os, sys, time, traceback
import numpy as np

sys.path.insert(0, ".")

OUT = os.environ.get("OUT", "/workspace/e5_runs")
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("E5_DEVICE", "cuda")          # "cpu" for a dry run
MAXN = int(os.environ.get("E5_MAXN", "34"))
REPORT = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def save():
    with open(os.path.join(OUT, "e5_report.json"), "w") as f:
        json.dump(REPORT, f, indent=2)


def section(name):
    def wrap(fn):
        print("=" * 70, f"\n[{name}]", flush=True)
        t0 = time.time()
        try:
            REPORT[name] = fn()
            REPORT[name + "_status"] = "ok"
        except Exception as e:
            traceback.print_exc()
            REPORT[name + "_status"] = f"error: {e!r}"
        REPORT[name + "_seconds"] = round(time.time() - t0, 1)
        print(json.dumps(REPORT.get(name), indent=1)[:2000], flush=True)
        save()
    return wrap


def random_layer(rng, n):
    """One layer: random RZ.RY.RZ on every qubit, then CNOTs on (i, i+1) pairs."""
    ops = [("rot", q, rng.uniform(0, np.pi, 3)) for q in range(n)]
    ops += [("cx", q, q + 1) for q in range(0, n - 1, 2)]
    return ops


def apply_ops(sim, ops):
    for op in ops:
        if op[0] == "rot":
            a, b, c = op[2]
            sim.rz(a, op[1]); sim.ry(b, op[1]); sim.rz(c, op[1])
        else:
            sim.cnot(op[1], op[2])


# ----------------------------------------------------------------------
@section("env")
def _():
    import torch, platform
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 1))
    return info


# ----------------------------------------------------------------------
@section("sv_correctness")
def _():
    import torch
    from phase1_basic.simulators import StateVectorSimulator
    from phase1_basic.gpu_backend import TorchSVSimulator
    out = {}
    for n in (6, 10, 14, 18):
        rng = np.random.default_rng(n)
        layers = [random_layer(rng, n) for _ in range(4)]
        a, b = StateVectorSimulator(n), TorchSVSimulator(n, torch.device(DEV))
        for ops in layers:
            apply_ops(a, ops)
            apply_ops(b, ops)
        a._flush_all()
        b._flush_all()
        fa = a.state.reshape(-1)
        fb = b.state.detach().cpu().numpy().reshape(-1)
        fid = abs(np.vdot(fa, fb)) ** 2
        # engines may differ in qubit order; fall back to bit-reversal
        if fid < 1 - 1e-9:
            perm = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(2 ** n)])
            fid = max(fid, abs(np.vdot(fa, fb[perm])) ** 2)
        out[f"n{n}"] = fid
    out["pass"] = all(v > 1 - 1e-9 for k, v in out.items() if k.startswith("n"))
    return out


# ----------------------------------------------------------------------
def time_layers(make_sim, n, n_layers, sync=lambda: None, flush=lambda s: None):
    rng = np.random.default_rng(0)
    layers = [random_layer(rng, n) for _ in range(n_layers + 1)]
    sim = make_sim(n)
    apply_ops(sim, layers[0]); flush(sim); sync()        # warm-up
    t0 = time.perf_counter()
    for ops in layers[1:]:
        apply_ops(sim, ops)
    flush(sim); sync()
    dt = (time.perf_counter() - t0) / n_layers
    del sim
    return dt


@section("sv_gpu_scaling")
def _():
    import torch
    from phase1_basic.gpu_backend import TorchSVSimulator
    dev = torch.device(DEV)
    free = torch.cuda.mem_get_info()[0] if DEV == "cuda" else 8 * 2**30
    sync = torch.cuda.synchronize if DEV == "cuda" else (lambda: None)
    rows = []
    for n in range(20, MAXN):
        need = 3 * (2 ** n) * 16          # state + einsum temporaries
        if need > 0.92 * free:
            rows.append({"n": n, "skipped": f"needs ~{need / 2**30:.0f} GiB"})
            break
        try:
            dt = time_layers(lambda k: TorchSVSimulator(k, dev), n, 3,
                             sync=sync,
                             flush=lambda s: s._flush_all())
        except torch.cuda.OutOfMemoryError:
            rows.append({"n": n, "skipped": "OOM"})
            torch.cuda.empty_cache()
            break
        # one fused 1q sweep (n passes) + n/2 CNOTs: read+write 16B/amp each
        passes = n + (n // 2)
        gbps = passes * 2 * (2 ** n) * 16 / dt / 1e9
        rows.append({"n": n, "sec_per_layer": dt, "effective_GBps": round(gbps, 1)})
        print(rows[-1], flush=True)
        torch.cuda.empty_cache()
    return rows


@section("sv_cpu_baseline")
def _():
    from phase1_basic.simulators import StateVectorSimulator
    rows = []
    for n in [k for k in (20, 22, 24, 26) if k < MAXN]:
        dt = time_layers(lambda k: StateVectorSimulator(k), n, 2,
                         flush=lambda s: s._flush_all())
        rows.append({"n": n, "sec_per_layer": dt})
        print(rows[-1], flush=True)
    return rows


@section("sv_aer_gpu_reference")
def _():
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector", device="GPU", precision="double",
                       cuStateVec_enable=True)
    rows = []
    for n in (24, 26, 28, 30, 32):
        rng = np.random.default_rng(0)
        qc = QuantumCircuit(n)
        for _ in range(4):
            for op in random_layer(rng, n):
                if op[0] == "rot":
                    qc.rz(op[2][0], op[1]); qc.ry(op[2][1], op[1]); qc.rz(op[2][2], op[1])
                else:
                    qc.cx(op[1], op[2])
        qc.save_statevector()
        tq = transpile(qc, sim, optimization_level=0)
        try:
            t0 = time.perf_counter()
            r = sim.run(tq).result()
            dt = (time.perf_counter() - t0) / 4
            ok = r.success
        except Exception as e:
            rows.append({"n": n, "error": repr(e)[:200]})
            break
        rows.append({"n": n, "sec_per_layer_incl_overhead": dt, "success": ok})
        print(rows[-1], flush=True)
    return rows


# ----------------------------------------------------------------------
@section("decoders")
def _():
    from phase2_noise.qldpc import gross_code, BpOsdDecoder, logical_operators
    hx, hz = gross_code()
    _, lz = logical_operators(hx, hz)
    p, shots = 0.04, 5000
    rng = np.random.default_rng(7)
    err = (rng.random((shots, 144)) < p).astype(np.uint8)
    synd = (err.astype(np.int64) @ hz.T % 2).astype(np.uint8)

    def score(corr):
        resid = err[:len(corr)] ^ corr
        ok_synd = not np.any(resid.astype(np.int64) @ hz.T % 2)
        fails = int(np.any(resid.astype(np.int64) @ lz.T % 2, axis=1).sum())
        return ok_synd, fails

    res = {"p": p, "shots": shots}

    dec = BpOsdDecoder(hz, p)
    t0 = time.perf_counter(); corr = dec.decode_batch(synd); dt = time.perf_counter() - t0
    ok, f = score(corr)
    lat = []
    for s in synd[:200]:
        t1 = time.perf_counter(); dec.decode_batch(s[None]); lat.append(time.perf_counter() - t1)
    res["qweave_numpy"] = {"us_per_shot_batched": dt / shots * 1e6,
                           "median_latency_us": float(np.median(lat)) * 1e6,
                           "failures": f, "syndrome_ok": ok}

    try:
        from ldpc import BpOsdDecoder as Ref
        ref = Ref(hz, error_rate=p, bp_method="minimum_sum", ms_scaling_factor=0.625,
                  max_iter=100, osd_method="osd0", osd_order=0)
        lat = []
        corr = np.zeros_like(err)
        t0 = time.perf_counter()
        for i, s in enumerate(synd):
            t1 = time.perf_counter(); corr[i] = ref.decode(s); lat.append(time.perf_counter() - t1)
        dt = time.perf_counter() - t0
        ok, f = score(corr)
        res["ldpc_cpu"] = {"us_per_shot": dt / shots * 1e6,
                           "median_latency_us": float(np.median(lat)) * 1e6,
                           "failures": f, "syndrome_ok": ok}
    except Exception as e:
        res["ldpc_cpu"] = {"error": repr(e)[:300]}

    try:
        import cudaq_qec as qec
        gdec = qec.get_decoder("nv-qldpc-decoder", hz.astype(np.uint8),
                               error_rate_vec=np.full(144, p), use_osd=True,
                               osd_method=0, max_iterations=100, use_sparsity=True)
        gdec.decode(synd[0].astype(np.float64).tolist())      # warm-up
        lat = []
        corr = np.zeros_like(err)
        for i, s in enumerate(synd[:2000]):
            t1 = time.perf_counter()
            r = gdec.decode(s.astype(np.float64).tolist())
            lat.append(time.perf_counter() - t1)
            corr[i] = (np.asarray(r.result) > 0.5).astype(np.uint8)
        ok, f = score(corr[:2000])
        entry = {"median_latency_us": float(np.median(lat)) * 1e6,
                 "failures_first_2000": f, "syndrome_ok": ok}
        if hasattr(gdec, "decode_batch"):
            t0 = time.perf_counter()
            rs = gdec.decode_batch([s.astype(np.float64).tolist() for s in synd])
            entry["us_per_shot_batched"] = (time.perf_counter() - t0) / shots * 1e6
        res["cudaq_qec_gpu"] = entry
    except Exception as e:
        traceback.print_exc()
        res["cudaq_qec_gpu"] = {"error": repr(e)[:300]}
    return res


REPORT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print("E5 done ->", os.path.join(OUT, "e5_report.json"))
