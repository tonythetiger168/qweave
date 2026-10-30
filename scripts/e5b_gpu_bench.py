"""
Phase E5-b: cuStateVec backend + batched GPU decoder (run on a CUDA pod).

[1] CuStateVecSimulator vs TorchSVSimulator: fidelity after random layers
[2] Per-layer time, torch vs cuStateVec, n = 22 .. largest that fits
[3] Gross code decoding at p = 4%, code capacity:
    torch-cuda batched BP+OSD at several batch sizes, single-syndrome
    latency, vs ldpc (CPU) and CUDA-Q nv-qldpc-decoder with batching on.

Uses the helpers of e5_gpu_bench.py (same layers, same report format).
"""

import json, os, sys, time, traceback
import numpy as np

sys.path.insert(0, ".")
os.environ.setdefault("OUT", "/workspace/e5b_runs")
OUT = os.environ["OUT"]
os.makedirs(OUT, exist_ok=True)
REPORT = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def save():
    with open(os.path.join(OUT, "e5b_report.json"), "w") as f:
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
        print(json.dumps(REPORT.get(name), indent=1)[:3000], flush=True)
        save()
    return wrap


def random_layer(rng, n):
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


@section("env")
def _():
    import torch
    from phase1_basic import custatevec_backend as cb
    p = torch.cuda.get_device_properties(0)
    return {"torch": torch.__version__, "gpu": p.name,
            "gpu_mem_gb": round(p.total_memory / 2**30, 1),
            "custatevec_available": cb.available()}


@section("cusv_correctness")
def _():
    import torch
    from phase1_basic.gpu_backend import TorchSVSimulator
    from phase1_basic.custatevec_backend import CuStateVecSimulator
    out = {}
    for n in (4, 8, 12, 16, 20):
        rng = np.random.default_rng(n)
        layers = [random_layer(rng, n) for _ in range(4)]
        a = TorchSVSimulator(n, torch.device("cuda"))
        b = CuStateVecSimulator(n)
        for ops in layers:
            apply_ops(a, ops); apply_ops(b, ops)
        # extra gates exercising x / z / cz / generic 2q paths
        for sim in (a, b):
            sim.x(0); sim.z(n - 1); sim.cz(n - 1, 0); sim.h(1)
            sim._apply_2q(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1j]]), 0, n - 1)
            sim._flush_all()
        torch.cuda.synchronize()
        out[f"n{n}"] = float(abs(torch.vdot(a.state, b.state)) ** 2)
    out["pass"] = all(v > 1 - 1e-9 for k, v in out.items() if k.startswith("n"))
    return out


def time_layers(make, n, n_layers=3):
    import torch
    rng = np.random.default_rng(0)
    layers = [random_layer(rng, n) for _ in range(n_layers + 1)]
    sim = make(n)
    apply_ops(sim, layers[0]); sim._flush_all(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ops in layers[1:]:
        apply_ops(sim, ops)
    sim._flush_all(); torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_layers
    del sim
    torch.cuda.empty_cache()
    return dt


@section("sv_scaling")
def _():
    import torch
    from phase1_basic.gpu_backend import TorchSVSimulator
    from phase1_basic.custatevec_backend import CuStateVecSimulator
    rows = []
    for n in range(22, 34):
        row = {"n": n}
        free = torch.cuda.mem_get_info()[0]
        if 3 * 2 ** n * 16 < 0.92 * free:
            row["torch_s"] = time_layers(lambda k: TorchSVSimulator(k, torch.device("cuda")), n)
        if 1.1 * 2 ** n * 16 < 0.95 * free:
            try:
                row["cusv_s"] = time_layers(lambda k: CuStateVecSimulator(k), n)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        if "cusv_s" in row:
            # 1q fused gate per qubit + n/2 CNOTs, each one read+write pass
            passes = n + n // 2
            row["cusv_GBps"] = round(passes * 2 * 2 ** n * 16 / row["cusv_s"] / 1e9, 1)
        if "torch_s" in row and "cusv_s" in row:
            row["speedup"] = round(row["torch_s"] / row["cusv_s"], 2)
        print(row, flush=True)
        if len(row) == 1:
            break
        rows.append(row)
    return rows


@section("decoders")
def _():
    import torch
    from phase2_noise.qldpc import gross_code, logical_operators, make_decoder
    hx, hz = gross_code()
    _, lz = logical_operators(hx, hz)
    p = 0.04
    rng = np.random.default_rng(7)
    N = 100000
    err = (rng.random((N, 144)) < p).astype(np.uint8)
    synd = (err.astype(np.int64) @ hz.T % 2).astype(np.uint8)

    def fails(corr, k):
        r = err[:k] ^ corr
        assert not np.any(r.astype(np.int64) @ hz.T % 2)
        return int(np.any(r.astype(np.int64) @ lz.T % 2, axis=1).sum())

    res = {"p": p}
    g = make_decoder(hz, p, backend="torch-cuda")
    g.decode_batch(synd[:256])                                   # warm-up
    for B in (1000, 10000, 100000):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        corr = g.decode_batch(synd[:B])
        dt = time.perf_counter() - t0
        res[f"torch_cuda_B{B}"] = {"us_per_shot": dt / B * 1e6,
                                   "failures": fails(corr, B), "shots": B}
        print(res[f"torch_cuda_B{B}"], flush=True)
    lat = []
    for s in synd[:300]:
        t0 = time.perf_counter(); g.decode_batch(s[None]); lat.append(time.perf_counter() - t0)
    res["torch_cuda_latency_median_us"] = float(np.median(lat)) * 1e6

    ref = make_decoder(hz, p, backend="ldpc")
    B = 10000
    t0 = time.perf_counter(); corr = ref.decode_batch(synd[:B]); dt = time.perf_counter() - t0
    res["ldpc_cpu"] = {"us_per_shot": dt / B * 1e6, "failures": fails(corr, B), "shots": B}

    try:
        import cudaq_qec as qec
        kw = dict(error_rate_vec=np.full(144, p), use_osd=True, osd_method=0,
                  max_iterations=100, use_sparsity=True, bp_batch_size=1000)
        try:
            d = qec.get_decoder("nv-qldpc-decoder", hz.astype(np.uint8),
                                bp_method=1, scale_factor=0.625, **kw)
            kw_note = "min-sum, scale 0.625"
        except Exception:
            d = qec.get_decoder("nv-qldpc-decoder", hz.astype(np.uint8), **kw)
            kw_note = "default BP"
        sy = [s.astype(np.float64).tolist() for s in synd[:B]]
        d.decode_batch(sy[:1000])
        t0 = time.perf_counter(); rs = d.decode_batch(sy); dt = time.perf_counter() - t0
        corr = np.array([(np.asarray(r.result) > 0.5) for r in rs], np.uint8)
        res["cudaq_qec_gpu"] = {"us_per_shot": dt / B * 1e6, "failures": fails(corr, B),
                                "shots": B, "settings": kw_note + ", bp_batch_size 1000"}
    except Exception as e:
        traceback.print_exc()
        res["cudaq_qec_gpu"] = {"error": repr(e)[:300]}
    return res


REPORT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print("E5b done ->", os.path.join(OUT, "e5b_report.json"))
