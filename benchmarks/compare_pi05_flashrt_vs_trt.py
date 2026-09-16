#!/usr/bin/env python3
"""JAX vs TRT FP8 vs FlashRT FP8+FA4 — Sculptor pi0.5 on Thor (5-cam)."""
from __future__ import annotations

import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPORT = Path(
    os.environ.get(
        "EXPORT_DIR",
        "/home/nvidia/workspace/pace_sculptor/artifacts/exports/sculptor_0911",
    )
)
CKPT = Path(os.environ.get("CKPT", EXPORT / "ckpt"))
VERIFY = Path(os.environ.get("VERIFY_WORK", EXPORT / "benchmark" / ".verify_work"))
ENGINE = Path(os.environ.get("ENGINE", EXPORT / "engine" / "model_fp8.engine"))
OUT = Path(os.environ.get("OUT", EXPORT / "benchmark" / "jax_trt_flashrt_fa4.json"))

WARMUP = int(os.environ.get("WARMUP", "30"))
REPEATS = int(os.environ.get("REPEATS", "20"))
NUM_ACC = int(os.environ.get("NUM_ACC", "32"))
NUM_VIEWS = int(os.environ.get("NUM_VIEWS", "5"))
USE_FA4 = os.environ.get("USE_FA4", "1") == "1"
os.environ.setdefault("FLASHRT_PI05_CHUNK_SIZE", "15")

IMAGE_KEYS = (
    "base_0_rgb",
    "base_1_rgb",
    "base_2_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


def _stats(ms):
    return {
        "mean_ms": float(statistics.mean(ms)),
        "std_ms": float(statistics.stdev(ms) if len(ms) > 1 else 0.0),
        "min_ms": float(min(ms)),
        "max_ms": float(max(ms)),
        "p50_ms": float(statistics.median(ms)),
        "n": len(ms),
    }


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _metrics(ref, pred):
    """Error vs ref. pct = max|Δact7| / mean(|ref act7|) * 100."""
    ref = np.asarray(ref, np.float32)
    pred = np.asarray(pred, np.float32)
    d = ref - pred
    max_abs_act7 = float(np.max(np.abs(d[..., :7])))
    ref_act7_mean_abs = float(np.mean(np.abs(ref[..., :7])))
    return {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "cosine": _cos(ref, pred),
        "mae_act7": float(np.mean(np.abs(d[..., :7]))),
        "max_abs": float(np.max(np.abs(d))),
        "max_abs_act7": max_abs_act7,
        "ref_act7_mean_abs": ref_act7_mean_abs,
        "max_abs_act7_pct": (100.0 * max_abs_act7 / ref_act7_mean_abs) if ref_act7_mean_abs > 0 else 0.0,
        "all_finite": bool(np.isfinite(pred).all()),
    }


def _tok_valid(tok, mask):
    m = np.asarray(mask).astype(bool).ravel()
    t = np.asarray(tok, np.int64).ravel()
    return t[m]


def _obs_from_sample(s):
    imgs = [np.asarray(s["image"][k]) for k in IMAGE_KEYS[:NUM_VIEWS]]
    if len(imgs) != NUM_VIEWS:
        raise ValueError(f"need {NUM_VIEWS} images, got {len(imgs)}")
    return {"images": imgs}


def bench_trt(feed, warmup, repeats):
    import tensorrt as trt

    dtype = np.float16
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(ENGINE.read_bytes())
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    np_feed = {
        "images": np.ascontiguousarray(feed["images"], dtype=dtype),
        "img_masks": np.ascontiguousarray(feed["img_masks"].astype(np.bool_)),
        "lang_tokens": np.ascontiguousarray(feed["lang_tokens"], dtype=np.int64),
        "lang_masks": np.ascontiguousarray(feed["lang_masks"].astype(np.bool_)),
        "state": np.ascontiguousarray(feed["state"], dtype=dtype),
        "noise": np.ascontiguousarray(feed["noise"], dtype=dtype),
    }
    gpu = {}
    with torch.cuda.stream(stream):
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                arr = np_feed[name]
                context.set_input_shape(name, tuple(arr.shape))
                ten = torch.from_numpy(arr).to(device="cuda", non_blocking=True)
                gpu[name] = ten
                context.set_tensor_address(name, int(ten.data_ptr()))
            else:
                shape = tuple(context.get_tensor_shape(name))
                dt = torch.float16 if precision_is_fp16(engine, name) else torch.float32
                ten = torch.empty(shape, dtype=dt, device="cuda")
                gpu[name] = ten
                context.set_tensor_address(name, int(ten.data_ptr()))
    stream.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    gpu_ms, e2e_ms = [], []

    def once():
        t0 = time.perf_counter()
        start_ev.record(stream)
        ok = context.execute_async_v3(int(stream.cuda_stream))
        if not ok:
            raise RuntimeError("TRT execute failed")
        end_ev.record(stream)
        stream.synchronize()
        _ = gpu["actions"][0, 0, 0].item()
        e2e_ms.append((time.perf_counter() - t0) * 1000)
        gpu_ms.append(start_ev.elapsed_time(end_ev))

    for _ in range(warmup):
        once()
    gpu_ms.clear()
    e2e_ms.clear()
    for _ in range(repeats):
        once()
    return {"gpu": _stats(gpu_ms), "e2e": _stats(e2e_ms)}


def precision_is_fp16(engine, name):
    import tensorrt as trt

    return trt.nptype(engine.get_tensor_dtype(name)) == np.float16


def flashrt_infer_with_noise(pipe, obs, noise_sa):
    noise = np.asarray(noise_sa, np.float16).reshape(pipe.Sa, 32)
    orig = np.random.randn

    def _fixed(*shape):
        if len(shape) == 2 and shape[0] == pipe.Sa and shape[1] == 32:
            return noise.copy()
        return orig(*shape)

    np.random.randn = _fixed
    try:
        pipe.infer(obs)
    finally:
        np.random.randn = orig
    return pipe._g_noise.float().cpu().numpy()


def main():
    samples = pickle.loads((VERIFY / "samples.pkl").read_bytes())
    feeds = pickle.loads((VERIFY / "onnx_feeds.pkl").read_bytes())
    act_pt = np.load(VERIFY / "actions_pytorch.npy")
    act_trt_saved = np.load(VERIFY / "actions_onnx_fp8.npy")
    act_jax = np.load(VERIFY / "actions_jax.npy")

    feed0 = feeds[0]
    assert feed0["images"].shape[1] == NUM_VIEWS * 3, (
        f"expected images C={NUM_VIEWS*3}, got {feed0['images'].shape}"
    )
    assert feed0["img_masks"].shape[-1] == NUM_VIEWS
    n_acc = min(NUM_ACC, len(act_jax), len(samples), len(act_pt), len(act_trt_saved))

    print("=== load FlashRT FP8 + FA4 ===", flush=True)
    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
    from flash_rt.hardware.thor import fa4_backend

    print(f"FA4 requested={USE_FA4} status={fa4_backend.status()}", flush=True)
    pipe = Pi05TorchFrontendThor(
        str(CKPT), num_views=NUM_VIEWS, autotune=0, use_fa4=USE_FA4
    )
    assert pipe.Sa == 15, f"expected Sa=15, got {pipe.Sa}"
    print(f"FlashRT Sa={pipe.Sa} num_views={pipe.num_views} use_fa4={pipe.use_fa4}", flush=True)

    s0 = samples[0]
    toks0 = _tok_valid(s0["tokenized_prompt"], s0["tokenized_prompt_mask"])
    pipe.set_prompt(toks0.tolist())

    obs0 = _obs_from_sample(s0)
    print("=== calibrate FlashRT ===", flush=True)
    cal_obs = [_obs_from_sample(samples[i]) for i in range(min(8, len(samples)))]
    pipe.calibrate(cal_obs, percentile=99.9)

    print("=== FlashRT accuracy ===", flush=True)
    frt_actions = []
    per = []
    for i in range(n_acc):
        s = samples[i]
        toks = _tok_valid(s["tokenized_prompt"], s["tokenized_prompt_mask"])
        pipe.set_prompt(toks.tolist())
        noise = np.asarray(s["noise"][0], np.float16).reshape(15, 32)
        out = flashrt_infer_with_noise(pipe, _obs_from_sample(s), noise)[:15]
        frt_actions.append(out)
        m = _metrics(act_jax[i], out)
        per.append(m)
        print(
            f"  sample {i}: max|Δact7|/mean|jax|={m['max_abs_act7_pct']:.4f}%  "
            f"(abs={m['max_abs_act7']:.4e}, scale={m['ref_act7_mean_abs']:.4e})",
            flush=True,
        )
    frt_actions = np.stack(frt_actions, 0)
    jax_slice = act_jax[:n_acc]
    pt_slice = act_pt[:n_acc]
    trt_slice = act_trt_saved[:n_acc]
    acc = {
        "flashrt_fa4_vs_jax": _metrics(jax_slice, frt_actions),
        "trt_fp8_vs_jax": _metrics(jax_slice, trt_slice),
        "pytorch_vs_jax": _metrics(jax_slice, pt_slice),
        "flashrt_fa4_vs_pytorch": _metrics(pt_slice, frt_actions),
        "trt_fp8_vs_pytorch": _metrics(pt_slice, trt_slice),
        "flashrt_fa4_vs_trt": _metrics(trt_slice, frt_actions),
        "n": n_acc,
        "num_steps": 10,
        "action_horizon": 15,
        "flashrt_chunk_size": int(pipe.Sa),
        "num_views": NUM_VIEWS,
        "use_fa4": bool(pipe.use_fa4),
        "fa4_status": fa4_backend.status(),
        "image_keys": list(IMAGE_KEYS[:NUM_VIEWS]),
        "jax_actions_source": str(VERIFY / "actions_jax.npy"),
        "jax_backend": "cpu",
        "per_sample_vs_jax": per,
    }
    if per:
        pcts = sorted(float(x["max_abs_act7_pct"]) for x in per)
        def _pct(xs, p):
            k = (len(xs) - 1) * p / 100.0
            f = int(k)
            c = min(f + 1, len(xs) - 1)
            return xs[f] + (xs[c] - xs[f]) * (k - f)
        acc["flashrt_fa4_vs_jax_per_sample_pct"] = {
            "median": _pct(pcts, 50),
            "p95": _pct(pcts, 95),
            "max": pcts[-1],
            "mean": float(statistics.mean(pcts)),
            "n": len(pcts),
        }
    print(json.dumps({k: v for k, v in acc.items() if k != "per_sample_vs_jax"}, indent=2), flush=True)

    skip_lat = os.environ.get("SKIP_LAT", "0") == "1"
    jax_lat_note = (
        "JAX latency not measured on Thor (no CUDA 12). "
        "Accuracy reference regenerated with JAX_PLATFORMS=cpu (32 frames)."
    )
    if skip_lat:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(
            json.dumps({"accuracy": acc, "latency": {"jax": {"skipped": jax_lat_note}}}, indent=2),
            encoding="utf-8",
        )
        print("wrote", OUT, flush=True)
        return

    print("=== FlashRT latency ===", flush=True)
    pipe.set_prompt(toks0.tolist())
    for _ in range(WARMUP):
        pipe.infer(obs0)
    frt_ms = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        pipe.infer(obs0)
        torch.cuda.synchronize()
        frt_ms.append((time.perf_counter() - t0) * 1000)
    frt_lat = _stats(frt_ms)
    print("FlashRT FA4", frt_lat, flush=True)

    del pipe
    torch.cuda.empty_cache()

    print("=== TRT FP8 latency ===", flush=True)
    trt = bench_trt(feed0, WARMUP, REPEATS)
    print("TRT", trt["gpu"], trt["e2e"], flush=True)

    result = {
        "device": "NVIDIA Thor",
        "checkpoint": str(CKPT),
        "engine": str(ENGINE),
        "config": {
            "num_views": NUM_VIEWS,
            "action_horizon": 15,
            "num_steps": 10,
            "warmup": WARMUP,
            "repeats": REPEATS,
            "num_acc": n_acc,
            "use_fa4": USE_FA4,
            "image_keys": list(IMAGE_KEYS[:NUM_VIEWS]),
        },
        "accuracy": acc,
        "latency": {
            "jax": {"skipped": True, "reason": jax_lat_note},
            "flashrt_fa4_e2e": frt_lat,
            "trt_fp8_gpu": trt["gpu"],
            "trt_fp8_e2e": trt["e2e"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)

    frt_p50 = frt_lat["p50_ms"]
    trt_p50 = trt["gpu"]["p50_ms"]
    print("\n===== SUMMARY =====")
    print(f"FlashRT FP8+FA4 P50: {frt_p50:.2f} ms ({1000/frt_p50:.1f} Hz)")
    print(f"TRT FP8 P50:         {trt_p50:.2f} ms ({1000/trt_p50:.1f} Hz)")
    print(f"speedup FlashRT/TRT: {trt_p50/frt_p50:.2f}x")
    print(f"JAX latency:         SKIPPED ({jax_lat_note})")
    print("Accuracy = max|Δact7| / mean(|ref act7|) × 100%")
    for name, key in (
        ("FlashRT FP8+FA4 vs JAX", "flashrt_fa4_vs_jax"),
        ("TRT FP8 vs JAX", "trt_fp8_vs_jax"),
        ("PyTorch vs JAX", "pytorch_vs_jax"),
        ("FlashRT vs PyTorch", "flashrt_fa4_vs_pytorch"),
        ("TRT vs PyTorch", "trt_fp8_vs_pytorch"),
    ):
        m = acc[key]
        print(
            f"  {name:<24} {m['max_abs_act7_pct']:7.4f}%  "
            f"(max={m['max_abs_act7']:.4e} / mean|ref|={m['ref_act7_mean_abs']:.4e})"
        )
    dist = acc.get("flashrt_fa4_vs_jax_per_sample_pct")
    if dist:
        print(
            f"  FlashRT per-sample vs JAX: "
            f"median={dist['median']:.4f}%  p95={dist['p95']:.4f}%  "
            f"max={dist['max']:.4f}%  n={dist['n']}"
        )


if __name__ == "__main__":
    main()
