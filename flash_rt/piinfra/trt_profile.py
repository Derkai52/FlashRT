from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from flash_rt.piinfra.hierarchy import classify_kernel, model_path_from_name, normalize_name
from flash_rt.piinfra.schema import TrtLayer


def list_engine_layers(engine_path: str | Path) -> list[str]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        eng = runtime.deserialize_cuda_engine(f.read())
    insp = eng.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    return list(info.get("Layers") or [])


def layers_from_engine(engine_path: str | Path) -> list[TrtLayer]:
    out = []
    for name in list_engine_layers(engine_path):
        out.append(
            TrtLayer(
                name=name,
                layer_type=_infer_trt_type(name),
                model_path=model_path_from_name(name),
            )
        )
    return out


def parse_trtexec_profile(path: str | Path) -> list[TrtLayer]:
    p = Path(path)
    text = p.read_text(errors="ignore")
    if p.suffix.lower() == ".json" or text.lstrip().startswith(("{", "[")):
        return _parse_profile_json(text)
    return _parse_profile_text(text)


def _parse_profile_json(text: str) -> list[TrtLayer]:
    data = json.loads(text)
    rows = data if isinstance(data, list) else data.get("layers") or data.get("Layers") or data.get("profile") or []
    out: list[TrtLayer] = []
    total = 0.0
    parsed: list[tuple[str, float, str]] = []
    for r in rows:
        if isinstance(r, str):
            continue
        name = r.get("name") or r.get("Name") or r.get("layer") or ""
        t = r.get("timeMs") or r.get("time_ms") or r.get("averageMs") or r.get("avgMs") or r.get("time") or 0.0
        typ = r.get("type") or r.get("Type") or r.get("layerType") or ""
        try:
            t = float(t)
        except Exception:
            t = 0.0
        parsed.append((name, t, typ))
        total += t
    for name, t, typ in parsed:
        out.append(
            TrtLayer(
                name=name,
                time_ms=t,
                pct=(100.0 * t / total) if total > 0 else 0.0,
                layer_type=typ or _infer_trt_type(name),
                model_path=model_path_from_name(name),
            )
        )
    return out


def _parse_profile_text(text: str) -> list[TrtLayer]:
    out: list[TrtLayer] = []
    total = 0.0
    rows: list[tuple[str, float, float]] = []
    # Layer name, Time (ms), Avg. Time / % / etc
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "----" in line:
            continue
        m = re.match(
            r"^(?P<name>.+?)\s+(?P<ms>[\d.]+)\s+(?:ms\s+)?(?P<pct>[\d.]+)\s*%?",
            line,
        )
        if not m:
            parts = re.split(r"\s{2,}|\t", line)
            if len(parts) < 2:
                continue
            name = parts[0]
            nums = [float(x) for x in parts[1:] if re.fullmatch(r"[\d.]+", x)]
            if not nums:
                continue
            ms = nums[0]
            pct = nums[1] if len(nums) > 1 else 0.0
        else:
            name = m.group("name").strip()
            ms = float(m.group("ms"))
            pct = float(m.group("pct"))
        if name.lower() in ("layer", "name", "total"):
            continue
        rows.append((name, ms, pct))
        total += ms
    for name, ms, pct in rows:
        if pct <= 0 and total > 0:
            pct = 100.0 * ms / total
        out.append(
            TrtLayer(
                name=name,
                time_ms=ms,
                pct=pct,
                layer_type=_infer_trt_type(name),
                model_path=model_path_from_name(name),
            )
        )
    return out


def parse_profile_csv(path: str | Path) -> list[TrtLayer]:
    out: list[TrtLayer] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    total = 0.0
    parsed = []
    for r in rows:
        name = r.get("Name") or r.get("name") or r.get("Layer") or ""
        ms = r.get("Time (ms)") or r.get("timeMs") or r.get("time_ms") or r.get("Average Time [ms]") or "0"
        try:
            ms_f = float(ms)
        except Exception:
            ms_f = 0.0
        parsed.append((name, ms_f))
        total += ms_f
    for name, ms in parsed:
        out.append(
            TrtLayer(
                name=name,
                time_ms=ms,
                pct=(100.0 * ms / total) if total else 0.0,
                layer_type=_infer_trt_type(name),
                model_path=model_path_from_name(name),
            )
        )
    return out


def run_trtexec_profile(
    engine: str | Path,
    out_json: str | Path,
    *,
    trtexec: str = "trtexec",
    iterations: int = 50,
    extra: list[str] | None = None,
) -> Path:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtexec,
        f"--loadEngine={engine}",
        "--dumpProfile",
        f"--exportProfile={out_json}",
        f"--warmUp=200",
        f"--iterations={iterations}",
        "--useSpinWait",
    ]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True)
    return out_json


def _infer_trt_type(name: str) -> str:
    n = normalize_name(name).lower()
    cat = classify_kernel(name)
    if "matmul" in n or cat == "gemm":
        return "GEMM"
    if "softmax" in n:
        return "Softmax"
    if "cast" in n or "quant" in n:
        return "Cast"
    if "mha" in n or "fmha" in n:
        return "MHA"
    if cat == "layout":
        return "Shuffle"
    if "conv" in n:
        return "Conv"
    return cat.upper() if cat != "other" else "Other"
