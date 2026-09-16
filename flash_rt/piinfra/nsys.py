from __future__ import annotations

import csv
import io
import re
import subprocess
from pathlib import Path

from flash_rt.piinfra.hierarchy import classify_kernel, model_path_from_name
from flash_rt.piinfra.schema import KernelLaunch


def parse_nsys_kern_sum_csv(path: str | Path) -> list[KernelLaunch]:
    text = Path(path).read_text(errors="ignore")
    return parse_nsys_kern_sum_text(text)


def _find_csv_header(lines: list[str], *needles: str) -> int | None:
    for i, ln in enumerate(lines):
        if all(n in ln for n in needles):
            return i
        if ln.startswith("Time (%)") and "Name" in ln and "," in ln:
            return i
        if "Range" in ln and "Total Time" in ln and "," in ln:
            return i
    return None


def parse_nsys_kern_sum_text(text: str) -> list[KernelLaunch]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return []
    hdr_i = _find_csv_header(lines, "Total Time (ns)", "Name")
    if hdr_i is None:
        hdr_i = _find_csv_header(lines, "Name")
    if hdr_i is not None:
        return _from_csv(lines[hdr_i:])
    if "," in lines[0] and ("Time" in lines[0] or "Name" in lines[0] or "Kernel" in lines[0]):
        return _from_csv(lines)
    return _from_table(lines)


def parse_nsys_nvtx_sum_text(text: str) -> list[dict]:
    """Parse nsys nvtx_sum CSV → [{name, time_ms, instances, avg_ms}]."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    hdr_i = _find_csv_header(lines, "Range")
    if hdr_i is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[hdr_i:])))
    out = []
    for r in reader:
        name = (r.get("Range") or r.get("Name") or "").strip().lstrip(":")
        if not name:
            continue
        if r.get("Total Time (ns)") not in (None, ""):
            t_ms = _f(r.get("Total Time (ns)")) / 1e6
        else:
            t_ms = _f(r.get("Total Time") or 0)
        inst = int(float(r.get("Instances") or r.get("Count") or 1))
        if r.get("Avg (ns)") not in (None, ""):
            avg_ms = _f(r.get("Avg (ns)")) / 1e6
        else:
            avg_ms = t_ms / max(inst, 1)
        out.append({"name": name, "time_ms": t_ms, "instances": inst, "avg_ms": avg_ms})
    out.sort(key=lambda x: x["time_ms"], reverse=True)
    return out


def nsys_stats_report(rep: str | Path, report: str) -> str:
    rep = Path(rep)
    sqlite = rep.with_suffix(".sqlite") if rep.suffix == ".nsys-rep" else None
    if sqlite is not None and sqlite.is_file():
        src, force = sqlite, False
    else:
        src, force = rep, True
    cmd = ["nsys", "stats", "--report", report, "--format", "csv"]
    if force:
        cmd += ["--force-export", "true"]
    cmd.append(str(src))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout or ""


def nsys_stats_kern_sum(rep: str | Path, out_csv: str | Path | None = None) -> list[KernelLaunch]:
    rep = Path(rep)
    text = nsys_stats_report(rep, "cuda_gpu_kern_sum")
    if out_csv:
        Path(out_csv).write_text(text)
    kernels = parse_nsys_kern_sum_text(text)
    if not kernels and Path(rep).suffix == ".nsys-rep":
        text = nsys_stats_report(rep, "cuda_gpu_kern_sum")  # already tried
        # retry forcing nsys-rep path if sqlite was used
        cmd = [
            "nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
            "--force-export", "true", str(rep),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        text = proc.stdout or ""
        kernels = parse_nsys_kern_sum_text(text)
    if not kernels:
        raise RuntimeError(f"failed to parse cuda_gpu_kern_sum from {rep}")
    return kernels


def nsys_stats_nvtx_sum(rep: str | Path) -> list[dict]:
    text = nsys_stats_report(rep, "nvtx_sum")
    return parse_nsys_nvtx_sum_text(text)


def _from_csv(lines: list[str]) -> list[KernelLaunch]:
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = list(reader)
    parsed: list[tuple[str, float, int, float]] = []
    total = 0.0
    for r in rows:
        name = (
            r.get("Name")
            or r.get("Kernel Name")
            or r.get("Short Name")
            or r.get("name")
            or ""
        ).strip().strip('"')
        if not name:
            continue
        # Prefer explicit ns columns from nsys cuda_gpu_kern_sum
        if r.get("Total Time (ns)") is not None and r.get("Total Time (ns)") != "":
            t_ms = _f(r.get("Total Time (ns)")) / 1e6
        else:
            t = _f(r.get("Total Time") or r.get("Time") or 0)
            t_ms = t / 1e6 if t >= 1e4 else t
        inst = int(float(r.get("Instances") or r.get("Count") or r.get("instances") or 0))
        if r.get("Avg (ns)") is not None and r.get("Avg (ns)") != "":
            avg_us = _f(r.get("Avg (ns)")) / 1e3
        else:
            avg = _f(r.get("Average") or 0)
            avg_us = avg / 1e3 if avg >= 1e3 else avg
        parsed.append((name, t_ms, inst, avg_us))
        total += t_ms
    return _to_kernels(parsed, total)


def _from_table(lines: list[str]) -> list[KernelLaunch]:
    parsed: list[tuple[str, float, int, float]] = []
    total = 0.0
    for ln in lines:
        if re.match(r"^Time\b|^Name\b|^---", ln.strip()):
            continue
        parts = re.split(r"\s{2,}|\t", ln.strip())
        if len(parts) < 2:
            continue
        # common: Time(%) Total Time(ns) Instances Avg... Name
        nums = []
        name_parts = []
        for p in parts:
            if re.fullmatch(r"[\d.eE+-]+%?", p):
                nums.append(p.rstrip("%"))
            else:
                name_parts.append(p)
        if not name_parts or not nums:
            continue
        name = " ".join(name_parts)
        # Prefer total time column (usually 2nd numeric)
        t_raw = float(nums[1] if len(nums) > 1 else nums[0])
        t_ms = t_raw / 1e6 if t_raw > 1e5 else t_raw
        inst = int(float(nums[2])) if len(nums) > 2 else 0
        avg_us = float(nums[3]) / 1e3 if len(nums) > 3 and float(nums[3]) > 1e3 else (float(nums[3]) if len(nums) > 3 else 0.0)
        parsed.append((name, t_ms, inst, avg_us))
        total += t_ms
    return _to_kernels(parsed, total)


def _to_kernels(parsed: list[tuple[str, float, int, float]], total: float) -> list[KernelLaunch]:
    out = []
    for name, t_ms, inst, avg_us in parsed:
        out.append(
            KernelLaunch(
                name=name,
                time_ms=t_ms,
                pct=(100.0 * t_ms / total) if total else 0.0,
                instances=inst,
                avg_us=avg_us,
                category=classify_kernel(name),
                model_path=model_path_from_name(name),
            )
        )
    out.sort(key=lambda k: k.time_ms, reverse=True)
    return out


def run_nsys_profile(cmd: list[str], out_stem: str | Path, *, extra: list[str] | None = None) -> Path:
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    nsys_cmd = [
        "nsys",
        "profile",
        "-t",
        "cuda,nvtx,osrt",
        "-o",
        str(out_stem),
        "--force-overwrite",
        "true",
    ]
    if extra:
        nsys_cmd.extend(extra)
    nsys_cmd.extend(cmd)
    subprocess.run(nsys_cmd, check=True)
    rep = Path(str(out_stem) + ".nsys-rep")
    if not rep.exists():
        rep = Path(str(out_stem) + ".qdrep")
    return rep


def _f(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0
