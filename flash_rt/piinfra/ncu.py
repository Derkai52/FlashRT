from __future__ import annotations

import csv
import shlex
import subprocess
from pathlib import Path
from typing import Any

from flash_rt.piinfra.schema import KernelLaunch


def build_ncu_command(
    kernel_regex: str,
    app_cmd: list[str],
    *,
    out: str = "ncu_out",
    set_name: str = "full",
    replay_mode: str = "kernel",
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        "ncu",
        "--kernel-name",
        f"regex:{kernel_regex}",
        "--set",
        set_name,
        "--replay-mode",
        replay_mode,
        "-o",
        out,
        "--force-overwrite",
    ]
    if extra:
        cmd.extend(extra)
    cmd.extend(app_cmd)
    return cmd


def format_ncu_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def parse_ncu_csv(path: str | Path) -> dict[str, dict[str, Any]]:
    """Map kernel name -> selected metrics."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, Any]] = {}
    metric_aliases = {
        "sm_util": ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm_utilization", "SM Utilization"),
        "dram_util": ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram_utilization", "DRAM Utilization"),
        "tensorcore_util": (
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "tensor_utilization",
            "Tensor Core Utilization",
        ),
        "occupancy": (
            "sm__maximum_warps_avg_per_active_cycle",
            "achieved_occupancy",
            "Achieved Occupancy",
        ),
        "duration_us": ("gpu__time_duration.sum", "Duration", "duration"),
    }
    for r in rows:
        kname = r.get("Kernel Name") or r.get("Name") or r.get("kernel") or ""
        metric = r.get("Metric Name") or r.get("Metric") or ""
        val = r.get("Metric Value") or r.get("Value") or r.get("Avg") or ""
        if not kname:
            continue
        slot = out.setdefault(kname, {})
        slot.setdefault("raw", {})[metric] = val
        for key, aliases in metric_aliases.items():
            if metric in aliases or any(a.lower() in metric.lower() for a in aliases if isinstance(a, str)):
                try:
                    slot[key] = float(val)
                except Exception:
                    slot[key] = val
    return out


def attach_ncu(kernels: list[KernelLaunch], ncu_metrics: dict[str, dict[str, Any]]) -> None:
    for k in kernels:
        for nk, metrics in ncu_metrics.items():
            if k.name in nk or nk in k.name:
                k.ncu = metrics
                break


def run_ncu(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def top_kernel_regexes(kernels: list[KernelLaunch], category: str | None = None, top: int = 5) -> list[str]:
    sel = [k for k in kernels if category is None or k.category == category or category in k.name.lower()]
    sel = sel[:top]
    regs = []
    for k in sel:
        # Escape lightly; keep readable regex stem
        stem = k.name.split("(")[0]
        stem = stem.replace("|", r"\|")
        regs.append(stem[:80])
    return regs
