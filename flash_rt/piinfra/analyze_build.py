from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flash_rt.piinfra.model_spec import Pi05ExportSpec
from flash_rt.piinfra.schema import HierarchyNode, KernelLaunch, ProfileDB
from flash_rt.piinfra.tax import analyze_taxes


def _nvtx_map(nvtx_rows: list[dict]) -> dict[str, dict]:
    out = {}
    for r in nvtx_rows:
        name = r["name"].lstrip(":")
        out[name] = r
        # also bare suffix
        if "/" in name:
            out[name.split("/")[-1]] = r
    return out


def load_layers_json(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    raw = json.loads(p.read_text())
    return list(raw.get("layers") or [])


def build_analyze_hierarchy(
    spec: Pi05ExportSpec,
    *,
    nvtx_rows: list[dict],
    layer_rows: list[dict],
    kernels: list[KernelLaunch],
) -> tuple[HierarchyNode, float, dict[str, Any]]:
    """Build timed hierarchy: NVTX stages + layers.json blocks. No zero-ms stubs."""
    nv = _nvtx_map(nvtx_rows)
    infer = nv.get("flashrt/infer") or nv.get("infer")
    siglip = nv.get("flashrt/vision.siglip") or nv.get("vision.siglip")
    enc_ae = nv.get("flashrt/enc_ae") or nv.get("enc_ae")

    # Prefer per-call average for "latency" display
    e2e_ms = float(infer["avg_ms"]) if infer else sum(k.time_ms for k in kernels)
    vision_ms = float(siglip["avg_ms"]) if siglip else 0.0
    enc_ae_ms = float(enc_ae["avg_ms"]) if enc_ae else 0.0

    # Split enc_ae into paligemma / AE using layers.json totals if available
    pal_ms = 0.0
    ae_ms = 0.0
    for row in layer_rows:
        mod = row.get("module") or ""
        ms = float(row.get("time_ms") or 0.0)
        if mod == "paligemma":
            pal_ms += ms
        elif mod == "action_expert":
            ae_ms += ms
        elif mod == "vision" and vision_ms <= 0:
            vision_ms += ms
    if enc_ae_ms > 0 and (pal_ms + ae_ms) > 0:
        scale = enc_ae_ms / (pal_ms + ae_ms)
        pal_ms *= scale
        ae_ms *= scale
    elif enc_ae_ms > 0:
        # heuristic without layers.json: ~60% paligemma / 40% AE from prior benches
        pal_ms = enc_ae_ms * 0.60
        ae_ms = enc_ae_ms * 0.40

    root = HierarchyNode(path=spec.export_name or "pi05", kind="model", time_ms=e2e_ms, pct=100.0)
    meta = {
        "e2e_ms": e2e_ms,
        "vision_ms": vision_ms,
        "paligemma_ms": pal_ms,
        "action_expert_ms": ae_ms,
        "enc_ae_ms": enc_ae_ms,
        "kernel_sum_ms": sum(k.time_ms for k in kernels),
        "nvtx": {k: {"avg_ms": v["avg_ms"], "instances": v["instances"]} for k, v in nv.items() if k.startswith("flashrt")},
    }

    def add_module(path: str, ms: float, children: list[HierarchyNode] | None = None):
        if ms <= 0 and not children:
            return
        node = HierarchyNode(
            path=path,
            kind="module",
            time_ms=ms,
            pct=(100.0 * ms / e2e_ms) if e2e_ms else 0.0,
            children=children or [],
        )
        root.children.append(node)

    # Block children from layers.json (only timed)
    by_mod_blocks: dict[str, list[HierarchyNode]] = {"vision": [], "paligemma": [], "action_expert": []}
    for row in sorted(layer_rows, key=lambda r: -float(r.get("time_ms") or 0)):
        ms = float(row.get("time_ms") or 0)
        if ms <= 0:
            continue
        mod = row.get("module") or "other"
        path = row.get("path") or ""
        if mod not in by_mod_blocks:
            continue
        # scale block times into module budget if we have NVTX module ms
        by_mod_blocks[mod].append(
            HierarchyNode(
                path=path,
                kind="block",
                time_ms=ms,
                pct=0.0,
                flags=["ncu"] if row.get("ncu") else [],
            )
        )

    # Rescale block children so they sum to module ms (keep relative ratios)
    def rescale(mod: str, mod_ms: float) -> list[HierarchyNode]:
        kids = by_mod_blocks.get(mod) or []
        if not kids or mod_ms <= 0:
            return kids[:12]  # top blocks only in tree
        s = sum(k.time_ms for k in kids) or 1.0
        out = []
        for k in kids[:12]:
            scaled = k.time_ms * (mod_ms / s)
            out.append(
                HierarchyNode(
                    path=k.path,
                    kind="block",
                    time_ms=scaled,
                    pct=(100.0 * scaled / e2e_ms) if e2e_ms else 0.0,
                    flags=k.flags,
                )
            )
        return out

    add_module("vision", vision_ms, rescale("vision", vision_ms))
    add_module("paligemma", pal_ms, rescale("paligemma", pal_ms))
    add_module("action_expert", ae_ms, rescale("action_expert", ae_ms))
    root.children.sort(key=lambda c: c.time_ms, reverse=True)
    return root, e2e_ms, meta


def assemble_analyze_db(
    *,
    spec: Pi05ExportSpec,
    kernels: list[KernelLaunch],
    nvtx_rows: list[dict],
    layer_rows: list[dict],
    source: str,
) -> ProfileDB:
    hier, e2e_ms, meta = build_analyze_hierarchy(
        spec, nvtx_rows=nvtx_rows, layer_rows=layer_rows, kernels=kernels
    )
    # Scale whole-capture kern_sum → per-infer so Tax/% match Latency
    from dataclasses import replace

    ksum = float(meta.get("kernel_sum_ms") or 0.0)
    scale = (e2e_ms / ksum) if ksum > 0 and e2e_ms > 0 else 1.0
    scaled = []
    for k in kernels:
        kk = replace(k, time_ms=k.time_ms * scale)
        kk.pct = (100.0 * kk.time_ms / e2e_ms) if e2e_ms else 0.0
        scaled.append(kk)
    meta["tax_scale"] = scale
    db = ProfileDB(
        source=source,
        total_ms=e2e_ms,
        kernels=sorted(scaled, key=lambda x: x.time_ms, reverse=True),
        hierarchy=hier,
        meta={"model_spec": spec.to_dict(), **meta},
    )
    db.taxes = analyze_taxes(db.kernels, [])
    return db
