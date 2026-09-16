from __future__ import annotations

from flash_rt.piinfra.schema import HierarchyNode, ProfileDB
from flash_rt.piinfra.tax import optimization_opportunities


def format_flashrt_demo(result: dict) -> str:
    db = result["db"]
    spec = result["spec"]
    stages = result["stages"]
    lines = [
        "FlashRT Pi0.5 Demo / Profile",
        "═" * 40,
        "",
        spec.hierarchy_text().rstrip(),
        "",
        f"Latency e2e p50           {stages['e2e'].p50_ms:7.2f} ms",
        f"  vision.siglip            {stages['siglip'].p50_ms:7.2f} ms",
        f"  enc_ae (PaliGemma+AE)    {stages['enc_ae'].p50_ms:7.2f} ms",
        "",
        f"prompt: {db.meta.get('prompt', '')}",
        f"frame_idx: {db.meta.get('frame_idx')}",
        f"action[0]: {db.meta.get('actions0')}",
        "",
        "Hierarchy is from ckpt (FlashRT CUDA graphs), not ONNX/TRT.",
        "",
    ]
    return "\n".join(lines)


def format_analyze(db: ProfileDB, *, top_k: int = 15) -> str:
    lines: list[str] = []
    lines.append("Pi0.5 Performance Analysis")
    lines.append("═" * 40)
    lines.append("")
    e2e = db.total_ms
    lines.append(f"Latency (NVTX infer avg) {e2e:7.2f} ms")
    meta = db.meta or {}
    if meta.get("kernel_sum_ms"):
        lines.append(f"Kernel sum (whole capture){meta['kernel_sum_ms']:7.2f} ms")
    if meta.get("vision_ms") is not None:
        lines.append("")
        lines.append("[Stage / NVTX]")
        lines.append(f"  vision.siglip            {meta.get('vision_ms', 0):7.2f} ms")
        lines.append(f"  enc_ae                   {meta.get('enc_ae_ms', 0):7.2f} ms")
        lines.append(f"    paligemma              {meta.get('paligemma_ms', 0):7.2f} ms")
        lines.append(f"    action_expert          {meta.get('action_expert_ms', 0):7.2f} ms")

    lines.append("")
    lines.append("[Module]")
    if db.hierarchy:
        for m in db.hierarchy.children:
            if m.time_ms <= 0:
                continue
            flag = " 🔴" if m.pct >= 40 else ""
            lines.append(f"  {m.path:<24} {m.time_ms:7.2f} ms  {m.pct:5.1f}%{flag}")
            for b in m.children[:8]:
                if b.time_ms <= 0:
                    continue
                ncu = " 🎯" if "ncu" in (b.flags or []) else ""
                lines.append(f"    {b.path:<22} {b.time_ms:7.2f} ms  {b.pct:5.1f}%{ncu}")
            if len(m.children) > 8:
                lines.append(f"    ... +{len(m.children) - 8} more blocks")

    lines.append("")
    lines.append("[Tax]")
    for t in db.taxes:
        if t.time_ms < 0.01:
            continue
        sev = f"  {t.severity}" if t.severity else ""
        lines.append(f"  {t.name:<28} {t.time_ms:7.2f} ms  {t.pct:5.1f}%{sev}")

    lines.append("")
    lines.append("[Top Kernel]")
    for i, k in enumerate(db.kernels[:top_k], 1):
        lines.append(
            f"  {i:>2}. {k.name[:52]:<52} {k.time_ms:7.2f} ms  {k.pct:5.1f}%  [{k.category}]"
        )

    opps = optimization_opportunities(db.taxes, db.total_ms)
    if opps and opps[0]["items"]:
        lines.append("")
        lines.append("[Recommendation]")
        for it in opps[0]["items"]:
            lines.append(f"  {it['severity']} {it['name']}  ~{it['save_ms']:.2f} ms")
            for r in it["recommendations"][:3]:
                lines.append(f"     → {r}")
        lines.append(
            f"\n  Potential: ~{opps[0]['potential_ms']:.2f} ms / {db.total_ms:.2f} ms"
            f"  ({opps[0]['potential_pct']:.0f}% optimization space)"
        )
    lines.append("")
    return "\n".join(lines)


def format_drilldown(db: ProfileDB, query: str, *, top: int = 20) -> str:
    q = query.lower()
    lines = [f"Drilldown: {query}", "─" * 40, ""]
    kernels = [
        k
        for k in db.kernels
        if k.time_ms >= 0.01
        and (q in k.category.lower() or q in k.name.lower() or q in (k.model_path or "").lower())
    ]
    layers = [
        L
        for L in db.trt_layers
        if q in L.name.lower() or q in (L.model_path or "").lower() or q in (L.layer_type or "").lower()
    ]
    corrs = [
        c
        for c in db.correlations
        if q in c.kernel.lower() or q in c.model_node.lower() or q in c.trt_layer.lower()
    ]

    if kernels:
        lines.append("[Kernels]")
        for k in kernels[:top]:
            ncu = ""
            if k.ncu:
                ncu = (
                    f"  SM={k.ncu.get('sm_util', '?')}%"
                    f" DRAM={k.ncu.get('dram_util', '?')}%"
                    f" TC={k.ncu.get('tensorcore_util', '?')}%"
                )
            lines.append(
                f"  {k.time_ms:7.2f} ms  x{k.instances:<6}  {k.name[:60]}  ({k.model_path}){ncu}"
            )
        lines.append("")

    if layers:
        lines.append("[TRT Layers]")
        for L in sorted(layers, key=lambda x: x.time_ms, reverse=True)[:top]:
            lines.append(
                f"  {L.time_ms:7.2f} ms  {L.pct:5.1f}%  {L.layer_type:<8}  {L.name[:70]}"
            )
        lines.append("")

    if corrs:
        lines.append("[Correlation]")
        for c in corrs[:top]:
            lines.append(f"  {c.model_node}")
            lines.append(f"    TRT    {c.trt_layer or '-'}")
            lines.append(f"    Kernel {c.kernel}")
            lines.append(f"    {c.duration_us:.1f} us")
        lines.append("")

    tax = [t for t in db.taxes if q in t.name.lower()]
    for t in tax:
        lines.append(f"[Diagnosis] {t.name} {t.severity}")
        lines.append(f"  {t.diagnosis}")
        lines.append(f"  count={t.count}  time={t.time_ms:.2f} ms  ({t.pct:.1f}%)")
        for r in t.recommendations:
            lines.append(f"  → {r}")
        lines.append("")
    return "\n".join(lines)


def format_onnx_summary(index: dict) -> str:
    lines = ["Pi0.5 ONNX Graph Index", "═" * 40, ""]
    lines.append(f"Total nodes:          {index['nodes']:,}")
    lines.append(f"Initializers:         {index.get('initializers', 0):,}")
    lines.append("")
    lines.append("[Ops]")
    for op, n in list(index.get("ops", {}).items())[:20]:
        lines.append(f"  {op:<28} {n:>7,}")
    lines.append("")
    lines.append("[Module op counts]")
    for mod, ops in index.get("module_ops", {}).items():
        total = sum(ops.values())
        lines.append(f"  {mod:<20} {total:>7,}")
    lines.append("")
    lines.append("[Hierarchy (top paths)]")
    for path, n in list(index.get("hierarchy_counts", {}).items())[:25]:
        lines.append(f"  {path:<48} {n:>6,}")
    lines.append("")
    return "\n".join(lines)


def _fmt_tree(node: HierarchyNode, depth: int, max_depth: int, per_level: int) -> list[str]:
    if depth > max_depth:
        return []
    pad = "  " * depth
    lines = [f"{pad}{node.path:<40} {node.time_ms:7.2f} ms  {node.pct:5.1f}%"]
    for c in node.children[:per_level]:
        lines.extend(_fmt_tree(c, depth + 1, max_depth, per_level))
    if len(node.children) > per_level:
        lines.append(f"{pad}  ... +{len(node.children) - per_level} more")
    return lines
