from __future__ import annotations

from collections import defaultdict

from flash_rt.piinfra.hierarchy import module_of
from flash_rt.piinfra.schema import (
    CorrelationRecord,
    HierarchyNode,
    KernelLaunch,
    ProfileDB,
    TrtLayer,
)


def correlate(
    trt_layers: list[TrtLayer],
    kernels: list[KernelLaunch],
) -> list[CorrelationRecord]:
    by_path: dict[str, list[TrtLayer]] = defaultdict(list)
    for L in trt_layers:
        by_path[L.model_path or "other"].append(L)

    recs: list[CorrelationRecord] = []
    for k in kernels:
        path = k.model_path or "other"
        cands = by_path.get(path) or []
        trt_name = ""
        if cands:
            # Prefer same category / longest common name fragment
            scored = []
            kn = k.name.lower()
            for L in cands:
                score = 0
                ln = L.name.lower()
                if L.layer_type and L.layer_type.lower() in kn:
                    score += 5
                for tok in ("matmul", "cast", "softmax", "mha", "gemm", "conv"):
                    if tok in kn and tok in ln:
                        score += 3
                score += _overlap(ln, kn)
                scored.append((score, L))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] > 0:
                trt_name = scored[0][1].name
                k.trt_layer = trt_name
        recs.append(
            CorrelationRecord(
                model_node=path,
                trt_layer=trt_name,
                kernel=k.name,
                duration_us=k.avg_us if k.avg_us else k.time_ms * 1000.0,
                nsys={"time_ms": k.time_ms, "pct": k.pct, "instances": k.instances},
                ncu=dict(k.ncu),
            )
        )
    return recs


def build_hierarchy(
    trt_layers: list[TrtLayer],
    kernels: list[KernelLaunch],
    *,
    total_ms: float | None = None,
) -> HierarchyNode:
    # Prefer timed TRT layers; fall back to kernels.
    timed = [L for L in trt_layers if L.time_ms > 0]
    use_kernels = not timed
    items: list[tuple[str, float, str, str]] = []
    if use_kernels:
        for k in kernels:
            items.append((k.model_path or "other", k.time_ms, "kernel", k.name))
        total = total_ms or sum(k.time_ms for k in kernels)
    else:
        for L in timed:
            items.append((L.model_path or "other", L.time_ms, "trt", L.name))
        total = total_ms or sum(L.time_ms for L in timed)

    root = HierarchyNode(path="pi05", kind="model", time_ms=total, pct=100.0)
    modules: dict[str, HierarchyNode] = {}
    blocks: dict[str, HierarchyNode] = {}

    agg: dict[str, float] = defaultdict(float)
    members: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, ms, src, name in items:
        agg[path] += ms
        members[path].append((src, name))

    for path, ms in sorted(agg.items(), key=lambda x: -x[1]):
        parts = path.split(".")
        mod = module_of(path)
        if mod not in modules:
            modules[mod] = HierarchyNode(path=mod, kind="module")
            root.children.append(modules[mod])
        mnode = modules[mod]

        block_key = ".".join(parts[:2]) if len(parts) >= 2 else mod
        if block_key not in blocks:
            blocks[block_key] = HierarchyNode(path=block_key, kind="block")
            mnode.children.append(blocks[block_key])
        bnode = blocks[block_key]

        leaf = HierarchyNode(
            path=path,
            kind="op",
            time_ms=ms,
            pct=(100.0 * ms / total) if total else 0.0,
        )
        for src, name in members[path]:
            if src == "trt":
                leaf.trt_layers.append(name)
            else:
                leaf.kernels.append(name)
        if any("cast" in (x.lower()) for x in leaf.trt_layers + leaf.kernels) or ".attn." in path and "cast" in path:
            pass
        bnode.children.append(leaf)
        bnode.time_ms += ms
        mnode.time_ms += ms

    for n in modules.values():
        n.pct = (100.0 * n.time_ms / total) if total else 0.0
        n.children.sort(key=lambda c: c.time_ms, reverse=True)
        for b in n.children:
            b.pct = (100.0 * b.time_ms / total) if total else 0.0
            b.children.sort(key=lambda c: c.time_ms, reverse=True)
    root.children.sort(key=lambda c: c.time_ms, reverse=True)
    return root


def fill_db(
    db: ProfileDB,
    *,
    trt_layers: list[TrtLayer] | None = None,
    kernels: list[KernelLaunch] | None = None,
) -> ProfileDB:
    if trt_layers is not None:
        db.trt_layers = trt_layers
    if kernels is not None:
        db.kernels = kernels
    db.total_ms = max(
        sum(L.time_ms for L in db.trt_layers),
        sum(k.time_ms for k in db.kernels),
    )
    db.correlations = correlate(db.trt_layers, db.kernels)
    db.hierarchy = build_hierarchy(db.trt_layers, db.kernels, total_ms=db.total_ms or None)
    return db


def _overlap(a: str, b: str) -> int:
    toks_a = set(x for x in a.replace("/", "_").split("_") if len(x) > 2)
    toks_b = set(x for x in b.replace("/", "_").split("_") if len(x) > 2)
    return len(toks_a & toks_b)
