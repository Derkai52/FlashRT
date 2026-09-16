from __future__ import annotations

from flash_rt.piinfra.schema import KernelLaunch, TaxBucket, TrtLayer


def analyze_taxes(
    kernels: list[KernelLaunch],
    trt_layers: list[TrtLayer] | None = None,
    *,
    small_us: float = 50.0,
) -> list[TaxBucket]:
    total = sum(k.time_ms for k in kernels) or sum(L.time_ms for L in (trt_layers or []))
    if total <= 0:
        return []

    by_cat: dict[str, list[KernelLaunch]] = {}
    for k in kernels:
        by_cat.setdefault(k.category or "other", []).append(k)

    def bucket(name: str, ks: list[KernelLaunch], severity: str, diagnosis: str, recs: list[str]) -> TaxBucket:
        t = sum(k.time_ms for k in ks)
        return TaxBucket(
            name=name,
            time_ms=t,
            pct=(100.0 * t / total) if total else 0.0,
            count=sum(max(k.instances, 1) for k in ks),
            severity=severity,
            diagnosis=diagnosis,
            recommendations=recs,
        )

    taxes: list[TaxBucket] = []

    gemm = by_cat.get("gemm", [])
    taxes.append(
        bucket(
            "Compute Tax (GEMM)",
            gemm,
            "",
            "TensorCore / GEMM compute",
            [],
        )
    )

    cast = by_cat.get("cast", [])
    taxes.append(
        bucket(
            "Cast Tax",
            cast,
            "🔴" if cast and sum(k.time_ms for k in cast) / total > 0.08 else ("🟠" if cast else ""),
            "FP8↔FP16 / dtype boundary conversions",
            [
                "eliminate redundant Cast",
                "push Cast into producer/consumer",
                "enable TRT fusion",
                "inspect strongly-typed boundary",
            ],
        )
    )

    layout = by_cat.get("layout", [])
    taxes.append(
        bucket(
            "Layout Tax",
            layout,
            "🔴" if layout and sum(k.time_ms for k in layout) / total > 0.08 else ("🟠" if layout and sum(k.time_ms for k in layout) >= 0.01 else ""),
            "Transpose / Shuffle / l2tc format conversion",
            [
                "propagate preferred tensor format",
                "remove layout conversion chains",
                "fuse producer layout into GEMM",
            ],
        )
    )

    small = [k for k in kernels if (k.avg_us and k.avg_us < small_us) or (k.instances > 100 and k.time_ms / max(k.instances, 1) * 1000 < small_us)]
    taxes.append(
        bucket(
            "Launch Tax (small kernels)",
            small,
            "🟠" if small else "",
            f"High-frequency kernels (<{small_us:g} us) / launch overhead",
            [
                "fuse small elementwise ops",
                "reduce kernel count via TRT / graph capture",
            ],
        )
    )

    attn = by_cat.get("attention", []) + by_cat.get("softmax", [])
    # Fusion tax heuristic: attention category small relative to qkv/softmax fragments
    attn_ms = sum(k.time_ms for k in attn)
    qkv_like = [k for k in kernels if any(x in k.name.lower() for x in ("qk", "av", "softmax", "mha", "fmha"))]
    frag_ms = sum(k.time_ms for k in qkv_like)
    fusion_ks = qkv_like if attn_ms < 0.05 * total and frag_ms > 0 else attn
    taxes.append(
        bucket(
            "Attention Fusion Tax",
            fusion_ks,
            "🔴" if fusion_ks and attn_ms < 0.05 * total else "",
            "Unfused Q/K/V/Softmax path vs fused MHA",
            [
                "enable fused MHA / FA",
                "inspect Q/K/V/Softmax fusion",
            ],
        )
    )

    elem = by_cat.get("elementwise", [])
    taxes.append(
        bucket(
            "Elementwise Tax",
            elem,
            "🟠" if elem and sum(k.time_ms for k in elem) / total > 0.1 else "",
            "Standalone elementwise / activation / norm epilogues",
            ["fuse epilogues into GEMM", "merge residual chains"],
        )
    )

    other = by_cat.get("other", [])
    taxes.append(bucket("Other", other, "", "Uncategorized", []))

    taxes.sort(key=lambda t: t.time_ms, reverse=True)
    return taxes


def optimization_opportunities(taxes: list[TaxBucket], total_ms: float) -> list[dict]:
    # Non-overlapping priority: Cast > Layout > Attention Fusion > Launch > Elementwise
    priority = ("Cast Tax", "Layout Tax", "Attention Fusion Tax", "Launch Tax (small kernels)", "Elementwise Tax")
    by_name = {t.name: t for t in taxes}
    out = []
    for name in priority:
        t = by_name.get(name)
        if not t or not t.severity or t.time_ms < 0.01:
            continue
        out.append(
            {
                "name": t.name,
                "save_ms": t.time_ms,
                "pct": t.pct,
                "severity": t.severity,
                "recommendations": t.recommendations,
            }
        )
    potential = sum(x["save_ms"] for x in out)
    # Cap display potential at total (buckets overlap in reality)
    potential = min(potential, total_ms) if total_ms else potential
    return [
        {
            "items": out,
            "potential_ms": potential,
            "potential_pct": (100.0 * potential / total_ms) if total_ms else 0.0,
            "note": "bucket times overlap; potential capped at total latency",
        }
    ]
