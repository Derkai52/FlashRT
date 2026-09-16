from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class LayerStat:
    path: str
    module: str
    time_ms: float
    pct: float
    icon: str = ""
    ncu: bool = False
    note: str = ""


class CudaLayerTimer:
    """CUDA-event layer timer; pass via dims['layer_timer']."""

    def __init__(self):
        self._stack: list[tuple[str, torch.cuda.Event]] = []
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._nvtx = hasattr(torch.cuda, "nvtx")

    def begin(self, name: str) -> None:
        if self._nvtx:
            # No '/' — NCU treats A/B as nested NVTX stack, not a single name.
            torch.cuda.nvtx.range_push(f"layer.{name}")
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        self._stack.append((name, e0))

    def end(self, name: str) -> None:
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        if not self._stack:
            return
        n0, e0 = self._stack.pop()
        if n0 != name:
            # mismatched — still close
            name = n0
        e1.synchronize()
        self.samples[name].append(float(e0.elapsed_time(e1)))
        if self._nvtx:
            torch.cuda.nvtx.range_pop()

    def median_ms(self) -> dict[str, float]:
        return {k: float(statistics.median(v)) for k, v in self.samples.items() if v}


def _module_of(path: str) -> str:
    if path.startswith("vision"):
        return "vision"
    if path.startswith("paligemma"):
        return "paligemma"
    if path.startswith("action_expert"):
        return "action_expert"
    return "other"


def score_layers(med_ms: dict[str, float], *, top_ncu: int = 8) -> list[LayerStat]:
    total = sum(med_ms.values()) or 1.0
    by_mod: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for path, ms in med_ms.items():
        by_mod[_module_of(path)].append((path, ms))

    stats: list[LayerStat] = []
    for path, ms in med_ms.items():
        mod = _module_of(path)
        peers = [x[1] for x in by_mod[mod]]
        med = float(statistics.median(peers)) if peers else ms
        pct = 100.0 * ms / total
        if ms >= med * 1.5 or pct >= 5.0:
            icon, note = "🔴", "hot vs peers" if ms >= med * 1.5 else "≥5% total"
        elif ms >= med * 1.15:
            icon, note = "🟠", "above median"
        else:
            icon, note = "⚪", ""
        stats.append(LayerStat(path=path, module=mod, time_ms=ms, pct=pct, icon=icon, note=note))

    stats.sort(key=lambda s: s.time_ms, reverse=True)

    # Global top-3 always NCU
    for s in stats[:3]:
        s.ncu = True
        if s.icon == "⚪":
            s.icon = "🎯"
        if "NCU" not in s.note:
            s.note = (s.note + " · NCU").strip(" ·") if s.note else "top hotspot · NCU"

    # Per-module hottest (balanced NCU shortlist)
    per_mod = max(1, top_ncu // 3)
    picked = {s.path for s in stats if s.ncu}
    for mod, peers in by_mod.items():
        peers_sorted = sorted(peers, key=lambda x: -x[1])
        for path, _ in peers_sorted[:per_mod]:
            if path in picked:
                continue
            for s in stats:
                if s.path == path:
                    s.ncu = True
                    if s.icon == "⚪":
                        s.icon = "🎯"
                    if "NCU" not in s.note:
                        s.note = (s.note + " · NCU").strip(" ·") if s.note else f"{mod} hotspot · NCU"
                    picked.add(path)
                    break
        if len(picked) >= top_ncu:
            break

    # Fill remaining NCU slots from 🔴/🟠
    for s in stats:
        if len(picked) >= top_ncu:
            break
        if s.icon in ("🔴", "🟠") and s.path not in picked:
            s.ncu = True
            if "NCU" not in s.note:
                s.note = (s.note + " · NCU").strip(" ·")
            picked.add(s.path)
    return stats


def aggregate_ae_blocks(med_ms: dict[str, float]) -> dict[str, float]:
    """Sum action_expert.step*.block{i} → action_expert.block{i}."""
    out: dict[str, float] = {}
    for path, ms in med_ms.items():
        if path.startswith("action_expert.step") and ".block" in path:
            bi = path.split(".block", 1)[1]
            key = f"action_expert.block{bi}"
            out[key] = out.get(key, 0.0) + ms
        else:
            out[path] = out.get(path, 0.0) + ms
    return out


def format_layer_report(
    stats: list[LayerStat],
    *,
    module_filter: str | None = None,
    show_all: bool = False,
    bar_width: int = 20,
) -> str:
    rows = [s for s in stats if module_filter is None or s.module == module_filter]
    if not show_all:
        # always show 🔴/🟠/🎯 and top 15
        keep = []
        for s in rows:
            if s.icon in ("🔴", "🟠", "🎯") or s.ncu or len(keep) < 15:
                keep.append(s)
        # unique preserve order
        seen = set()
        rows = []
        for s in keep:
            if s.path in seen:
                continue
            seen.add(s.path)
            rows.append(s)

    total = sum(s.time_ms for s in stats) or 1.0
    max_ms = max((s.time_ms for s in rows), default=1.0)
    lines = [
        "FlashRT Per-Layer Latency",
        "═" * 56,
        f"{'':2} {'layer':<42} {'ms':>7} {'%':>6}  bar",
        "─" * 56,
    ]
    for s in rows:
        n = max(1, int(round(bar_width * s.time_ms / max_ms)))
        bar = "█" * n + "░" * (bar_width - n)
        tag = "NCU" if s.ncu else "   "
        lines.append(
            f"{s.icon} {tag} {s.path:<40} {s.time_ms:7.2f} {s.pct:5.1f}%  {bar}"
        )
        if s.note and s.ncu:
            lines.append(f"      └─ {s.note}")

    # module rollup
    mod_ms: dict[str, float] = defaultdict(float)
    for s in stats:
        mod_ms[s.module] += s.time_ms
    lines.append("")
    lines.append("[Module]")
    for m, ms in sorted(mod_ms.items(), key=lambda x: -x[1]):
        lines.append(f"  {m:<20} {ms:7.2f} ms  {100.0 * ms / total:5.1f}%")

    ncu_list = [s for s in stats if s.ncu][:12]
    lines.append("")
    lines.append("[Nsight Compute targets]  (run these first)")
    for i, s in enumerate(ncu_list, 1):
        lines.append(f"  {i:>2}. {s.icon} {s.path:<40} {s.time_ms:6.2f} ms")
    lines.append("")
    lines.append("Hint: piinfra ncu-layer <path> --app \"python -m flash_rt.piinfra layers ...\"")
    lines.append("")
    return "\n".join(lines)


def run_layer_profile(pipe, obs: dict, *, repeats: int = 5, warmup: int = 2) -> dict[str, Any]:
    """Eager (non-graph) per-layer timed forward on a warm Pi05Thor pipe."""
    import flash_rt.flash_rt_kernels as fvk
    from flash_rt.hardware.thor.shared_primitives import encoder_forward, siglip_forward
    from flash_rt.models.pi05.pipeline_thor import decoder_forward

    timer = CudaLayerTimer()
    nv = pipe.num_views

    # --- dicts mirroring capture ---
    sig_dims = dict(pipe._sig_dims)
    sig_dims["layer_timer"] = timer

    Se = pipe.Se
    total_keys = pipe.total_keys
    enc_bufs = {
        "x": pipe._enc_x.data_ptr(),
        "x_fp8": pipe._enc_x_fp8.data_ptr(),
        "qkv": pipe._enc_qkv_buf.data_ptr(),
        "logits": pipe._enc_logits.data_ptr(),
        "attn_out": pipe._enc_attn.data_ptr(),
        "o_fp8": pipe._enc_o_fp8.data_ptr(),
        "gate": pipe._enc_gate.data_ptr(),
        "hidden": pipe._enc_hidden.data_ptr(),
        "hid_fp8": pipe._enc_hid_fp8.data_ptr(),
        "fg": pipe._enc_fg.data_ptr(),
        "ctx": pipe._ctx,
        "x_norm": pipe._enc_attn.data_ptr(),
        "ones": (pipe._enc_ones_fp16.data_ptr() if pipe._enc_ones_fp16 is not None else 0),
    }
    enc_weights = {
        "qkv_w": [w.data_ptr() for w in pipe._enc_qkv_w],
        "o_w": [w.data_ptr() for w in pipe._enc_o_w],
        "gate_w": [w.data_ptr() for w in pipe._enc_gu_w],
        "down_w": [w.data_ptr() for w in pipe._enc_d_w],
        "rope": pipe._enc_rope.data_ptr(),
        "Kc": pipe._Kc.reshape(-1).data_ptr(),
        "Vc": pipe._Vc.reshape(-1).data_ptr(),
        "act_scales": pipe._enc_calib_scales.data_ptr(),
        "alpha_host": pipe._enc_alpha_host,
    }
    enc_dims = {
        "Se": Se,
        "D": pipe.De,
        "H": pipe.He,
        "NH": pipe.NHe,
        "HD": pipe.HDe,
        "L": pipe.Le,
        "total_keys": total_keys,
        "layer_timer": timer,
    }
    ae_bufs = {
        "noise": pipe._g_noise.data_ptr(),
        "x": pipe._ae_x.data_ptr(),
        "xn": pipe._ae_xn.data_ptr(),
        "gate": pipe._ae_gate.data_ptr(),
        "qkv": pipe._ae_qkv.data_ptr(),
        "logits": pipe._ae_logits.data_ptr(),
        "attn_out": pipe._ae_attn.data_ptr(),
        "hid": pipe._ae_hid.data_ptr(),
        "fg": pipe._ae_fg.data_ptr(),
        "action_f32": pipe._ae_action_f32.data_ptr(),
        "xn_fp8": pipe._ae_xn_fp8.data_ptr(),
        "hid_fp8": pipe._ae_hid_fp8.data_ptr(),
        "ctx_fp8": pipe._ae_ctx_fp8.data_ptr(),
    }
    ae_weights = {
        "ain_w": pipe._ain_w.data_ptr(),
        "ain_b": pipe._ain_b.data_ptr(),
        "sa": pipe._sa_all.data_ptr(),
        "qw": pipe._dec_qkv_flat.data_ptr(),
        "Kc": pipe._Kc.reshape(-1).data_ptr(),
        "Vc": pipe._Vc.reshape(-1).data_ptr(),
        "dec_devpos": pipe._attn.dec_devpos.data_ptr(),
        "ow": pipe._dec_o_flat.data_ptr(),
        "sf": pipe._sf_all.data_ptr(),
        "gw": pipe._dec_gu_flat.data_ptr(),
        "dw": pipe._dec_d_flat.data_ptr(),
        "aow": pipe._aow.data_ptr(),
        "aob": pipe._aob.data_ptr(),
        "aob_dt": pipe._aob_dt.data_ptr(),
        "dt": pipe._ae_dt,
        "fs": pipe._fs_all.data_ptr(),
        "rope": pipe._dec_rope.data_ptr(),
        "w_scales": pipe._ae_w_dev.data_ptr(),
        "act_scales": pipe._ae_calib_scales.data_ptr(),
    }
    ae_dims = {
        "S": pipe.Sa,
        "S_gemm": pipe.Sa_gemm,
        "D": pipe.Da,
        "H": pipe.Ha,
        "NH": 8,
        "HD": 256,
        "steps": 10,
        "layers": pipe.La,
        "enc_seq": Se,
        "total_keys": total_keys,
        "fixed_shape": pipe._fixed_shape_active,
        "layer_timer": timer,
    }

    def _one():
        img_list = obs["images"]
        for index, image in enumerate(img_list[:nv]):
            np.copyto(pipe._infer_images_u8_np[index], image)
        pipe._img_u8_buf.upload(pipe._infer_images_u8_np)
        pipe._patch_embed_ops(0, uint8_input=True)
        siglip_forward(
            pipe._gemm, fvk, pipe._sig_bufs, pipe._sig_weights,
            sig_dims, stream=0, attn=pipe._attn, use_fp8=pipe.use_fp8,
        )
        pipe._postln_project_ops(0)
        pipe._Kc.zero_()
        pipe._Vc.zero_()
        R_np = np.random.randn(pipe.Sa, 32).astype(np.float16)
        pipe._g_noise.view(-1, 32).copy_(torch.from_numpy(R_np).cuda())
        encoder_forward(
            pipe._gemm, fvk, enc_bufs, enc_weights, enc_dims,
            stream=0, attn=pipe._attn, use_fp8=pipe.use_fp8,
        )
        decoder_forward(
            pipe._ctx, fvk, ae_bufs, ae_weights, ae_dims,
            stream=0, attn=pipe._attn, use_fp8=pipe.use_fp8,
        )
        torch.cuda.synchronize()

    for _ in range(warmup):
        timer.samples.clear()
        _one()
    timer.samples.clear()
    for _ in range(repeats):
        _one()

    raw = timer.median_ms()
    agg = aggregate_ae_blocks(raw)
    # Prefer AE block aggregates in the main view; keep step detail in raw
    view = {k: v for k, v in agg.items()}
    stats = score_layers(view, top_ncu=8)
    return {"raw": raw, "agg": agg, "stats": stats, "total_ms": sum(view.values())}
