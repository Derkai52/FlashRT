from __future__ import annotations

import re
from typing import Iterable

# Pi0.5 ONNX/TRT name → module hierarchy.
# Prefix encode uses /layers.{i}/self_attn/... (no step suffix).
# Denoise unrolls as /layers.{i}/self_attn_{k}/... and *_proj_{k}.

_RE_VISION = re.compile(r"/vision_tower(?:/|$)")
_RE_LAYER = re.compile(r"/layers[._](\d+)(?:/|$)")
_RE_STEP = re.compile(
    r"(?:self_attn|mlp|input_layernorm|post_attention_layernorm|"
    r"q_proj|k_proj|v_proj|o_proj|qkv_proj|gate_proj|up_proj|"
    r"down_proj|gateup_proj)_(\d+)"
)
_RE_SUB = re.compile(
    r"(q_proj|k_proj|v_proj|o_proj|qkv_proj|qk_matmul|av_matmul|"
    r"gate_proj|up_proj|down_proj|gateup_proj|fc1|fc2|"
    r"input_layernorm|post_attention_layernorm|"
    r"self_attn|mlp|patch_embedding|embeddings)"
)


def normalize_name(name: str) -> str:
    n = name or ""
    for p in ("__myl_Repl_", "__myl_", "myl0_"):
        if n.startswith(p):
            n = n[len(p) :]
    n = n.split("+", 1)[0]
    n = re.sub(r"_myl0_\d+$", "", n)
    n = n.replace("/layers_", "/layers.")
    return n


def classify_kernel(name: str) -> str:
    n = (name or "").lower()
    if "mha" in n or "fmha" in n or "flash" in n or "attention" in n:
        return "attention"
    if any(x in n for x in ("matmul", "gemm", "mata", "mma", "cublas", "cutlass", "wmma", "hmma", "qkv_proj", "gateup", "nvjet")):
        return "gemm"
    if "softmax" in n:
        return "softmax"
    if "cast" in n or "quantize" in n or "dequant" in n or "qdq" in n:
        return "cast"
    if n.startswith("l2tc_") or "transpose" in n or "shuffle" in n:
        return "layout"
    if any(x in n for x in ("memcpy", "memset", "reorder", "reshape")):
        return "layout"
    if "tran" in n and "tanh" not in n:
        return "layout"
    if any(x in n for x in ("add", "mul", "div", "exp", "tanh", "gelu", "silu", "relu", "norm", "mean", "sqrt")):
        return "elementwise"
    if "move" in n or "conc" in n or "slic" in n or "resh" in n:
        return "layout"
    return "other"


def model_path_from_name(name: str) -> str:
    n = normalize_name(name)
    if not n:
        return "other"

    if _RE_VISION.search(n) or n.startswith("vision_tower"):
        m = _RE_LAYER.search(n)
        block = f"block{m.group(1)}" if m else "embed"
        sub = _subop(n)
        base = f"vision.{block}"
        return f"{base}.{sub}" if sub else base

    if n.startswith("/action_") or "action_in_proj" in n or "action_out_proj" in n:
        which = "in_proj" if "action_in" in n else "out_proj" if "action_out" in n else "action"
        return f"action_expert.{which}"

    if "multi_modal_projector" in n:
        return "paligemma.projector"

    m = _RE_LAYER.search(n)
    if m:
        bi = int(m.group(1))
        step = _step_index(n)
        sub = _subop(n)
        if step is None:
            base = f"paligemma.block{bi}"
        else:
            base = f"action_expert.step{step}.block{bi}"
        return f"{base}.{sub}" if sub else base

    cat = classify_kernel(n)
    if cat != "other":
        return f"runtime.{cat}"
    return "other"


def _step_index(n: str) -> int | None:
    hits = [int(x) for x in _RE_STEP.findall(n)]
    if not hits:
        if "qkv_proj" in n or "gateup_proj" in n:
            return 0
        return None
    return max(hits)


def _subop(n: str) -> str:
    m = _RE_SUB.search(n)
    if not m:
        return ""
    s = m.group(1)
    if s in ("q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj", "qk_matmul", "av_matmul", "self_attn"):
        return f"attn.{s}"
    if s in ("gate_proj", "up_proj", "down_proj", "gateup_proj", "fc1", "fc2", "mlp"):
        return f"mlp.{s}"
    if s in ("input_layernorm", "post_attention_layernorm"):
        return f"norm.{s}"
    if s in ("patch_embedding", "embeddings"):
        return s
    return s


def module_of(path: str) -> str:
    if path.startswith("vision"):
        return "vision"
    if path.startswith("paligemma"):
        return "paligemma"
    if path.startswith("action_expert"):
        return "action_expert"
    if path.startswith("runtime"):
        return "runtime"
    return "other"


def group_paths(paths: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in paths:
        out.setdefault(module_of(p), []).append(p)
    return out
