from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Pi05ExportSpec:
    """Deployed Pi0.5 variant — always load from export ckpt, not stock 3-cam defaults."""

    export_name: str = ""
    export_root: str = ""
    pi05: bool = True
    num_views: int = 5
    image_keys: list[str] = field(default_factory=list)
    action_dim: int = 32
    action_horizon: int = 15
    loss_action_dim: int = 10
    max_token_len: int = 200
    num_steps: int = 10
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    paligemma_layers: int = 18
    action_expert_layers: int = 18
    vision_layers: int = 27
    precision: str = "FP8"
    discrete_state_input: bool = True
    # ONNX packed images: [B, num_views*3, 224, 224]
    image_packed_channels: int = 15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hierarchy_text(self) -> str:
        cams = "\n".join(f"│   │   ├── {k}" for k in self.image_keys) or "│   │   ├── <image_keys>"
        return f"""{self.export_name or 'pi05'}
│
├── Vision Encoder (SigLIP) × {self.num_views} cams  [{self.vision_layers} blocks]
│   ├── images packed [B, {self.image_packed_channels}, 224, 224] = {self.num_views}×RGB
{cams}
│   └── shared vision_tower / multi_modal_projector
│
├── PaliGemma ({self.paligemma_variant})  [{self.paligemma_layers} blocks]
│   ├── lang tokens (max_token_len={self.max_token_len}) + discrete state
│   └── Block 0 … {self.paligemma_layers - 1}
│
└── Action Expert ({self.action_expert_variant})  [{self.action_expert_layers} blocks × {self.num_steps} denoise steps]
    ├── action_horizon={self.action_horizon}, action_dim={self.action_dim}
    ├── Step 1 … {self.num_steps}
    │     └── Block 0 … {self.action_expert_layers - 1}
    │           ├── attn (qkv / qk / av / o)
    │           ├── mlp (gateup / down)
    │           └── norm
    └── action_out_proj
"""


def load_export_spec(export_or_ckpt: str | Path) -> Pi05ExportSpec:
    root = Path(export_or_ckpt)
    if (root / "ckpt" / "config.json").is_file():
        export_root = root
        ckpt = root / "ckpt"
    elif (root / "config.json").is_file():
        ckpt = root
        export_root = root.parent if root.name == "ckpt" else root
    else:
        raise FileNotFoundError(f"no ckpt/config.json under {root}")

    cfg = json.loads((ckpt / "config.json").read_text())
    image_keys = list(cfg.get("image_keys") or [])
    num_views = len(image_keys) if image_keys else int(cfg.get("num_views") or 5)

    resolved: dict[str, Any] = {}
    ry = ckpt / "resolved.yaml"
    if ry.is_file():
        try:
            import yaml

            resolved = yaml.safe_load(ry.read_text()) or {}
        except Exception:
            resolved = {}
    model = ((resolved.get("train") or {}).get("model") or {}) if resolved else {}

    manifest: dict[str, Any] = {}
    for mp in (export_root / "manifest.yaml", ckpt / "manifest.yaml"):
        if mp.is_file():
            try:
                import yaml

                manifest = yaml.safe_load(mp.read_text()) or {}
            except Exception:
                manifest = {}
            break

    num_steps = int((manifest.get("export") or {}).get("num_steps") or 10)
    precision = str((manifest.get("quantization") or {}).get("precision") or cfg.get("precision") or "FP8")

    return Pi05ExportSpec(
        export_name=str(manifest.get("export_name") or export_root.name),
        export_root=str(export_root),
        pi05=bool(cfg.get("pi05", True)),
        num_views=num_views,
        image_keys=image_keys,
        action_dim=int(cfg.get("action_dim") or model.get("action_dim") or 32),
        action_horizon=int(cfg.get("action_horizon") or model.get("action_horizon") or 15),
        loss_action_dim=int(model.get("loss_action_dim") or 10),
        max_token_len=int(model.get("max_token_len") or 200),
        num_steps=num_steps,
        paligemma_variant=str(cfg.get("paligemma_variant") or model.get("paligemma_variant") or "gemma_2b"),
        action_expert_variant=str(
            cfg.get("action_expert_variant") or model.get("action_expert_variant") or "gemma_300m"
        ),
        precision=precision,
        discrete_state_input=bool(model.get("discrete_state_input", True)),
        image_packed_channels=num_views * 3,
    )


def default_sculptor_0911() -> Pi05ExportSpec:
    return load_export_spec(
        "/home/nvidia/workspace/pace_sculptor/artifacts/exports/sculptor_0911"
    )
