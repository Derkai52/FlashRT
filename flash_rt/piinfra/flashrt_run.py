from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flash_rt.piinfra.hierarchy import classify_kernel
from flash_rt.piinfra.model_spec import Pi05ExportSpec, load_export_spec
from flash_rt.piinfra.schema import HierarchyNode, KernelLaunch, ProfileDB, TaxBucket
from flash_rt.piinfra.tax import analyze_taxes


@dataclass
class StageTiming:
    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def p50_ms(self) -> float:
        return float(statistics.median(self.times_ms)) if self.times_ms else 0.0


def flashrt_hierarchy(spec: Pi05ExportSpec) -> HierarchyNode:
    root = HierarchyNode(path=spec.export_name or "pi05", kind="model", pct=100.0)
    vision = HierarchyNode(path="vision", kind="module")
    for k in spec.image_keys:
        vision.children.append(HierarchyNode(path=f"vision.cam.{k}", kind="input"))
    vision.children.append(
        HierarchyNode(path=f"vision.siglip.blocks0-{spec.vision_layers - 1}", kind="block")
    )
    vision.children.append(HierarchyNode(path="vision.projector", kind="op"))
    pal = HierarchyNode(path="paligemma", kind="module")
    for i in range(spec.paligemma_layers):
        b = HierarchyNode(path=f"paligemma.block{i}", kind="block")
        b.children = [
            HierarchyNode(path=f"paligemma.block{i}.attn", kind="op"),
            HierarchyNode(path=f"paligemma.block{i}.mlp", kind="op"),
        ]
        pal.children.append(b)
    ae = HierarchyNode(path="action_expert", kind="module")
    for s in range(1, spec.num_steps + 1):
        step = HierarchyNode(path=f"action_expert.step{s}", kind="block")
        for i in range(spec.action_expert_layers):
            b = HierarchyNode(path=f"action_expert.step{s}.block{i}", kind="block")
            b.children = [
                HierarchyNode(path=f"action_expert.step{s}.block{i}.attn", kind="op"),
                HierarchyNode(path=f"action_expert.step{s}.block{i}.mlp", kind="op"),
            ]
            step.children.append(b)
        ae.children.append(step)
    root.children = [vision, pal, ae]
    return root


def _nvtx_range(name: str):
    try:
        return torch.cuda.nvtx.range(name)
    except Exception:
        from contextlib import nullcontext

        return nullcontext()


def _cuda_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def build_pipe(ckpt: str | Path, *, num_views: int, use_fa4: bool = True, autotune: int = 0):
    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

    return Pi05TorchFrontendThor(
        str(ckpt),
        num_views=num_views,
        autotune=autotune,
        use_fa4=use_fa4,
    )


def run_demo(
    *,
    export_dir: str | Path,
    calib_dir: str | Path | None = None,
    num_samples: int = 4,
    warmup: int = 5,
    repeats: int = 20,
    use_fa4: bool = True,
    profile: bool = True,
) -> dict[str, Any]:
    export_dir = Path(export_dir)
    ckpt = export_dir / "ckpt"
    calib_dir = Path(calib_dir or export_dir / "calibration")
    spec = load_export_spec(export_dir)

    from flash_rt.piinfra.calib_io import load_calibration_samples

    samples = load_calibration_samples(
        calib_dir,
        image_keys=spec.image_keys,
        num_samples=num_samples,
        ckpt_dir=ckpt,
        state_dim=spec.action_dim,
    )
    if not samples:
        raise RuntimeError(f"no calibration samples in {calib_dir}")

    pipe = build_pipe(ckpt, num_views=spec.num_views, use_fa4=use_fa4)
    s0 = samples[0]
    pipe.set_prompt(s0["prompt"], state=s0["state"])
    cal_obs = [{"images": s["images"]} for s in samples[: min(8, len(samples))]]
    pipe.calibrate(cal_obs, percentile=99.9)

    obs0 = {"images": s0["images"]}
    stages = {
        "e2e": StageTiming("e2e"),
        "siglip": StageTiming("vision.siglip"),
        "enc_ae": StageTiming("paligemma+action_expert"),
    }

    # warmup
    for _ in range(warmup):
        with _nvtx_range("flashrt/infer"):
            pipe.infer(obs0)
    torch.cuda.synchronize()

    actions = None
    for i in range(repeats):
        with _nvtx_range("flashrt/infer"):
            if profile:
                # split siglip / enc_ae via graph replay hooks
                t_e2e0 = time.perf_counter()
                # upload images (same as infer path) then time graphs
                img_list = obs0["images"]
                for index, image in enumerate(img_list[: spec.num_views]):
                    np.copyto(pipe._infer_images_u8_np[index], image)
                pipe._img_u8_buf.upload(pipe._infer_images_u8_np)

                with _nvtx_range("flashrt/vision.siglip"):
                    stages["siglip"].times_ms.append(_cuda_ms(pipe._siglip_u8_graph.replay))

                if pipe.use_fp8 and not pipe._real_data_calibrated:
                    pipe._recalibrate_with_real_data()
                    pipe._real_data_calibrated = True

                R_np = np.random.randn(pipe.Sa, 32).astype(np.float16)
                R = torch.from_numpy(R_np).to("cuda", non_blocking=True)
                pipe._g_noise.view(-1, 32).copy_(R)

                with _nvtx_range("flashrt/enc_ae"):
                    stages["enc_ae"].times_ms.append(_cuda_ms(pipe._enc_ae_graph.replay))

                from flash_rt.core.utils.actions import LIBERO_ACTION_DIM, unnormalize_actions

                raw = pipe._g_noise.float().cpu().numpy()
                actions = unnormalize_actions(raw, pipe.norm_stats)[:, :LIBERO_ACTION_DIM]
                stages["e2e"].times_ms.append((time.perf_counter() - t_e2e0) * 1000.0)
            else:
                t0 = time.perf_counter()
                out = pipe.infer(obs0)
                torch.cuda.synchronize()
                stages["e2e"].times_ms.append((time.perf_counter() - t0) * 1000.0)
                actions = out["actions"]

    total = stages["e2e"].p50_ms
    hier = flashrt_hierarchy(spec)
    # attach measured module times
    for child in hier.children:
        if child.path == "vision":
            child.time_ms = stages["siglip"].p50_ms
        elif child.path in ("paligemma", "action_expert"):
            # enc_ae is fused in one CUDA graph — attribute jointly under action_expert parent note
            pass
    # put fused enc+ae on a synthetic child under root
    fused = HierarchyNode(
        path="enc_ae_graph",
        kind="module",
        time_ms=stages["enc_ae"].p50_ms,
        pct=(100.0 * stages["enc_ae"].p50_ms / total) if total else 0.0,
    )
    hier.time_ms = total
    for child in hier.children:
        if child.path == "vision":
            child.pct = (100.0 * child.time_ms / total) if total else 0.0
    hier.children.append(fused)

    db = ProfileDB(
        source=f"flashrt:{export_dir}",
        total_ms=total,
        hierarchy=hier,
        meta={
            "model_spec": spec.to_dict(),
            "stages": {k: {"p50_ms": v.p50_ms, "n": len(v.times_ms)} for k, v in stages.items()},
            "prompt": s0["prompt"],
            "frame_idx": s0["frame_idx"],
            "action_shape": list(np.asarray(actions).shape) if actions is not None else None,
            "actions0": np.asarray(actions[0]).tolist() if actions is not None else None,
        },
    )
    return {"db": db, "actions": actions, "spec": spec, "stages": stages}
