from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

# sculptor cam → Pi0.5 image_keys (ckpt order)
CAMERA_SLOTS = {
    "cam3": "base_0_rgb",
    "cam0": "base_1_rgb",
    "cam1": "base_2_rgb",
    "cam9": "left_wrist_0_rgb",
    "cam10": "right_wrist_0_rgb",
}


def _task_text(calib: Path) -> str:
    tasks = pq.read_table(calib / "meta/tasks.parquet")
    if "task" in tasks.column_names:
        return str(tasks.column("task")[0].as_py())
    if "__index_level_0__" in tasks.column_names:
        return str(tasks.column("__index_level_0__")[0].as_py())
    return "perform the demonstrated sculptor task"


def _index_episodes(calib: Path, cameras: tuple[str, ...]) -> list[dict]:
    eps = pq.read_table(calib / "meta/episodes/chunk-000/file-000.parquet")
    meta = []
    for i in range(eps.num_rows):
        meta.append(
            {
                "from": int(eps.column("dataset_from_index")[i].as_py()),
                "to": int(eps.column("dataset_to_index")[i].as_py()),
                "files": {
                    cam: (
                        int(eps.column(f"videos/observation.images.{cam}/file_index")[i].as_py()),
                        float(eps.column(f"videos/observation.images.{cam}/from_timestamp")[i].as_py()),
                    )
                    for cam in cameras
                },
            }
        )
    return meta


def _decode_frame(calib: Path, cam: str, file_index: int, from_ts: float, local_frame: int, fps: float = 30.0) -> np.ndarray:
    path = calib / f"videos/observation.images.{cam}/chunk-000/file-{file_index:03d}.mp4"
    file_frame = int(round(from_ts * fps)) + local_frame
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, file_frame)
    ok, bgr = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(file_frame + 1):
            ok, bgr = cap.read()
            if not ok:
                break
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"failed to read {path} frame {file_frame}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != 224 or rgb.shape[1] != 224:
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _normalize_state(raw: np.ndarray, norm_stats: dict | None, out_dim: int = 32) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32).reshape(-1)
    out = np.zeros(out_dim, dtype=np.float32)
    n = min(len(x), out_dim)
    if norm_stats and "state" in norm_stats:
        st = norm_stats["state"]
        mean = np.asarray(st.get("mean") or st.get("q01"), dtype=np.float32).reshape(-1)
        std = np.asarray(st.get("std") or st.get("q99"), dtype=np.float32).reshape(-1)
        m = min(n, len(mean), len(std))
        # if quantile stats, approximate mid / half-range
        if "mean" not in st and "q01" in st and "q99" in st:
            q01 = np.asarray(st["q01"], dtype=np.float32).reshape(-1)
            q99 = np.asarray(st["q99"], dtype=np.float32).reshape(-1)
            m = min(n, len(q01), len(q99))
            mid = 0.5 * (q01[:m] + q99[:m])
            half = np.maximum(0.5 * (q99[:m] - q01[:m]), 1e-6)
            out[:m] = (x[:m] - mid) / half
            if n > m:
                out[m:n] = x[m:n]
        else:
            out[:m] = (x[:m] - mean[:m]) / np.maximum(std[:m], 1e-6)
            if n > m:
                out[m:n] = x[m:n]
    else:
        out[:n] = x[:n]
    return out


def load_norm_stats_dict(ckpt: Path) -> dict | None:
    hits = list((ckpt / "assets").rglob("norm_stats.json")) if (ckpt / "assets").is_dir() else []
    if (ckpt / "norm_stats.json").is_file():
        hits.append(ckpt / "norm_stats.json")
    if not hits:
        return None
    raw = json.loads(hits[0].read_text())
    return raw.get("norm_stats") or raw


def load_calibration_samples(
    calib_dir: str | Path,
    *,
    image_keys: list[str] | None = None,
    num_samples: int | None = None,
    ckpt_dir: str | Path | None = None,
    state_dim: int = 32,
) -> list[dict]:
    """Load FlashRT-ready samples from export calibration/ (videos + parquet)."""
    calib = Path(calib_dir)
    report = {}
    rp = calib / "calib_report.json"
    if rp.is_file():
        report = json.loads(rp.read_text())

    keys = list(image_keys or [
        "base_0_rgb", "base_1_rgb", "base_2_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
    ])
    inv = {v: k for k, v in CAMERA_SLOTS.items()}
    cameras = tuple(inv[k] for k in keys)

    states = np.stack(
        [
            np.asarray(x, dtype=np.float32)
            for x in pq.read_table(calib / "data/chunk-000/file-000.parquet")
            .column("observation.state")
            .to_pylist()
        ]
    )
    meta = _index_episodes(calib, cameras)
    fps = float(report.get("fps") or 30.0)
    task = str(report.get("task") or _task_text(calib))
    # Packed calibration videos are contiguous 0..N-1 (N == len(states)).
    # calib_report.sample_indices are source-dataset frame ids — do not seek by them.
    n = len(states)
    local_indices = list(range(n))
    if num_samples is not None:
        local_indices = local_indices[: int(num_samples)]

    norm = load_norm_stats_dict(Path(ckpt_dir)) if ckpt_dir else None
    src_ids = list(report.get("sample_indices") or local_indices)

    out = []
    for local_i in local_indices:
        # packed bundle: single episode [0, N), timestamp 0 → video frame == local_i
        ep = meta[0] if meta else {
            "from": 0,
            "to": n,
            "files": {cam: (0, 0.0) for cam in cameras},
        }
        images = {}
        for cam, key in zip(cameras, keys):
            fi, ts0 = ep["files"][cam]
            # prefer direct frame index into packed mp4
            images[key] = _decode_frame(calib, cam, fi, 0.0, local_i, fps=fps)
        state = _normalize_state(states[local_i], norm, out_dim=state_dim)
        out.append(
            {
                "frame_idx": int(src_ids[local_i]) if local_i < len(src_ids) else local_i,
                "local_idx": local_i,
                "prompt": task,
                "state": state,
                "images": [images[k] for k in keys],
                "image_dict": images,
            }
        )
    return out
