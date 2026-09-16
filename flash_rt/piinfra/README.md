# piinfra

FlashRT-only profiler for **sculptor_0911**（5 路相机）。

从 `$EXPORT/ckpt` 建 hierarchy，对 FlashRT 做计时 / Tax / NCU 定位。  
**不读 ONNX，不读 TensorRT engine。**

```text
ckpt/config.json ──► Model Spec（5-cam）
calibration/     ──► demo / layers 输入
FlashRT .so + safetensors
        │
        ├── demo     CUDA Graph：siglip / enc_ae stage ms
        ├── layers   Eager：每 block ms + 🔴/🟠/🎯 → NCU 短名单
        └── analyze  必须吃 nsys（.nsys-rep / kern csv），不是 layers.json
```

## Setup

```bash
EXPORT=/home/nvidia/workspace/pace_sculptor/artifacts/exports/sculptor_0911
OUT=$EXPORT/benchmark/piinfra
mkdir -p "$OUT"

# $EXPORT/ckpt/model.safetensors   FlashRT 权重
# $EXPORT/calibration/             标定视频 + state + task
```

| field | sculptor_0911 |
|---|---|
| cameras | **5**：`base_0/1/2_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` |
| images | calib `256²` → demo resize `224²` |
| action_horizon / steps | 15 / 10 |
| Vision | SigLIP **27** blocks（0…26，共享） |
| PaliGemma | gemma_2b **18** blocks（0…17） |
| Action Expert | gemma_300m **18** blocks × **10** denoise steps |
| runtime | FlashRT FP8 + CUDA Graph（可选 FA4） |

```bash
python -m flash_rt.piinfra spec "$EXPORT"
```

## 别混：`layers` vs `analyze`

| | `layers` | `analyze` |
|---|---|---|
| 输入 | calibration + FlashRT eager forward | **nsys** `.nsys-rep` / kern_sum `.csv` |
| 输出 | 每 block 真实 ms + 图标 | Tax / Top CUDA kernel |
| 典型文件 | `flashrt_layers.json` | `piinfra_db.json` |
| 错用 | — | ❌ 把 `flashrt_layers.json` 当 `--nsys` → 全 0ms 骨架 |

看层耗时只用：

```bash
python -m flash_rt.piinfra layers --export "$EXPORT" --all
python -m flash_rt.piinfra layers --export "$EXPORT" --module vision --all   # 27 个 SigLIP block
```

## 推荐流程

```bash
# 1) 端到端（部署路径 / CUDA Graph）
python -m flash_rt.piinfra demo \
  --export "$EXPORT" --calib "$EXPORT/calibration" \
  --num-samples 4 --warmup 5 --repeats 20 \
  -o "$OUT/flashrt_demo.json"

# 2) 按层热点 → NCU 短名单
python -m flash_rt.piinfra layers \
  --export "$EXPORT" --calib "$EXPORT/calibration" \
  --repeats 5 --top-ncu 8 \
  -o "$OUT/flashrt_layers.json"

# 3) 对短名单层发 NCU
python -m flash_rt.piinfra ncu-layer paligemma.block8 --export "$EXPORT"

# 4) （可选）整图 nsys → Tax —— 必须先 profile 出 .nsys-rep
nsys profile -t cuda,nvtx -o "$OUT/flashrt" --force-overwrite true \
  python -m flash_rt.piinfra demo --export "$EXPORT" --repeats 10
python -m flash_rt.piinfra analyze --export "$EXPORT" \
  --nsys "$OUT/flashrt.nsys-rep" -o "$OUT/piinfra_db.json"
python -m flash_rt.piinfra drilldown cast --db "$OUT/piinfra_db.json"
```

## demo

```bash
python -m flash_rt.piinfra demo --export "$EXPORT" --calib "$EXPORT/calibration"
```

输出：`e2e` / `vision.siglip` / `enc_ae` p50 ms（Graph 路径）。

## layers（按层 + 图标 → NCU）

Eager + CUDA Event，按 Transformer **block** 计时。

```bash
python -m flash_rt.piinfra layers --export "$EXPORT" --top-ncu 8
python -m flash_rt.piinfra layers --export "$EXPORT" --module paligemma --all
python -m flash_rt.piinfra layers --export "$EXPORT" --module action_expert --all
python -m flash_rt.piinfra layers --export "$EXPORT" --module vision --all
python -m flash_rt.piinfra layers --export "$EXPORT" --detail --all   # AE: step×block
```

| 图标 | 含义 |
|---|---|
| 🔴 | 相对同模块偏热，或占比 ≥5% |
| 🟠 | 高于同模块中位数 |
| 🎯 + `NCU` | 建议优先 Nsight Compute（全局 Top + 每模块热点） |
| ⚪ | 正常（各层差不多时也会标相对 Top，不代表坏层） |

默认 AE：`step1…10.block{i}` **加总** → `action_expert.block{i}`；`--detail` 展开逐步。

默认列表只打热点；要看齐 **27 / 18 / 18** 层请加 `--all` 或 `--module … --all`。

NVTX：`layer.vision.block{i}`、`layer.paligemma.block{i}`、`layer.action_expert.step{s}.block{i}`。

```bash
# 写出 GUI 可打开的 .ncu-rep（NVTX push/pop 过滤，默认 --set detailed）
python -m flash_rt.piinfra ncu-layer paligemma.block12 --export "$EXPORT" --run
# 报告: $EXPORT/benchmark/piinfra/ncu_paligemma_block12.ncu-rep
# 需要更全指标: 加 --set full（很慢）
```

## nsys / Tax

```bash
# 正确：--nsys 指向 nsys 产物
python -m flash_rt.piinfra analyze --export "$EXPORT" \
  --nsys "$OUT/flashrt.nsys-rep" -o "$OUT/piinfra_db.json"

# 错误：不要把 layers/demo 的 json 传给 --nsys（会报错退出）
# python -m flash_rt.piinfra analyze --nsys "$OUT/flashrt_layers.json"   # ❌

python -m flash_rt.piinfra drilldown cast --db "$OUT/piinfra_db.json"
python -m flash_rt.piinfra ncu cast --db "$OUT/piinfra_db.json" --top 5 \
  --app "python -m flash_rt.piinfra demo --export $EXPORT --repeats 3"
python -m flash_rt.piinfra show --db "$OUT/piinfra_db.json"
```

NVTX（demo）：`flashrt/infer`、`flashrt/vision.siglip`、`flashrt/enc_ae`。

## Commands

| cmd | 作用 |
|---|---|
| `spec` | 从 ckpt 打印 5-cam hierarchy |
| `demo` | calibration → FlashRT infer + stage ms |
| `layers` | per-block ms + 🔴/🟠/🎯 + NCU 短名单 |
| `ncu-layer` | 针对某一 layer 生成 NCU 命令 |
| `analyze` | **nsys** → Tax / top kernels |
| `drilldown` | 下钻 cast/layout/gemm/module |
| `ncu` | 按 Tax bucket 生成/跑 NCU |
| `show` | 重打 analyze/db 报告 |

## Notes

- hierarchy / 相机数来自 `$EXPORT/ckpt/config.json`，不是默认 3-cam Pi0.5。
- `demo` / `layers` 读 `$EXPORT/calibration`（packed 32 帧）；勿用 report 里原始 dataset frame id seek。
- State 经 ckpt `norm_stats` 归一化后进 `set_prompt`。
- `layers` = eager（可见每层）；`demo` = CUDA Graph（更接近部署延迟）。
- 各 PaliGemma block ~2.3–2.5ms 很均匀时，🎯 只是相对排序，不代表单层异常。
