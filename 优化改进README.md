# Thor Pi0.5（sculptor_0911）优化改进实验记录

设备：NVIDIA Thor · 模型：Pi0.5 sculptor_0911（5 相机）· 路径：FlashRT FP8 + FA4 / TRT FP8 / JAX  
评测包：`pace_sculptor/artifacts/exports/sculptor_0911`

---

## 1. 性能剖析工具（piinfra）

新建 `flash_rt/piinfra/`：FlashRT-only 分层剖析（非 ONNX/Netron）。

| 命令 | 作用 |
|---|---|
| `demo` | CUDA Graph e2e / vision / enc_ae |
| `layers` | Eager 每 block ms + NCU 短名单 |
| `analyze` | nsys kern_sum + NVTX + layers → Tax |
| `ncu-layer` | 按 NVTX 出 `.ncu-rep`（GUI 可开） |

关键修复：

- `nsys stats` 在已有 `.sqlite` 时勿 `--force-export`（否则 NVTX 空 → 全 0ms）
- NVTX 名避免 `/`（NCU 当成嵌套）；push/pop 过滤用 `layer.xxx]`
- analyze 用 NVTX avg + layers 比例填 Module，不再骨架 0ms

产物示例：`benchmark/piinfra/piinfra_db.json`、`ncu_paligemma_block12.ncu-rep`

---

## 2. 端到端延迟结构（NVTX / demo）

典型 FlashRT Graph：**e2e ≈ 77–78 ms**

| 模块 | ≈ms | 占比 |
|---|---:|---:|
| PaliGemma | ~41 | ~52% |
| Action Expert | ~26 | ~33% |
| Vision SigLIP×5 | ~11 | ~14% |

enc_ae（PG+AE）≈ 66–67 ms，为绝对主体。

Tax（按 infer 归一后量级）：

- Launch / 碎 kernel：很大（与 GEMM 重叠，勿直接相加）
- GEMM Compute：~25 ms
- **Cast**：~14 ms（可优化）
- Attention Fusion：~2 ms（已开 FA4，空间小）
- Layout：≈0

---

## 3. NCU：`paligemma.block12`

- 报告：`benchmark/piinfra/ncu_paligemma_block12.ncu-rep`（`--set detailed`）
- 典型 CUTLASS FP8 GEMM（`MainloopSm100…F8F6F4`）：
  - Memory throughput ~**95%**
  - Compute throughput ~**55%**
  - Runtime improvement ≈ **0** → **已 memory-bound**
- 结论：再抠单条 GEMM 的 TC% 无收益；应减少 HBM 往返（融 cast / epilogue）

---

## 4. 三路对比：JAX / TRT FP8 / FlashRT FP8+FA4

脚本：`benchmarks/compare_pi05_flashrt_vs_trt.py`  
结果：`benchmark/jax_trt_flashrt_fa4.json` / `*_n32.json`

### 速度（P50，Thor）

| Backend | P50 | Hz |
|---|---:|---:|
| FlashRT FP8+FA4 | **~77 ms** | ~13 |
| TRT FP8 | ~92–93 ms | ~11 |
| JAX | **未测 GPU** | — |

FlashRT / TRT ≈ **1.19–1.20×**

说明：Thor 为 CUDA 13；`jax_cuda12_plugin` 要 `libcudart.so.12`，**不改 CUDA 环境**。精度参考用 `JAX_PLATFORMS=cpu` 重算 32 帧 `actions_jax.npy`。

### 精度口径

\[
\text{pct} = \frac{\max|\Delta \mathrm{act7}|}{\mathrm{mean}(|\mathrm{ref\,act7}|)} \times 100\%
\]

（相对 JAX act7 平均绝对幅度；不是 `max×100`。）

### n=8（早期）

| | max\|Δact7\|/mean\|jax\| |
|---|---:|
| PyTorch vs JAX | ~1.8% |
| FlashRT vs JAX | ~5.6% |
| TRT vs JAX | ~22% |

### n=32（CPU JAX 参考）

| | 全局 max% | 备注 |
|---|---:|---|
| PyTorch vs JAX | **~1.21%** | 正常 |
| TRT vs JAX | ~21.4% | |
| FlashRT vs JAX | 全局 max **~658%** | **sample27 爆炸**（abs≈2） |

FlashRT **逐样本** vs JAX：median **~10.4%**，p95 **~25.9%**，max ~560%（#27）。  
前 8 帧仍约 3–6%，与 n=8 一致；变差主要来自 **更多难帧 + #27 异常**，不是 JAX CPU 重算偏差（前 8 帧 CPU vs 旧 dump act7 max 差仅 ~2.7e-3）。

---

## 5. 模块级优化方向（结论）

| 优先级 | 方向 | 预期 |
|---|---|---|
| P0 | 减 Cast / quantize，融进 GEMM epilogue | 数 ms～十余 ms |
| P0 | 融碎 kernel（norm / residual / Abs） | 降 Launch |
| P1 | AE **减 denoise step**（10→8/5） | 近似线性，最大杠杆 |
| P1 | 修 FlashRT sample27 数值爆炸 | 精度可信前提 |
| P2 | Vision | 空间小 |
| — | 单条 FP8 GEMM 再抬 Compute% | ≈0（带宽墙） |

---

## 6. AE Decoder cross-attn 改 FA4（已做 A/B）

代码：`flash_rt/hardware/thor/attn_backend.py`  
- 支持 decoder 走 FA4 GQA（与 encoder 同布局）  
- 开关：`FLASHRT_FA4_DECODER=1`；**默认 0（仍用 `attention_qkv_fp16`）**

A/B（`benchmark/ae_decoder_fa4_ab.json`，Sa=15）：

| | P50 | sample0 | sample27 |
|---|---:|---:|---:|
| decoder fvk | **77.01 ms** | 2.93% | 560% |
| decoder FA4 | 79.43 ms | 2.93% | 560% |

Δ ≈ **−2.4 ms（FA4 更慢）**。原因：Q 太短（15），FA4 launch/拷贝开销大于收益。  
**结论：此路径不是加速点**；默认保持 fvk。

---

## 7. 产物索引

| 文件 | 内容 |
|---|---|
| `benchmark/piinfra/piinfra_db.json` | nsys Tax + Module |
| `benchmark/piinfra/ncu_paligemma_block12.ncu-rep` | block12 NCU |
| `benchmark/jax_trt_flashrt_fa4_n32.json` | 32 帧三路对比 |
| `benchmark/ae_decoder_fa4_ab.json` | AE decoder FA4 A/B |
| `benchmark/.verify_work/actions_jax.npy` | 32 帧 JAX CPU 参考 |

---

## 8. 下一步建议

1. 修 sample27（或同类）FlashRT 爆炸后再报精度  
2. 优先 **Cast/epilogue 融合** +（可选）**AE 减 step**  
3. 不再投入 AE decoder FA4；可用 `FLASHRT_FA4_DECODER=1` 做回归对照
