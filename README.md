# SPARK — 混合架构 LLM 的敏感度分档量化引擎

> **S 系 · 推理引擎** —— 面向混合注意力架构（DeltaNet/线性注意力 + 全注意力）的极低比特量化 + QAT 全链路。

SPARK 为端侧/边缘设备设计，核心是一个**发现**：混合注意力架构（如 Qwen3.5）中 DeltaNet 投影层（尤其 `out_proj`）对量化的敏感度远超 lm_head——SSM 式状态传播逐 token 累积量化噪声。一刀切的低比特方案在这些层上崩溃，**按敏感度分档**是唯一出路。

## 量化分档（核心设计）

| 档位 | 模块 | 格式 | 密度 |
|---|---|---|---|
| **NVFP4** | attn/DeltaNet 投影 + embeddings | E2M1 + FP8 块 scale（对齐 NVIDIA 规范） | 4.5 bit |
| **SPFP2** | MLP (gate/up/down) | 三元 {0,±1} + 4-bit 共享指数，v2 双块位打包 | 2.25 bit |
| **INT8** | lm_head（untied 模型，导出时 PTQ） | per-channel 对称 | 8 bit |
| **BF16** | A_log / dt_bias 等状态参数 | — | 16 bit |

**Qwen3.5-4B 实测分派**：NVFP4 154 层 2.58B (53.2%) / SPFP2 96 层 2.26B (46.8%) → 综合 **3.56 bit/权重**。

## 实测结果（Qwen3.5-4B, WikiText-2）

| 格式 | PPL | 退化 | 磁盘 | decode tok/s | 显存 |
|---|---|---|---|---|---|
| BF16 | 8.88 | 基准 | 8.4 GB | 39.5 | 7.9 GB |
| NVFP4 (PTQ, torchao) | 9.57 | ×1.08 | ~2.4 GB | n/a* | — |
| NF4 (bitsandbytes) | 9.61 | ×1.08 | ~2.4 GB | 32.5 | 3.2 GB |
| **SPARK v2 (QAT 8k steps)** | **14.58** | **×1.64** | **1.95 GB** | 40.0 | 7.9 GB† |

\* torchao NVFP4Tensor 的 GQA repeat_kv dispatch 未实现。
† 当前推理解码为 BF16 物化（v3 kernel 将改为码流驻留 + 在线反量化，预期显存 ~2GB / decode ×3-4）。

**QAT 恢复曲线**（force_fake_quant 量化语义前向，12 块口径）：

```
PTQ 起点: PPL ~26 → step 500: 25.8 → 2k: 18.6 → 4k: 15.3 → 7k: 12.7 → 8k: 12.5
```

训练配置：C4:Magpie 6:4 样本级混流（Bresenham）、长度分桶（padded/active 1.08）、fake-quant 缓存、DeepSpeed ZeRO-2 bf16、cosine 7e-5→1e-6，双 RTX PRO 6000 约 4.6 小时。

## 技术要点

- **SPFP2 v2 双块位打包**：2 块 72bit = 9 字节（2.25 bit/权重，较 v1 -10%）；窗口式指数搜索（blockmax ±2，与全候选等价）。
- **v2 压缩容器**（`spark_v2_container.py`）：NVFP4 原生码流 + SPFP2 zstd（实测 -22%）+ INT8 head + fp32 参数；加载解压为标准格式，roundtrip **bit 级一致**。
- **原生码流 checkpoint**：NVFP4 层直接存 E2M1+FP8 码（训练 ckpt 5.7GB→~2GB，pack 幂等）。
- **QAT 训练器**：STE 直通（零区梯度不设 mask）、loss 归一化加权、续训（`--resume-ckpt` + `--warmup-steps`）、采样配比 CLI 化、单文件语料模式。
- **评测工具链**（loglikelihood 口径，与 lm-eval-harness 一致）：
  - `eval_matrix.py`：**三维对比矩阵**（PPL / decode 速度差分法 / 峰值显存），bf16/spark/nvfp4/nf4/int8 同台；
  - `eval_ppl.py` / `eval_pmmeval.py`（4 任务 × 10 语言）/ `eval_qat_ckpt.py` / `ptq_eval.py`；
  - 训练内 PPL 曲线（`--ppl-data`，每 500 步自动）。

## 快速开始

```bash
# 1) QAT 训练（Qwen3.5-4B，双卡，~5h）
./04_scripts/run_qat_4b.sh

# 2) 续训（换语料/分布，resume + warmup）
STEPS=3000 CKPT=saves/spark-qat-4b/spark-qat-7000.pt ./04_scripts/run_qat_4b_finetune.sh

# 3) 打包 .spark v2 引擎（bit 级 roundtrip 验证内置）
python3 04_scripts/pack_spark.py --ckpt saves/spark-qat-4b/spark-qat-final.pt \
    --model /path/to/Qwen3.5-4B --out engine.spark

# 4) 对比矩阵
python3 04_scripts/eval_matrix.py --model /path/to/Qwen3.5-4B \
    --spark-ckpt engine.spark --formats bf16,spark,nvfp4,nf4
```

## 目录结构

```
SPARK/
├── 01_core/          CUDA kernel + emu（block_fp2 / nvfp4 / int8）
├── 02_model/         量化层 / v2 容器 / loader / 分档替换
├── 03_training/      QAT 训练器（混流 dataloader / PPL 曲线 / 续训）
├── 04_scripts/       run_qat{,_2b,_4b,_7b,_9b,_finetune}.sh · pack_spark · eval_matrix / eval_ppl / eval_pmmeval 等
├── 05_tests/         roundtrip / QAT 冒烟 / 解码 bit-exact
└── docs/             发布物料（reddit_launch.md）
```

## 路线图

- [x] 混合分档量化（NVFP4/SPFP2/INT8/BF16 四档）+ QAT 全链路
- [x] v2 压缩容器（原生码流 + zstd，roundtrip bit 级一致）
- [x] 三维评测矩阵 + Qwen3.5-4B 实测
- [ ] 9B 档位训练与验证
- [ ] **v3：在线反量化 kernel + 2:4 结构化稀疏**（SPFP2 层天然 55% 非零率，距 2:4 阈值仅 5 点）——目标显存 ~2GB、decode ×3-4
- [ ] 各档位精度-体积-速度完整对比报告

## License

Apache License 2.0 · Copyright (c) 2025 DJAzzs
