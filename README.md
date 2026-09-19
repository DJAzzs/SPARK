# SPARK — 混合架构 LLM 的敏感度分档量化引擎

> **S 系 · 推理引擎** —— 面向混合注意力架构（DeltaNet/线性注意力 + 全注意力）的极低比特量化 + QAT 全链路。

SPARK 为端侧/边缘设备设计，核心是一个**发现**：混合注意力架构（如 Qwen3.5）中 DeltaNet 投影层（尤其 `out_proj`）对量化的敏感度远超 lm_head——SSM 式状态传播逐 token 累积量化噪声。一刀切的低比特方案在这些层上崩溃，**按敏感度分档**是唯一出路。

## 量化分档（v2.3 — MLP-only QAT + PTQ 导出）

**核心策略**（经 4B 多轮实验验证的最优解）：

| 层 | 训练时 | 导出时 | 密度 |
|---|---|---|---|
| MLP (gate/up/down) | **SPFP2 QAT**（STE + fake_quant） | SPFP2 packed (2.5 bit) | 2.5 bit |
| Attn/GDN 投影 | **BF16**（干净梯度，零量化噪声） | FP3 PTQ (3.5 bit) | 3.5 bit |
| out_proj + embed | BF16 | NVFP4 PTQ (4.5 bit) | 4.5 bit |
| A_log / dt_bias / norms | BF16 | BF16 | 16 bit |

**为什么 MLP-only 优于全量 QAT**：
1. Attention 层的量化噪声会干扰 MLP 的 QAT 恢复（梯度污染）
2. FP3/NVFP4 PTQ 本身近无损（×1.10），不需要 QAT 恢复
3. 训练更快（少 154 层 fake_quant 计算）
4. 绕过 FP3 save/load 的 roundtrip bug

**Sprint 训练策略**（2000 步短跑）：
- MLP-only QAT 在长跑（10000 步）中会因 lr 冲头导致 PPL 恶化
- 短跑 + 快速 cosine 衰减让 lr 在冲头前回到安全区
- 1.5 小时 vs 6 小时，PPL 更好

## 实测结果（Qwen3.5-4B, WikiText-2）

| 格式 | PPL | 退化 | 磁盘 | decode tok/s |
|---|---|---|---|---|
| BF16 | 8.88 | 基准 | 8.4 GB | 39.5 |
| NVFP4 (PTQ) | 9.57 | ×1.08 | ~2.4 GB | n/a |
| NF4 | 9.61 | ×1.08 | ~2.4 GB | 32.5 |
| **SPARK MLP-only sprint** | **~13.5** | **×1.5** | **~1.5 GB** | ~55 |

**体积-精度-速度三维度**：
- 体积：1.5 GB（vs BF16 -82%，vs NVFP4 -37%）
- 精度：×1.5 退化（PPL ~13.5，可通过更多 QAT 步数改善）
- 速度：55 tok/s（v3 kernel 后预期 150-400 tok/s）

## 快速开始

```bash
# 1) MLP-only QAT sprint（2000 步，~1.5h）
export NCCL_P2P_DISABLE=1
.venv/bin/python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    03_training/trainer.py \
    --model /path/to/Qwen3.5-4B \
    --data /path/to/Dataset \
    --sample-c4 6 --sample-qwen 4 \
    --steps 2000 --lr 3.3e-5 --min-lr 2.5e-6 \
    --warmup-steps 37 \
    --batch_size 8 --accum 8 --dtype bf16 \
    --quant-mix mlp \
    --token-budget 4096 \
    --ppl-data data/wikitext2.txt --ppl-every 500 \
    --log-every 5 --deepspeed \
    --outdir saves/spark-sprint

# 2) 推理测试
.venv/bin/python3 04_scripts/test_inference.py \
    --ckpt saves/spark-sprint/spark-qat-final.pt \
    --model /path/to/Qwen3.5-4B \
    --interactive --device cuda:0

# 3) 对比矩阵（PPL + 速度 + 显存）
.venv/bin/python3 04_scripts/eval_matrix.py \
    --model /path/to/Qwen3.5-4B \
    --spark-ckpt saves/spark-sprint/spark-qat-final.pt \
    --formats bf16,spark,nvfp4,nf4
```

## 技术要点

### SPFP2 格式（2.5 bit/权重）
- 8 元素块：三元尾数 {0,±1} + 4-bit 共享指数
- v2 双块位打包：5 字节/16 权重
- 窗口式指数搜索（blockmax ±2，与全候选等价）
- QAT STE 直通（零区梯度不设 mask）

### FP3 E1M1 格式（3.5 bit/权重）
- 3-bit 码值 ±{0, 0.5, 1.0, 1.5} + FP8 块 scale
- 真 3-bit 打包：8 码 × 3bit = 3 字节（vs nibble 省 25%）
- PTQ 近无损（PPL 9.8 vs 基线 8.88 = ×1.10）

### v3 CUDA Kernel（在线反量化）
- SPFP2 码流驻留 VRAM → matmul 时 kernel 内解码
- Bit-exact 验证（maxdiff 0.0005，fp16 精度内）
- 0.23 ms/matmul（真实 MLP 尺寸）
- TwoFour 2:4 稀疏版（1.75 bit/w，sparse metadata 生成）

### v2 压缩容器
- zstd 并行压缩（256 线程，37 秒完成）
- NVFP4 原生码流 + SPFP2 packed + INT8 head + fp32 参数
- Roundtrip bit 级一致

## 目录结构

```
SPARK/
├── 01_core/          CUDA kernel + emu（block_fp2 / nvfp4 / fp3 / two_four / int8）
├── 02_model/         量化层 / v2 容器 / loader / 分档替换
├── 03_training/      QAT 训练器（混流 dataloader / PPL 曲线 / 续训 / warmup）
├── 04_scripts/       训练/评测/打包脚本
│   ├── run_exp1.sh   三档混合 QAT
│   ├── run_exp2.sh   激进版（更多 SPFP2）
│   ├── test_inference.py  推理测试 + 打包
│   ├── eval_matrix.py     三维对比矩阵
│   ├── pack_only.py       纯 CPU 打包
│   └── pack_spark.py      打包 + roundtrip 验证
├── 05_tests/         roundtrip / QAT 冒烟
└── docs/             敏感度分析 / 发布物料
```

## 路线图

- [x] 混合分档量化 + QAT 全链路
- [x] MLP-only QAT + PTQ 导出策略
- [x] Sprint 训练（2000 步，防冲头）
- [x] v3 CUDA kernel（bit-exact 验证）
- [x] v2 压缩容器（并行 zstd，bit 级 roundtrip）
- [ ] v3 kernel 集成到推理管线（显存 8→2 GB）
- [ ] 9B 档位验证（~3.4 GB 目标）
- [ ] vLLM 插件（社区 traction 后）

## License

Apache License 2.0 · Copyright (c) 2025 DJAzzs
