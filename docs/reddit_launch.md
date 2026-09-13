# r/LocalLLaMA 发布帖（草稿 v1）

> 发布时机：验收数据齐 → 填占位符 → 周日晚/周一早（trending 低谷）
> 占位符格式：[xxx] / [×1.2x] / [12.x] —— 全部待实测后替换，不可虚填

---

## 标题（三选一）

1. **[Project] SPARK: 3.56-bit hybrid quantization for hybrid-attention LLMs — 1.95GB Qwen3.5-4B with PPL within [×1.2x] of BF16, full QAT toolchain included**

2. **[Project] We quantized Qwen3.5-4B to 3.56 bits (1.95GB) and recovered it to [×1.2x] PPL with QAT — the DeltaNet layers taught us why uniform quantization fails on hybrid-attention models**

3. **[Project] A 2.25-bit ternary format for MLPs + NVFP4 for attention + QAT recovery: the SPARK quantization engine**

（推荐 #2：有故事弧线——"taught us why"比数字更能引发讨论）

---

## 正文

Hey folks, we've been building **SPARK** — an open-source quantization engine for edge inference, and just finished the first full training run on Qwen3.5-4B. The interesting part isn't just the numbers, it's *why* uniform quantization breaks on hybrid-attention architectures.

### The finding that started it

Qwen3.5 mixes 24 linear-attention (DeltaNet) layers with 8 full-attention layers. We measured per-module quantization sensitivity and found the DeltaNet projections (especially `out_proj`, which reads out the recurrent state) are **orders of magnitude more sensitive than lm_head**. Quantize those to 2-bit and the model outputs garbage — even when only 12% of parameters are touched. The SSM-style state propagation accumulates quantization noise across every token.

### The approach: sensitivity-tiered hybrid format

| Tier | Modules | Format | Density |
|---|---|---|---|
| NVFP4 | attention/DeltaNet projections, embeddings | E2M1 + FP8 block scales | 4.5 bit |
| SPFP2 | all MLPs (gate/up/down) | ternary {0,±1} + 4-bit shared exponent, pair-packed | 2.25 bit |
| INT8 | lm_head (untied models) | per-channel, PTQ post-training | 8 bit |
| BF16 | A_log / dt_bias and other state params | — | 16 bit |

The MLPs carry ~47% of parameters at 2.25 bits — that's where the compression comes from — while everything touching recurrent state stays at higher precision.

### Results on Qwen3.5-4B (WikiText-2)

| Format | PPL | Degradation | Size | decode tok/s* |
|---|---|---|---|---|
| BF16 | [9.xx] | — | 8.4 GB | [xxx] |
| NVFP4 (PTQ) | [xx.x] | ×[1.xx] | 2.4 GB | [xxx] |
| NF4 | [xx.x] | ×[1.xx] | 2.4 GB | [xxx] |
| **SPARK v2 (QAT, 7k steps)** | **[12.x]** | **×[1.2x]** | **1.95 GB** | [xxx] |

*single RTX PRO 6000 Blackwell, batch=1. Honest caveat: current inference decodes weights to BF16 in memory, so speed ≈ BF16 for now — the dequant-on-the-fly kernel is the next milestone (v3, with 2:4 sparsity on the ternary layers since they're naturally 55% sparse).

### QAT recovery curve (the fun part)

Pure PTQ of this config lands at PPL ~26 (vs ~9.3 baseline). After 7,000 steps of QAT with STE fake-quant:

```
step  500: 25.8 → step 2000: 18.6 → step 4000: 15.3 → step 7000: 12.7
```

Training details: mixed 6:4 C4/instruction data, length-bucketed batches, cosine 7e-5, DeepSpeed ZeRO-2, ~5 hours on 2×RTX PRO 6000. Full PPL trajectory is logged every 500 steps in the repo.

### Toolchain (everything is reproducible)

```bash
# QAT training (dual GPU)
./04_scripts/run_qat_4b.sh

# Pack to .spark v2 container (zstd + native bitstreams)
python3 04_scripts/pack_spark.py --ckpt ... --out engine.spark

# Benchmark matrix (PPL + speed + VRAM vs bf16/nvfp4/nf4)
python3 04_scripts/eval_matrix.py --formats bf16,spark,nvfp4,nf4
```

### Limitations (being upfront)

- Speed/VRAM advantage requires the v3 kernel (WIP) — today the win is **disk size and quality-per-bit**, not throughput
- Tested on Qwen3.5-4B (9B run queued); Llama-family needs the sensitivity map re-measured
- 2.25-bit MLPs will never match 4-bit PTQ on raw PPL — the trade is deliberate (check the size column)

**Repo**: github.com/DJAzzs/SPARK (Apache-2.0, v2.0 release with the prebuilt 4B engine coming this week)

Happy to answer anything about the DeltaNet sensitivity measurements or the ternary format design.

---

## 发布 checklist

- [ ] 账号注册后**先参与 1-2 天社区讨论**（避 9:1 自我推广过滤）
- [ ] 验收数据填入所有 [占位符]（eval_matrix 实测，一个不能虚）
- [ ] v2.0 tag + push（帖子链接到的必须是已更新的 README）
- [ ] HF Repo 放预训练 4B 引擎（"免训练可玩"是转化关键）
- [ ] 发布后 2 小时内积极回评（trending 算法看互动）
- [ ] 准备好回答 "vs GPTQ?" —— 答案在矩阵表里，主动引用
- [ ] 帖子分类选 [Project] flair
