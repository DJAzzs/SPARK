# SPARK — 端侧模型 FP2 量化推理引擎

> **S 系 · 推理引擎** —— 面向资源受限设备的极低比特量化 + QAT 全链路。

SPARK 是一个为端侧/边缘设备设计的大模型极低比特（FP2 / 2-bit 尾数）量化推理引擎：
自研 Block-FP2 与 ChannelFP2 自定义 CUDA kernel，将权重按 16 元素块以「2-bit 尾数 + 共享指数」打包为紧凑格式，显著压缩模型体积，并通过**量化感知训练（QAT）**弥补精度损失。

## 技术要点

- **多精度量化格式（混合量化）**：
  - **SPFP2（Block-FP2 / ChannelFP2）**：自研格式与 CUDA kernel（C++/CUDA），16 元素块按 2-bit 尾数 + 共享指数打包；**v2 双块位打包密度 2.25 bit/权重**（2 块 72bit = 9 字节，较 v1 5 字节格式 **-10% 存储**），为当前主存储路线；
  - **NVFP4**：对齐 NVIDIA 规范（E2M1 码值 + 每 16 元素块 FP8 E4M3 scale），有效密度 **4.5 bit/权重**（4 bit 数据 + 8 bit scale ÷ 16），PyTorch 参考仿真 + STE；
  - **INT8 per-channel 对称**：lm_head 等敏感层的混合量化"安全牌"档位。
- **指数搜索（主链路，打包时内嵌）**：`pack_blockwise_paired` 采用**窗口式搜索**——以 blockmax 指数为中心搜 [c−2, c+2]，与全候选（0–15）搜索结果一致，显著快于逐档遍历；指数直接打包进权重格式。
- **离线校准器（独立工具，未接入推理链路）**：`calibrator.py` 为独立离线分析脚本（逐 Block 全候选搜索 → `scale_index_table.pt`）；当前推理主链路走 **packed 打包路线**，不查该表。
- **QAT 全链路**：各格式 STE 直通估计 fake-quant → 打包（窗口式指数搜索）→ GPU dataloader → 训练器 → 引擎导出；提供 0.5B / 2B / 4B / 7B / 9B 多档位 QAT 启动脚本。
- **评测工具链（loglikelihood 口径，与 lm-eval-harness 一致）**：
  - `eval_ppl.py` / `ptq_eval.py`：困惑度与 PTQ 质量（wikitext2 / c4-holdout）；
  - `eval_pmmeval.py`：P-MMEval 验收，**4 任务**（mmmlu 4 选 1 / mhellaswag 4 选 1 / mlogiqa 逻辑推理 / xnli 3 选 1）**× 多语言**（`--langs`，默认 zh,en，数据集覆盖 10 语言；`--split test`（正式）/ `val`（调试）），量化模型 vs 基线；
  - `run_closed_loop.py` / `eval_qat_ckpt.py`：闭环评估与 checkpoint 直接评测。
- **硬件适配**：面向单卡端侧部署，支持低带宽/低显存环境下以最小体积换取可用精度。

## 快速开始

```bash
# 离线校准（只需一次，推理查表）
python 03_training/calibrator.py --model /path/to/model

# QAT 训练（双卡）
./04_scripts/run_qat.sh /path/to/Dataset /output/dir
```

## 目录结构

```
SPARK/
├── 01_core/          自定义 CUDA kernel（block_fp2_kernel.cu / pack.h / fp2 仿真）
├── 02_model/         量化层与模型封装
├── 03_training/      校准器 / GPU dataloader / QAT 训练器
├── 04_scripts/       run_qat{,_2b,_4b,_7b,_9b}.sh · pack_spark.py · export_spark_engine.py · eval_ppl / eval_pmmeval / eval_qat_ckpt / ptq_eval · run_closed_loop.py
├── 05_tests/         单元测试（block 解码 / QAT 冒烟 / roundtrip）
└── inference.py      推理入口
```

## 路线图

- [x] Block-FP2 / ChannelFP2 CUDA kernel（**当前仅支持 v1 5 字节布局**；v2 packed 在 GPU 路径自动退化 emu）
- [x] 窗口式指数搜索（blockmax ±2，与全候选等价，内嵌于打包主链路）
- [x] QAT 全链路（训练 / 导出，0.5B–9B 档位脚本）
- [x] NVFP4 / INT8 混合量化仿真 + STE（对齐 NVIDIA 规范）
- [x] 评测工具链（PPL / PTQ / P-MMEval loglikelihood / 闭环）
- [ ] 汇总各档位量化模型的精度-体积对比报告
- [ ] 推理时动态反量化加速（前提：kernel 支持 v2 packed 布局，见上）

## License

Apache License 2.0 · Copyright (c) 2025 DJAzzs
