# SPARK 当前状态与下一步（2026-09-15 交接）

## 仓库状态
- Git 对象文件损坏（`08/e04cc...` 为空），需要 `git fsck --full` 检查
- 所有代码文件完整，仅 commit/push 受影响
- 最后成功 commit: `08e04cc` (v2.2: 块大小 16→8)

## 已完成
- [x] 4B 主训 8000 步（PPL 12.53 @12块 / 14.58 @40块, ×1.64）
- [x] 续训 2500 步确认格式容量天花板（2500步仅压0.26）
- [x] 块大小 16→8 改造（量化误差 -10.6%, 密度 2.5bit, 全链路验证通过）
- [x] v3 Phase 1（two_four.py: 2:4约束 + 1.75bit 位流, roundtrip bit级一致）
- [x] 敏感度分析：**发现 down_proj 系统性敏感**（所有 top-20 都是 down_proj）
- [x] v2 容器（bit级 roundtrip ✓）、eval_matrix、P-MMEval
- [x] 续训基建（resume + warmup + 采样权重 CLI + 单文件语料）

## 关键发现（按重要度）
1. **down_proj 全量比其他 MLP 投影敏感** → 改动②应从"选top-5层"改为"down_proj 全升 NVFP4"
2. RelMSE 分布均匀（σ=0.001）→ 不存在异常差层，逐层选层无意义
3. 格式容量天花板确认 → 更多步数/语料不改善 PPL
4. NVFP4 PTQ ×1.08 vs SPARK ×1.64 → 差距在 MLP 2.25bit 信息瓶颈

## 下一步（按优先级）
1. **修 git**: `git fsck --full` → 如需要 `rm .git/objects/08/e04cc*` + `git reset HEAD~1`
2. **down_proj 全升 NVFP4 分派** + QAT 2000 步验证
3. 块大小 8 的 QAT 验证（代码已改好，只需跑）
4. torchao 2:4 + sm_120 spike（GPU 空闲时几分钟）
5. v3 kernel（CUDA 在线反量化）
6. 9B 训练（等 4B 实验定型后再上）

## 训练命令备忘
```bash
# 4B QAT（块大小 8，需从头训练）
STEPS=2000 ./04_scripts/run_qat_4b.sh

# 4B 续训
STEPS=3000 CKPT=saves/spark-qat-4b/spark-qat-7000.pt ./04_scripts/run_qat_4b_finetune.sh

# 9B
./04_scripts/run_qat_9b.sh

# 评测矩阵
.venv/bin/python3 04_scripts/eval_matrix.py --model /home/dja/桌面/Models/Qwen3.5-4B \
    --spark-ckpt saves/spark-qat-4b/spark-qat-final.pt \
    --formats bf16,spark,nvfp4,nf4 --device cuda:0

# 打包 v2 引擎
.venv/bin/python3 04_scripts/pack_spark.py \
    --ckpt saves/spark-qat-4b/spark-qat-final.pt \
    --model /home/dja/桌面/Models/Qwen3.5-4B --out data/spark-4b-v2
```
