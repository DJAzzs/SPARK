"""SPARK QAT 主循环（真正的量化感知训练，带实时输出）

训练链路：
  1. 加载 base 模型（CPU → fp32）
  2. apply_channel_fp2_qat()  → Linear 替换为 ChannelFP2QATLinear
     （训练 forward 走 channel-FP2 STE fake-quant，梯度直通回 weight）
  3. 流式 dataloader（c4 / qwen 混合；worker×rank 行级分片，不重复消费）
  4. AdamW + 梯度累积 + bf16 autocast
  5. 训练中对各量化层 .quantize() 生成 packed 检查点

实时输出设计：
  - stdout 强制行缓冲（torchrun/后台/nohup 下也能实时看到）
  - 首个 batch 就绪打一条心跳（此前 dataloader 初始化阶段也有日志）
  - 前 3 个 optimizer step 每步打印（热身观察），之后每 log_every 步
  - 日志含 loss / c4 / qwen 分源损失 / 吞吐 tok/s
"""
from __future__ import annotations

import os, sys, argparse, time

# stdout 行缓冲：解决 torchrun / 重定向 / 后台运行时 print 攒批不刷新
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)                 # SPARK/
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "02_model"),
           os.path.join(_PROJECT_ROOT, "01_core"),
           os.path.join(_PROJECT_ROOT, "03_training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from quant_linear import (apply_channel_fp2_qat, apply_mixed_quant,
                          ChannelFP2QATLinear, bump_wq_version,
                          set_force_fake_quant)
from gpu_dataloader import SPARKIterableDataset, SPARKDataLoader


def _p(msg):
    print(msg, flush=True)


_USE_DDP = "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _ddp_setup():
    """torchrun 环境下的 DDP 初始化。单进程 / 非 torchrun 时返回 -1。

    含 NCCL 通信自检：若卡在 all_reduce，即可确诊 NCCL/P2P 问题
    （可尝试 NCCL_P2P_DISABLE=1）。显式 5 分钟超时，不再默认干等 30 分钟。
    """
    if not _USE_DDP:
        return -1
    from datetime import timedelta
    _p("[DDP] init_process_group(nccl) ...")
    torch.distributed.init_process_group(backend="nccl",
                                         timeout=timedelta(minutes=5))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    _p(f"[DDP] rank={os.environ['RANK']} local_rank={local_rank} "
       f"world={os.environ['WORLD_SIZE']}")

    # NCCL 通信自检（小 all_reduce）：卡在这里 = NCCL/P2P 问题
    _p("[DDP] NCCL 通信自检 (all_reduce) ...")
    t = torch.ones(1, device=f"cuda:{local_rank}")
    torch.distributed.all_reduce(t)
    ws = int(os.environ["WORLD_SIZE"])
    _p(f"[DDP] NCCL 通信自检通过: all_reduce(1×{ws}) = {t.item():.0f} ✓")
    return int(os.environ["RANK"])


class SPARKQATrainer:
    def __init__(self, model_path: str, device="cuda:0",
                 max_seq_len: int = 2048,
                 dtype_str: str = "bf16",
                 data_parallel: bool = False,
                 rank: int = 0, world_size: int = 1,
                 deepspeed: bool = False, lr: float = 1e-4,
                 accum_steps: int = 8,
                 no_checkpointing: bool = False,
                 token_budget: int = 8192,
                 eight_bit_optim: bool = False,
                 w_c4: float = 0.6, w_qwen: float = 0.4,
                 quant_mix: str = "fp2", quantize_head: bool = False,
                 ppl_data: str | None = None,
                 micro_batch: int = 8,
                 resume_ckpt: str | None = None,
                 fp3_tier: bool = False):
        self.data_parallel = data_parallel
        self.use_deepspeed = deepspeed
        self.token_budget = token_budget   # 单次 fwd+bwd 的最大 token 数（防 logits OOM）
        self.use_deepspeed = deepspeed
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if data_parallel or deepspeed:
            device = f"cuda:{self.local_rank}"
        self.device = torch.device(device)
        if self.device.type != "cuda":
            _p(f"[WARN] 训练设备 {self.device} 不是 CUDA —— QAT 极慢，仅用于冒烟验证")
        self.dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
        self.use_amp = (self.device.type == "cuda") and not deepspeed

        _p(f"[SPARK-QAT] 加载 base 模型 {model_path} ...")
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map={"": "cpu"}, low_cpu_mem_usage=True,
            torch_dtype=torch.float32)
        _p(f"[SPARK-QAT] 模型加载完成 ({time.time()-t0:.1f}s), "
           f"参数 {sum(p.numel() for p in self.model.parameters())/1e6:.0f}M")

        if quant_mix == "mixed":
            _p("[SPARK-QAT] 混合量化替换 (NVFP4 attn/DeltaNet + SPFP2 MLP + BF16 head)...")
            t1 = time.time()
            apply_mixed_quant(self.model, quantize_head=quantize_head,
                          fp3_tier=fp3_tier)
        else:
            _p("[SPARK-QAT] 替换为 QAT 训练层 (channel-FP2 STE fake-quant)...")
            t1 = time.time()
            apply_channel_fp2_qat(self.model)

        if resume_ckpt:
            # 续训：恢复 QAT 权重 + norm/head 等全部训练终态
            #（优化器动量不恢复——配合 warmup 重建）
            _p(f"[SPARK-QAT] 续训: 恢复权重 {resume_ckpt}")
            from spark_loader import apply_spark_state
            if os.path.isdir(resume_ckpt):
                from spark_v2_container import load_spark_v2
                st = load_spark_v2(resume_ckpt)
            else:
                st = torch.load(resume_ckpt, map_location="cpu",
                                weights_only=False)
            n1, n2, n3 = apply_spark_state(self.model, st)
            _p(f"[SPARK-QAT] resume 应用: packed={n1} nvfp4={n2} params={n3}")

        _p(f"[SPARK-QAT] 替换完成 ({time.time()-t1:.1f}s), 拷贝模型到 {device} ...")
        t1 = time.time()
        self.model.to(device)
        _p(f"[SPARK-QAT] 模型已上设备 ({time.time()-t1:.1f}s)")
        # gradient checkpointing：0.5B/4B 显存富余时应关闭（反向免重算 forward，
        # 白拿 ~25% 提速）；仅 7B+ 显存紧张时开启。
        # non-reentrant 版：reentrant 在 backward 时重跑 forward，与 DDP 的
        # "每个参数只 ready 一次" 假设冲突（报 marked ready twice）。
        if no_checkpointing:
            _p("[SPARK-QAT] gradient checkpointing 已关闭（显存富余，换速度）")
        else:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.train()

        self.ds_engine = None
        if deepspeed:
            _p("[SPARK-QAT] 初始化 DeepSpeed ZeRO-2 (bf16) ...")
            t1 = time.time()
            import deepspeed
            import json
            cfg = {
                "bf16": {"enabled": dtype_str == "bf16"},
                "fp16": {"enabled": dtype_str == "fp16"},
                "zero_optimization": {
                    "stage": 2,
                    "allgather_partitions": True,
                    "reduce_scatter": True,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                },
                "gradient_accumulation_steps": accum_steps,
                "gradient_clipping": 1.0,
                # 具体值（"auto" 需 training_data 才能解析；自管 dataloader 时
                # str 与 int 比较会崩 TypeError）
                "train_micro_batch_size_per_gpu": micro_batch,
                "wall_clock_breakdown": False,
                "steps_per_print": 100,
            }

            # 优化器：8bit（9B 显存兜底，每卡省 ~18GB）或 DeepSpeed AdamW
            custom_opt = None
            if eight_bit_optim:
                try:
                    import bitsandbytes as bnb
                    custom_opt = bnb.optim.AdamW8bit(
                        self.model.parameters(), lr=lr, weight_decay=0.01)
                    _p("[SPARK-QAT] 优化器: bitsandbytes AdamW8bit（ZeRO-2 分片）")
                except ImportError:
                    _p("[SPARK-QAT][WARN] bitsandbytes 未安装，退回 "
                       "DeepSpeed AdamW（显存占用更高）—— pip install bitsandbytes")
            if custom_opt is None:
                cfg["optimizer"] = {
                    "type": "AdamW",
                    "params": {"lr": lr, "weight_decay": 0.01,
                               "betas": [0.9, 0.999], "eps": 1e-8},
                }

            os.makedirs("/tmp/spark_ds", exist_ok=True)
            cfg_path = "/tmp/spark_ds/ds_zero2.json"
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            if custom_opt is not None:
                # 传 optimizer 实例时不能同时传 model_parameters
                self.ds_engine, _, _, _ = deepspeed.initialize(
                    model=self.model, optimizer=custom_opt, config=cfg_path)
            else:
                self.ds_engine, _, _, _ = deepspeed.initialize(
                    model=self.model, config=cfg_path,
                    model_parameters=self.model.parameters())
            self.model = self.ds_engine
            _p(f"[SPARK-QAT] DeepSpeed 引擎就绪 ({time.time()-t1:.1f}s)")
        elif data_parallel:
            _p("[SPARK-QAT] 构建 DDP wrapper（触发参数 broadcast；"
               "卡住通常是 NCCL/P2P 问题，可试 NCCL_P2P_DISABLE=1）...")
            t1 = time.time()
            from torch.nn.parallel import DistributedDataParallel
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank],
                find_unused_parameters=False)
            # bf16 梯度压缩：P2P 禁用后 allreduce 走 SHM，通信量减半（QAT 下
            # 梯度精度损失无感），端到端 ~+10%
            try:
                from torch.distributed.algorithms.ddp_comm_hooks import \
                    default_hooks as comm_hooks
                self.model.register_comm_hook(
                    None, comm_hooks.bf16_compress_hook)
                _p("[SPARK-QAT] DDP bf16 梯度压缩 comm hook 已注册")
            except Exception as e:
                _p(f"[SPARK-QAT][WARN] comm hook 注册失败（不影响正确性）: {e}")
            _p(f"[SPARK-QAT] DDP 就绪 ({time.time()-t1:.1f}s)")
        _p(f"[SPARK-QAT] 全部就绪: device={device}, "
           f"amp={self.dtype if self.use_amp else 'off'}, "
           f"deepspeed={deepspeed}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_seq_len = max_seq_len
        self.loss_weights = {"c4": w_c4, "qwen": w_qwen}
        self._ppl_blocks = self._preload_ppl(ppl_data)

    def _unwrap(self):
        """DDP/DeepSpeed 包装时取原始模块，否则返回 self.model。"""
        if self.ds_engine is not None:
            return self.ds_engine.module
        if self.data_parallel:
            return self.model.module
        return self.model

    def _preload_ppl(self, ppl_data):
        """预载 PPL 评测块（2048 token/块）。无数据则返回 None（跳过 PPL）。"""
        if not ppl_data:
            return None
        try:
            texts = None
            if ppl_data.endswith(".txt") and os.path.isfile(ppl_data):
                with open(ppl_data, encoding="utf-8") as f:
                    texts = [ln for ln in f if ln.strip()]
            elif os.path.isdir(ppl_data):
                from datasets import load_from_disk
                texts = [t for t in load_from_disk(ppl_data)["text"] if t.strip()]
            if not texts:
                return None
            full = "\n\n".join(texts + [""])
            ids = self.tokenizer(full, return_tensors="pt").input_ids[0]
            nb = ids.numel() // 2048
            blocks = ids[:nb * 2048].view(nb, 2048).to(self.device)
            _p(f"[PPL ] 评测集 {ppl_data}: {nb} 块×2048 预载完成")
            return blocks
        except Exception as e:
            _p(f"[PPL ][WARN] PPL 数据加载失败（训练继续，跳过 PPL）: {e}")
            return None

    @torch.no_grad()
    def _ppl_eval(self, max_blocks=40, batch=3, tag=""):
        """训练中 PPL：force_fake_quant 前向（量化语义），返回 ppl 或 None。"""
        if self._ppl_blocks is None:
            return None
        import math
        m = self._unwrap()
        was_training = m.training
        m.eval()
        nq = set_force_fake_quant(m, True)
        blocks = self._ppl_blocks[:max_blocks]
        total_nll, total_tok = 0.0, 0
        # bf16 autocast：PPL 直调裸模块（绕过 DS 引擎的 cast 包装），
        # 与训练时的 dtype 环境对齐，避免 fp32/bf16 混流崩溃
        ac = (torch.autocast("cuda", dtype=torch.bfloat16)
              if self.device.type == "cuda" else torch.autocast("cpu"))
        for i in range(0, blocks.shape[0], batch):
            ids = blocks[i:i + batch]
            with ac:
                out = m(input_ids=ids, labels=ids)
            total_nll += out.loss.item() * ids.numel()
            total_tok += ids.numel()
        set_force_fake_quant(m, False)
        if was_training:
            m.train()
        ppl = math.exp(total_nll / total_tok)
        _p(f"[PPL ] step{tag} ppl={ppl:.3f} ({nq} 量化层, {total_tok} tok)")
        return ppl

    def forward_with_separate_loss(self, batch):
        """前向 + 按数据源拆分的损失。返回 (loss, info, active_tokens).

        显存注意：不把 labels 传给 model —— HF 内置 CE 会物化 [B*S, vocab]
        的 fp32 softmax 中间张量（batch16×seq926 时一次 18GB+），且其结果我们
        并不使用（分源 loss 自己算）。这里只取 logits，逐样本串行计算 CE，
        每样本峰值 ~1GB，即算即释放。
        """
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        outputs = self.model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        sources = batch.get('source', ['qwen'] * input_ids.shape[0])
        n = input_ids.shape[0]
        if isinstance(sources, str):
            sources = [sources] * n

        l_c4, l_qw = [], []
        active_tokens = 0
        for i in range(n):
            src = sources[i]
            # 单样本级 upcast fp32：避免整个 [B, S, V] logits 的 fp32 拷贝
            shift_logits = logits[i, :-1].float().reshape(-1, logits.size(-1))
            shift_labels = labels[i, 1:].contiguous().reshape(-1)
            active = shift_labels != -100
            if not active.any():
                continue
            active_tokens += int(active.sum().item())
            loss = loss_fct(shift_logits[active], shift_labels[active]).mean()
            if src == "c4":
                l_c4.append(loss)
            else:
                l_qw.append(loss)

        # 归一化加权：total = (w_c4·c4 + w_qwen·qwen) / (w_c4 + w_qwen)。
        # 历史事故：1:1 权重未归一化时 total loss ×3.6 → 等效 lr 过冲 →
        # 权重塌缩 + PPL 停滞 2000 高原（2B 10000 步实测，2026-09-13）。
        wsum = self.loss_weights["c4"] + self.loss_weights["qwen"]
        total_loss = torch.tensor(0.0, device=self.device)
        c4 = qwen = None
        if l_c4:
            c4 = torch.stack(l_c4).mean()
            total_loss = total_loss + self.loss_weights["c4"] * c4
        if l_qw:
            qwen = torch.stack(l_qw).mean()
            total_loss = total_loss + self.loss_weights["qwen"] * qwen
        total_loss = total_loss / wsum

        info = {"c4": c4.detach().item() if c4 is not None else 0.0,
                "qwen": qwen.detach().item() if qwen is not None else 0.0}
        return total_loss, info, active_tokens

    def _fwd_bwd(self, batch, use_amp, amp_dtype, accum_steps, use_ds):
        """forward + backward 一体，带两级显存保护。返回 (loss, info, atk)。

        为什么 forward/backward 必须一体：backward 无法在 OOM 后无损重试
        （autograd 图已消耗），所以显存控制要在进入 forward 前完成：
          1. token 预算预劈分：B×S 超过 token_budget 就先劈半，各自
             forward+backward（梯度自然累加，数学语义不变）。
             动机：logits 及其梯度是 [B, S, vocab] 张量（本模型 vocab=151936，
             16×926 batch 的 dlogits fp32 一次 9.3GB），长样本组合是 OOM 主因。
          2. OOM 兜底：forward 阶段（图未消耗）仍可 catch 后劈半重试。
        """
        n = batch['input_ids'].shape[0]
        n_tok = n * batch['input_ids'].shape[1]

        # ---- 1) token 预算预劈分（主保护）----
        if n > 1 and n_tok > self.token_budget:
            half = n // 2
            results = []
            for sl in (slice(0, half), slice(half, n)):
                sub = {'input_ids': batch['input_ids'][sl],
                       'labels': batch['labels'][sl],
                       'source': [batch['source'][j]
                                  for j in range(*sl.indices(n))]}
                results.append(self._fwd_bwd(sub, use_amp, amp_dtype,
                                             accum_steps, use_ds))
            # 加权合并供日志（梯度已在各半 backward 中累加）
            tot = sum(r[2] for r in results) or 1
            merged_loss = sum(r[0].detach() * r[2] for r in results) / tot
            infos = {k: sum(r[1][k] * r[2] for r in results) / tot
                     for k in ("c4", "qwen")}
            return merged_loss, infos, tot

        # ---- 2) 单块 forward+backward ----
        try:
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss, info, atk = self.forward_with_separate_loss(batch)
            else:
                loss, info, atk = self.forward_with_separate_loss(batch)
        except torch.OutOfMemoryError:
            # forward 阶段 OOM（图未消耗）→ 劈半重试
            if n <= 1:
                raise
            torch.cuda.empty_cache()
            if self.is_main:
                _p(f"[WARN ] OOM on batch{tuple(batch['input_ids'].shape)}, "
                   f"劈半重试")
            half = n // 2
            results = []
            for sl in (slice(0, half), slice(half, n)):
                sub = {'input_ids': batch['input_ids'][sl],
                       'labels': batch['labels'][sl],
                       'source': [batch['source'][j]
                                  for j in range(*sl.indices(n))]}
                results.append(self._fwd_bwd(sub, use_amp, amp_dtype,
                                             accum_steps, use_ds))
            tot = sum(r[2] for r in results) or 1
            merged_loss = sum(r[0].detach() * r[2] for r in results) / tot
            infos = {k: sum(r[1][k] * r[2] for r in results) / tot
                     for k in ("c4", "qwen")}
            return merged_loss, infos, tot

        # backward（此处图必完整且显存可控）
        if use_ds:
            self.ds_engine.backward(loss)
            self.ds_engine.step()
        else:
            (loss / accum_steps).backward()
        return loss, info, atk

    def train(self, dataloader, steps: int = 15000,
              outdir: str = "/home/dja/桌面/SPARK/saves/spark-qat",
              accum_steps: int = 8, lr: float = 1e-4,
              log_every: int = 10,
              lr_schedule: str = "cosine", min_lr: float = 1e-6,
              ppl_every: int = 500, warmup_steps: int = 0):
        os.makedirs(outdir, exist_ok=True)

        use_ds = self.ds_engine is not None
        optimizer = None
        if not use_ds:
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=lr, weight_decay=0.01)
        amp_dtype = self.dtype
        # DeepSpeed bf16 引擎自己管理混合精度，不再叠 autocast
        use_amp = (self.use_amp and not use_ds
                   and amp_dtype in (torch.float16, torch.bfloat16))

        # ---- cosine LR 衰减（DDP/DeepSpeed 两分支通用）----
        # 总 optimizer 步数 = micro 步数 / 梯度累积；衰减到 min_lr。
        total_opt_steps = max(steps // max(accum_steps, 1), 1)
        scheduler = None
        if lr_schedule == "cosine" and total_opt_steps > 1 and not use_ds:
            # warmup>0 时走手动路径（与 DS 一致）；否则保留 torch scheduler
            if warmup_steps > 0:
                scheduler = None
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_opt_steps, eta_min=min_lr)
        if lr_schedule == "cosine" and total_opt_steps > 1 and self.is_main:
            _p(f"[TRAIN] cosine LR: {lr} -> {min_lr} over "
               f"{total_opt_steps} optimizer steps "
               f"({'DeepSpeed 手动 cosine' if use_ds else 'torch scheduler'})")

        import math as _math

        def _lr_at(t):
            """warmup(线性 0→lr) + cosine(lr→min_lr) 统一调度。"""
            t = min(t, total_opt_steps)
            if warmup_steps > 0 and t <= warmup_steps:
                return lr * (t / warmup_steps)
            return min_lr + (lr - min_lr) * (
                1 + _math.cos(_math.pi * (t - warmup_steps)
                              / max(total_opt_steps - warmup_steps, 1))) / 2

        _ds_cosine_lr = _lr_at   # 兼容旧名

        if self.is_main:
            _p(f"[TRAIN] start {steps} steps (accum={accum_steps}, "
               f"{amp_dtype}, amp={use_amp}, "
               f"{'DeepSpeed-ZeRO2' if use_ds else 'DDP/plain'}, "
               f"loss_w c4:{self.loss_weights['c4']} "
               f"qwen:{self.loss_weights['qwen']}, "
               f"log_every={log_every})")
        self.model.train()
        t0 = time.time()
        tokens_seen = 0
        t_tokens = t0
        opt_step = 0
        first_batch_seen = False

        for step, batch in enumerate(dataloader):
            if step >= steps:
                break
            if batch is None or batch['input_ids'].shape[0] == 0:
                continue

            if not first_batch_seen:
                if self.is_main:
                    _p(f"[DATA ] 首个 batch 就绪 ({time.time()-t0:.1f}s): "
                       f"shape={tuple(batch['input_ids'].shape)}, "
                       f"sources={batch['source'][:3]}")
                first_batch_seen = True

            # forward+backward 一体（含 token 预算预劈分与 OOM 兜底）
            loss, info, atk = self._fwd_bwd(batch, use_amp, amp_dtype,
                                            accum_steps, use_ds)
            tokens_seen += atk

            if (step + 1) % accum_steps == 0:
                opt_step += 1
                if not use_ds:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    if scheduler is None:      # warmup 手动路径
                        _lr_t = _lr_at(opt_step)
                        for _g in optimizer.param_groups:
                            _g['lr'] = _lr_t
                    optimizer.step()
                    optimizer.zero_grad()
                if use_ds:
                    # DeepSpeedZeroOptimizer 非 torch Optimizer 子类，
                    # cosine 手动写入 param_groups（与 torch scheduler 等价）
                    _lr_t = _ds_cosine_lr(opt_step)
                    for _g in self.ds_engine.optimizer.param_groups:
                        _g['lr'] = _lr_t
                elif scheduler is not None:
                    scheduler.step()
                # 权重已更新 → 所有 QAT 层的 fake-quant 缓存失效
                bump_wq_version()

                cur_lr = (scheduler.get_last_lr()[0] if scheduler is not None
                          else (_ds_cosine_lr(opt_step) if use_ds else lr))
                # 实时日志：前 3 个 optimizer step 每步打，之后每 log_every 步
                if self.is_main and (opt_step <= 3 or opt_step % log_every == 0):
                    now = time.time()
                    tokps = tokens_seen / max(now - t_tokens, 1e-9)
                    el = now - t0
                    _p(f"[step {step+1:6d}/{steps}] opt#{opt_step} "
                       f"loss={loss.item():.4f} c4:{info['c4']:.3f} "
                       f"qwen:{info['qwen']:.3f} | lr={cur_lr:.2e} | "
                       f"{tokps:7.0f} tok/s | {el/60:.1f}min")
                    t_tokens = now
                    tokens_seen = 0

            if ppl_every > 0 and (step + 1) % ppl_every == 0 and self.is_main:
                self._ppl_eval(tag=f" {step+1}")

            if (step + 1) % 1000 == 0:
                if self.is_main:
                    _p(f"[CKPT ] {step+1} 步，保存量化检查点...")
                self._save_quantized(outdir, tag=step + 1)

        if self.is_main:
            self._save_quantized(outdir, tag="final")
            _p(f"[DONE ] training finished in {(time.time()-t0)/60:.1f} min")

    def _save_quantized(self, outdir, tag="final"):
        """保存完整训练终态（仅 rank0）：
        - SPFP2 层：_packed(v2)/_scale_index/_meta
        - NVFP4 层（attn/embed/tied-head）：_nvfp4_weight（fake-quant 值, bf16）+ _meta
        - 其余全部参数（norm/lm_head/A_log 等训练中更新过的）：param::前缀 bf16
        历史 bug：曾只存 SPFP2 层 → mixed 模式丢 NVFP4/lm_head；
        norm 不同步会使 eval 成为"新量化层+旧norm"的错配体。
        """
        if not self.is_main:
            return
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        state = {}
        nq = 0
        m = self._unwrap()
        from quant_linear import (ChannelFP2QATLinear, NVFP4QATLinear,
                                  NVFP4EmbeddingQAT)
        try:
            from nvfp4_emu import nvfp4_fake_quant
        except ImportError:
            nvfp4_fake_quant = None
        quant_param_ids = set()
        with torch.no_grad():
            for name, module in m.named_modules():
                n = name.replace('.weight', '')
                if isinstance(module, ChannelFP2QATLinear):
                    module.quantize()
                    nq += 1
                    state[n + '_packed'] = module._packed_weight.cpu()
                    if module._channel_scale_index is not None:
                        state[n + '_scale_index'] = module._channel_scale_index.cpu()
                    state[n + '_meta'] = torch.tensor(
                        [module.out_features, module.in_features], dtype=torch.long)
                    quant_param_ids.add(id(module.weight))
                elif isinstance(module, (NVFP4QATLinear, NVFP4EmbeddingQAT)):
                    nq += 1
                    # 原生码流（E2M1 nibble + FP8 scale）：比 bf16 量化值小 4 倍，
                    # 且 pack 幂等（bf16 值二次量化会因边界穿越差一个码位）
                    from nvfp4_emu import nvfp4_pack
                    w = module.weight.detach().float()
                    wq = nvfp4_fake_quant(w) if nvfp4_fake_quant else w
                    codes, scales = nvfp4_pack(wq)
                    state[n + '_nvfp4_codes'] = codes.cpu()
                    state[n + '_nvfp4_scales'] = scales.cpu()
                    oc, ic = module.weight.shape
                    state[n + '_meta'] = torch.tensor([oc, ic], dtype=torch.long)
                    quant_param_ids.add(id(module.weight))
            # 非量化参数（norm / lm_head(BF16训练) / A_log 等）——训练更新必须随行
            for pname, p in m.named_parameters():
                if id(p) in quant_param_ids:
                    continue
                state['param::' + pname] = p.detach().float().bfloat16().cpu()
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"spark-qat-{tag}.pt")
        torch.save(state, path)
        size_mb = os.path.getsize(path) / (1024 ** 2)
        _p(f"[CKPT ] saved {nq} quant layers + "
           f"{sum(1 for k in state if k.startswith('param::'))} params "
           f"-> {path} ({size_mb:.1f}MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/dja/桌面/Models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="/home/dja/桌面/SPARK/Dataset")
    ap.add_argument("--outdir", default="/home/dja/桌面/SPARK/saves/spark-qat")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--log-every", type=int, default=10,
                    help="每 N 个 optimizer step 打一条日志（前3步总是打）")
    ap.add_argument("--ddp", action="store_true",
                    help="启用 DDP（配 torchrun --nproc_per_node=2）")
    ap.add_argument("--deepspeed", action="store_true",
                    help="启用 DeepSpeed ZeRO-2（bf16，优化器+梯度分片，"
                         "4B/7B 必备；配 torchrun --nproc_per_node=2）")
    ap.add_argument("--no-checkpointing", action="store_true",
                    help="关闭 gradient checkpointing（0.5B/4B 显存富余时用，"
                         "反向免重算 forward，约 25 percent 提速；7B 不要用）")
    ap.add_argument("--token-budget", type=int, default=8192,
                    help="单次 fwd+bwd 最大 token 数（logits 及其梯度是 "
                         "[B,S,vocab] 张量，超预算自动劈半防 OOM）")
    ap.add_argument("--optim-8bit", action="store_true",
                    help="DeepSpeed 分支下用 bitsandbytes AdamW8bit（9B 显存"
                         "兜底，每卡省约 18GB；需 pip install bitsandbytes）")
    ap.add_argument("--sample-c4", type=float, default=6.0,
                    help="混流采样权重 c4 分量（默认 6 → 6:4）")
    ap.add_argument("--sample-qwen", type=float, default=4.0,
                    help="混流采样权重 qwen 分量")
    ap.add_argument("--resume-ckpt", default=None,
                    help="续训：恢复 QAT 权重（.pt 或 v2 容器）；配合 --warmup-steps")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="优化器步数的线性热身（续训时建议 ~200 重建动量）")
    ap.add_argument("--ppl-every", type=int, default=500,
                    help="每 N 步测一次 PPL（0=关闭；checkpoint 仍每 1000 步）")
    ap.add_argument("--ppl-data", default="data/wikitext2-val",
                    help="PPL 评测数据（save_to_disk 目录或 .txt）；每 1000 步自动测。"
                         "不存在则跳过 PPL")
    ap.add_argument("--quantize-head", action="store_true",
                    help="混合策略下量化 embed(NVFP4)/lm_head(INT8)，tied 自动共享"
                         "（9B 级 ~4.3GB 目标用；默认 head 保持 BF16）")
    ap.add_argument("--fp3-tier", action="store_true",
                    help="三档混合: GDN门控+MLP→SPFP2, 其余attn/GDN→FP3, out_proj+embed→NVFP4")
    ap.add_argument("--quant-mix", default="fp2", choices=["fp2", "mixed"],
                    help="量化策略：fp2=统一 SPFP2 | mixed=NVFP4(attn/DeltaNet)"
                         "+SPFP2(MLP)+BF16(head/状态参数)，对混合注意力架构"
                         "（Qwen3.5 DeltaNet）推荐 mixed")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader worker 数（4-8；数据侧瓶颈时加大）")
    ap.add_argument("--lr-schedule", default="cosine",
                    choices=["cosine", "constant"],
                    help="学习率调度：cosine 衰减到 min_lr / constant")
    ap.add_argument("--min-lr", type=float, default=1e-6,
                    help="cosine 调度的终点学习率")
    ap.add_argument("--w-c4", type=float, default=1.0,
                    help="c4 数据源的 loss 权重（采样配比已在 dataloader 控制"
                         "（6:4 混流），loss 默认中性 1:1）")
    ap.add_argument("--w-qwen", type=float, default=1.0,
                    help="qwen 数据源的 loss 权重（默认中性）")
    args = ap.parse_args()

    rank = _ddp_setup()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ds = args.deepspeed and rank >= 0
    data_parallel = args.ddp and rank >= 0 and not use_ds
    if rank >= 0 and not (use_ds or data_parallel):
        _p("[FATAL] torchrun 多进程环境但未启用 --ddp/--deepspeed："
           "所有 rank 将挤在 cuda:0 上导致 OOM！检查启动脚本的续行符。")
        import sys as _sys
        _sys.exit(1)

    trainer = SPARKQATrainer(args.model, device=args.device,
                             max_seq_len=2048, dtype_str=args.dtype,
                             data_parallel=data_parallel,
                             rank=max(rank, 0), world_size=world_size,
                             deepspeed=use_ds, lr=args.lr,
                             accum_steps=args.accum,
                             no_checkpointing=args.no_checkpointing,
                             token_budget=args.token_budget,
                             eight_bit_optim=args.optim_8bit,
                             w_c4=args.w_c4, w_qwen=args.w_qwen,
                             quant_mix=args.quant_mix,
                             quantize_head=args.quantize_head,
                             ppl_data=args.ppl_data,
                             micro_batch=args.batch_size,
                             resume_ckpt=args.resume_ckpt,
                             fp3_tier=args.fp3_tier)

    _p("[DATA ] 初始化流式 dataloader（仅读 metadata，不整表加载）...")
    t0 = time.time()
    dataset = SPARKIterableDataset(
        args.data, trainer.tokenizer, 2048,
        rank=max(rank, 0), world_size=world_size,
        weights={"c4": args.sample_c4, "qwen": args.sample_qwen})
    dataloader = SPARKDataLoader(dataset, batch_size=args.batch_size,
                                 pin_memory=torch.cuda.is_available(),
                                 num_workers=args.num_workers)
    _p(f"[DATA ] dataloader 就绪 ({time.time()-t0:.1f}s, "
       f"num_workers={args.num_workers}, "
       f"rank={max(rank,0)}/{world_size} 行级分片)")

    trainer.train(dataloader, steps=args.steps, outdir=args.outdir,
                  accum_steps=args.accum, lr=args.lr,
                  log_every=args.log_every,
                  lr_schedule=args.lr_schedule, min_lr=args.min_lr,
                  ppl_every=args.ppl_every, warmup_steps=args.warmup_steps)


if __name__ == "__main__":
    main()
