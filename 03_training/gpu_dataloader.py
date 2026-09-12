"""SPARK 流式 GPU 数据加载器（CPU 内存友好）

本版修复：
  - parquet 行数统计只读 metadata（秒级，不再整表读入内存）
  - parquet 流式 iter_batches 逐块读取 + to_pylist（替代慢的 df.iterrows）
  - worker × rank 双维度行级分片：num_workers>0 或 DDP 时不重复消费数据
  - 流式过程有心跳日志（每个文件开始/结束各一条，flush 实时可见）
"""
from __future__ import annotations

import os, sys, json, gzip, glob, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


def _p(msg):
    print(msg, flush=True)


class SPARKIterableDataset(IterableDataset):
    """流式数据集：边读边处理，不预加载到内存。

    rank/world_size: DDP 分片用（每个 rank 处理互不重叠的行子集）。
    worker 分片自动生效（torch DataLoader 的 num_workers>0 时）。
    """
    def __init__(self, data_dir: str, tokenizer,
                 max_seq_len: int = 2048,
                 weights: dict[str, float] = None,
                 rank: int = 0, world_size: int = 1):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size

        if weights is None:
            weights = {"c4": 1.0, "qwen": 2.0}

        # 收集所有数据文件并分配权重
        self.files = []

        c4_files = sorted(glob.glob(os.path.join(data_dir, "c4-train.*.json.gz")))
        for f in c4_files:
            self.files.append((f, 'c4', weights['c4']))

        qwen_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
        t0 = time.time()
        for f in qwen_files:
            try:
                import pyarrow.parquet as pq
                # 只读 footer metadata，不加载数据
                n_rows = pq.ParquetFile(f).metadata.num_rows
                self.files.append((f, 'qwen', weights['qwen'] * n_rows))
            except Exception as e:
                _p(f"[SPARK][data] parquet metadata 读取失败 {f}: {e}")
        _p(f"[SPARK][data] found {len(c4_files)} C4 files, "
           f"{len(qwen_files)} Qwen parquet ({time.time()-t0:.1f}s, 仅读metadata)")

        total_weight = sum(w for _, _, w in self.files)
        self.file_weights = [w/total_weight for _, _, w in self.files] if total_weight else []

    def _shard(self):
        """worker × rank 双维度分片：返回 (我的起始索引, 步长)。"""
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        idx = self.rank * nw + wid
        stride = self.world_size * nw
        return idx, stride

    def _stream_samples(self):
        """迭代所有数据源的样本（生成器），按行级 stride 分片。"""
        import pyarrow.parquet as pq
        idx, stride = self._shard()

        for filepath, source, _ in self.files:
            t0 = time.time()
            n_yield = 0
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    for li, line in enumerate(f):
                        if li % stride != idx:          # 行级分片
                            continue
                        if not line.strip():
                            continue
                        try:
                            o = json.loads(line)
                            text = o.get("text", "")
                            if len(text) > 20:
                                s = self._make_sample(text, source)
                                if s is not None:
                                    yield s
                                    n_yield += 1
                        except Exception:
                            pass
            elif filepath.endswith('.parquet'):
                try:
                    pf = pq.ParquetFile(filepath)       # 流式，不整表加载
                    li = 0
                    for batch in pf.iter_batches(batch_size=1024):
                        for row in batch.to_pylist():
                            if li % stride != idx:      # 行级分片
                                li += 1
                                continue
                            li += 1
                            text = self._extract_text(row)
                            if text and len(text) > 20:
                                s = self._make_sample(text, 'qwen')
                                if s is not None:
                                    yield s
                                    n_yield += 1
                except Exception as e:
                    _p(f"[SPARK][data] parquet 流式读取失败 {filepath}: {e}")
            _p(f"[SPARK][data] {os.path.basename(filepath)} 完成: "
               f"产出 {n_yield} 样本 / {time.time()-t0:.1f}s (shard {idx}/{stride})")

    def _make_sample(self, text, source):
        tokens = self.tokenizer(
            text, max_length=self.max_seq_len, truncation=True,
            padding=False, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze(0)
        if len(input_ids) <= 5:
            return None
        return {'input_ids': input_ids, 'labels': input_ids.clone(),
                'source': source}

    def _extract_text(self, row) -> str | None:
        """从数据行提取文本。兼容多种格式：

        - messages 列（list[dict{role,content}]）：chat 格式
        - instruction/response 列：指令-回答对（本 Dataset 的 parquet 实际格式，
          列名曾不匹配导致这批数据被静默丢弃，见 qwen loss 恒 0 的历史）
        - text 列：纯文本
        """
        msgs = row.get("messages")
        if isinstance(msgs, list):
            return ''.join(m['content'] for m in msgs
                           if isinstance(m, dict) and m.get('role') != 'system')
        instr = row.get("instruction")
        resp = row.get("response")
        if instr or resp:
            return ((instr or "") + "\n" + (resp or "")).strip()
        return row.get("text", "") or ""

    def __iter__(self):
        if getattr(self, '_bucket', True) and getattr(self, '_batch_size', None):
            yield from self._iter_bucketed()
        else:
            yield from self._stream_samples()

    def _iter_bucketed(self):
        """长度分桶：蓄 buffer → 按长度排序 → 切批 → 批顺序打散 → 逐批 collate。

        动机：collate 按 batch 内最长 pad。实测本数据集（c4 平均 416 tok,
        p95 1433）batch16 的 padded/active 比值 ~3.6 —— 即 72% GPU 计算花在
        pad 上。桶内排序组批后批内长度差 <10%，比值降到 ~1.1。
        每个 DataLoader worker 独立分桶（行级分片已保证 worker 间不重复）。
        """
        import random as _random
        rng = _random.Random(os.getpid() ^ id(self))
        buf = []
        for s in self._stream_samples():
            buf.append(s)
            if len(buf) >= self._bucket_size:
                yield from self._flush_buffer(buf, rng)
                buf = []
        if buf:
            yield from self._flush_buffer(buf, rng)

    def _flush_buffer(self, buf, rng):
        buf.sort(key=lambda x: x['input_ids'].shape[0])
        bs = self._batch_size
        batches = [buf[i:i + bs] for i in range(0, len(buf), bs)]
        rng.shuffle(batches)                    # 批顺序打散，保持流式随机性
        for b in batches:
            if b:
                yield collate_batch(b)


def collate_batch(batch: list[dict]) -> dict:
    """统一截断到 2048；空 batch 返回 0 长度标记（训练循环跳过）"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {'input_ids': torch.zeros(0, 1, dtype=torch.long),
                'labels': torch.zeros(0, 1, dtype=torch.long),
                'source': []}

    input_ids = [b['input_ids'][:2048] for b in batch]
    labels = [b['labels'][:2048] for b in batch]

    lens = [len(x) for x in input_ids]
    max_len = max(lens)

    out_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    out_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)

    for i, (ids, lbs) in enumerate(zip(input_ids, labels)):
        L = ids.shape[-1]
        out_ids[i, :L] = ids
        out_labels[i, :L] = lbs

    sources = [b['source'] for b in batch]

    return {'input_ids': out_ids, 'labels': out_labels, 'source': sources}


def _unpack_single(x):
    """DataLoader batch_size=None 时 collate_fn 收到的结构在单进程/多 worker
    路径下不同（list 包装或直接 dict），这里做双态兼容。
    必须是模块级具名函数（lambda 无法跨 worker 进程 pickle）。"""
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


class SPARKDataLoader(DataLoader):
    """流式 DataLoader：默认长度分桶（pad 浪费 3.6x → ~1.1x）。

    bucket=True：dataset 侧组批（batch_size=None + 解包），批内长度相近；
    bucket=False：传统 DataLoader 逐样本组批（批间随机、批内 pad 大）。
    """
    def __init__(self, dataset: SPARKIterableDataset, batch_size=4,
                 bucket: bool = True, bucket_size: int = 512,
                 num_workers: int = 4, **kwargs):
        dataset._batch_size = batch_size
        dataset._bucket = bucket
        dataset._bucket_size = bucket_size
        if bucket:
            super().__init__(
                dataset,
                batch_size=None,           # dataset 已组好整批
                num_workers=num_workers,
                collate_fn=_unpack_single,  # 解包（batch_size=None 会包一层 list）
                shuffle=False,
                **kwargs
            )
        else:
            super().__init__(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_batch,
                shuffle=False,
                **kwargs
            )

    # 兼容旧接口
    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return collate_batch(batch)


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(
        "/home/dja/桌面/Models/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    t0 = time.time()
    dataset = SPARKIterableDataset("/home/dja/桌面/SPARK/Dataset", tok, 2048)
    _p(f"[init] {time.time()-t0:.1f}s")

    def pad_efficiency(loader, n_batches=30):
        """padded/active 比值：越接近 1 越好"""
        padded = active = 0
        for i, b in enumerate(loader):
            if b['input_ids'].shape[0] == 0:
                continue
            padded += b['input_ids'].numel()
            active += int((b['labels'] != -100).sum())
            if i >= n_batches:
                break
        return padded / max(active, 1)

    # bucket 关（传统）需要较大 batch 才看得出差异，用 16
    ds1 = SPARKIterableDataset("/home/dja/桌面/SPARK/Dataset", tok, 2048)
    l1 = SPARKDataLoader(ds1, batch_size=16, bucket=False)
    t0 = time.time()
    r1 = pad_efficiency(l1)
    _p(f"[bucket=off] padded/active = {r1:.2f}  ({time.time()-t0:.0f}s)")

    ds2 = SPARKIterableDataset("/home/dja/桌面/SPARK/Dataset", tok, 2048)
    l2 = SPARKDataLoader(ds2, batch_size=16, bucket=True, bucket_size=512)
    t0 = time.time()
    r2 = pad_efficiency(l2)
    _p(f"[bucket=on ] padded/active = {r2:.2f}  ({time.time()-t0:.0f}s)")
    _p(f"预计 GPU 计算节省: {(1 - r2/r1)*100:.0f}%")
    _p("\n[OK] bucketing dataloader works!")
