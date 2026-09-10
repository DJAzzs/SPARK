"""SPARK 流式 GPU 数据加载器（CPU 内存友好）"""
from __future__ import annotations

import os, sys, json, gzip, glob
sys.path.insert(0, '/home/dja/桌面/远苍')
sys.path.insert(0, '/home/dja/桌面/SPARK')

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, WeightedRandomSampler
from transformers import AutoTokenizer


class SPARKIterableDataset(IterableDataset):
    """流式数据集：边读边处理，不预加载到内存"""
    def __init__(self, data_dir: str, tokenizer,
                 max_seq_len: int = 2048,
                 weights: dict[str, float] = None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        if weights is None:
            weights = {"c4": 1.0, "qwen": 2.0}
        
        # 收集所有数据文件并分配权重
        self.files = []
        
        c4_files = sorted(glob.glob(os.path.join(data_dir, "c4-train.*.json.gz")))
        for f in c4_files:
            self.files.append((f, 'c4', weights['c4']))
        
        qwen_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
        try:
            import pyarrow.parquet as pq
            for f in qwen_files:
                n_rows = pq.read_table(f).num_rows
                self.files.append((f, 'qwen', weights['qwen'] * n_rows))
        except Exception:
            pass
        
        total_weight = sum(w for _, _, w in self.files)
        self.file_weights = [w/total_weight for _, _, w in self.files]
        
        print(f"[SPARK] found {len(c4_files)} C4 files, {len(qwen_files)} Qwen files")
    
    def _stream_samples(self):
        """迭代所有数据源的样本（生成器）"""
        import pyarrow.parquet as pq
        
        for filepath, source, _ in self.files:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            o = json.loads(line)
                            text = o.get("text", "")
                            if len(text) > 20:
                                tokens = self.tokenizer(
                                    text, max_length=self.max_seq_len, truncation=True,
                                    padding=False, return_tensors='pt'
                                )
                                input_ids = tokens['input_ids'].squeeze(0)
                                labels = input_ids.clone()
                                if len(input_ids) > 5:
                                    yield {'input_ids': input_ids, 'labels': labels, 'source': source}
                        except Exception:
                            pass
            elif filepath.endswith('.parquet'):
                try:
                    table = pq.read_table(filepath)
                    df = table.to_pandas()
                    for _, row in df.iterrows():
                        text = self._extract_text(row)
                        if text and len(text) > 20:
                            tokens = self.tokenizer(
                                text, max_length=self.max_seq_len, truncation=True,
                                padding=False, return_tensors='pt'
                            )
                            input_ids = tokens['input_ids'].squeeze(0)
                            labels = input_ids.clone()
                            if len(input_ids) > 5:
                                yield {'input_ids': input_ids, 'labels': labels, 'source': 'qwen'}
                except Exception:
                    pass
    
    def _extract_text(self, row) -> str | None:
        """从 dataclass 行提取文本"""
        if isinstance(row.get("messages"), list):
            return ''.join(m['content'] for m in row["messages"] if m.get('role') != 'system')
        return row.get("text", "") or ""
    
    def __iter__(self):
        yield from self._stream_samples()


class SPARKDataLoader(DataLoader):
    """带WeightedRandomSampler的 DataLoader（流式迭代器需特殊处理）"""
    def __init__(self, dataset: SPARKIterableDataset, batch_size=4, **kwargs):
        # 流式数据集无法用 WeightedRandomSampler，改用随机打乱文件顺序
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=2,
            collate_fn=self.collate_fn,
            shuffle=False,  # 文件已打乱
            **kwargs
        )
    
    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """统一截断到 2048"""
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        input_ids = [b['input_ids'][:2048] for b in batch]
        labels = [b['labels'][:2048] for b in batch]
        
        lens = [len(x) for x in input_ids]
        max_len = max(lens)
        
        out_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
        out_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        
        for i, (ids, lbs) in enumerate(zip(input_ids, labels)):
            L = ids.shape[-1]
            out_ids[i, :L] = ids
            out_labels[i, :L] = lbs
        
        sources = [b['source'] for b in batch]
        
        return {'input_ids': out_ids, 'labels': out_labels, 'source': sources}


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("/home/dja/桌面/Models/Qwen2.5-7B-Instruct", trust_remote_code=True)
    dataset = SPARKIterableDataset("/home/dja/桌面/SPARK/Dataset", tok, 2048)
    loader = SPARKDataLoader(dataset, batch_size=1)
    
    for i, b in enumerate(loader):
        if b is None: continue
        print(f"Batch {i}: input_ids={b['input_ids'].shape}, sources={b['source']}")
        if i >= 2:
            break
    print("\n[OK] Streaming dataloader works!")
