"""SPARK QAT 主循环"""
from __future__ import annotations

import os, sys, argparse, time, gzip
sys.path.insert(0, '/home/dja/桌面/远苍')
sys.path.insert(0, '/home/dja/桌面/SPARK')

# Load quant_linear module (avoiding "02_model" as import path)
exec(open('/home/dja/桌面/SPARK/02_model/quant_linear.py').read(), globals())

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load gpu_dataloader
with open('/home/dja/桌面/SPARK/03_training/gpu_dataloader.py') as f:
    exec(f.read())


class SPARKQATrainer:
    def __init__(self, model_path: str, device="cuda:0", max_seq_len: int = 2048):
        self.device = torch.device(device)
        
        print("[SPARK-QA] loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map={"": "cpu"}, low_cpu_mem_usage=True,
            torch_dtype=torch.float32)
        
        print("[SPARK-QA] converting to ChannelFP2Linear...")
        apply_channel_fp2_quant(self.model)
        
        self.model.to(device)
        self.model.gradient_checkpointing_enable()  # 节省显存
        self.model.train()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_seq_len = max_seq_len
        self.loss_weights = {"c4": 0.3, "qwen": 1.0}
    
    def forward_with_separate_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        outputs = self.model(input_ids=input_ids, labels=labels, return_dict=True)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        sources = batch.get('source', ['qwen'] * input_ids.shape[0])
        batch_loss_c4 = []
        batch_loss_qwen = []
        
        for i in range(input_ids.shape[0]):
            src = sources[i] if isinstance(sources, list) else sources.item()
            shift_logits = logits[i, :-1].contiguous()
            shift_labels = labels[i, 1:].contiguous()
            
            active = shift_labels != -100
            if not active.any():
                continue
            
            loss = loss_fct(shift_logits[active], shift_labels[active])
            
            if src == "c4":
                batch_loss_c4.append(loss.mean())
            else:
                batch_loss_qwen.append(loss.mean())
        
        total_loss = 0.0
        c4_loss = qwen_loss = 0
        
        if batch_loss_c4:
            c4_loss = torch.stack(batch_loss_c4).mean()
            total_loss += self.loss_weights["c4"] * c4_loss
        
        if batch_loss_qwen:
            qwen_loss = torch.stack(batch_loss_qwen).mean()
            total_loss += self.loss_weights["qwen"] * qwen_loss
        
        return total_loss, {"c4": float(c4_loss.detach()) if c4_loss.numel() > 0 else 0.0, "qwen": float(qwen_loss.detach()) if qwen_loss.numel() > 0 else 0.0}
    
    def train(self, dataloader, steps: int = 15000,
              outdir: str = "/home/dja/桌面/SPARK/saves/spark-qat"):
        os.makedirs(outdir, exist_ok=True)
        
        accum_steps = 8
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        scaler = torch.cuda.amp.GradScaler()
        
        print(f"[TRAIN] start {steps} steps")
        step_time = time.time()
        
        for step, batch in enumerate(dataloader):
            if step >= steps:
                break
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss, losses = self.forward_with_separate_loss(batch)
            
            scaler.scale(loss / accum_steps).backward()
            
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                elapsed = time.time() - step_time
                print(f"[step {step+1:5d}/{steps}] loss={loss.item():.4f} c4:{losses['c4']:.3f} qwen:{losses['qwen']:.3f}")
                step_time = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/dja/桌面/Models/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default="/home/dja/SPARK/Dataset/c4-train.*.json.gz")
    ap.add_argument("--outdir", default="/home/dja/桌面/SPARK/saves/spark-qat")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=15000)
    args = ap.parse_args()
    
    trainer = SPARKQATrainer(args.model, device="cuda:0", max_seq_len=2048)
    
    print("[GPU DATALOADER] loading...")
    dataset = SPARKIterableDataset(args.data, trainer.tokenizer, 2048)
    dataloader = SPARKDataLoader(dataset, batch_size=args.batch_size)
    
    trainer.train(dataloader, steps=args.steps, outdir=args.outdir)


if __name__ == "__main__":
    main()
