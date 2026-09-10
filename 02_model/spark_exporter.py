from __future__ import annotations
import os, sys, torch, shutil
sys.path.insert(0, '/home/dja/桌面/远苍')
sys.path.insert(0, '/home/dja/桌面/SPARK')

def export_spark_model(model: torch.nn.Module, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'embed' not in name.lower() and 'lm_head' not in name:
                if hasattr(module, '_packed_weight') and module._packed_weight is not None:
                    state[name.replace('.weight', '') + '_packed'] = module._packed_weight.cpu()
                    if hasattr(module, '_channel_scale_index'):
                        state[name.replace('.weight', '') + '_scale_index'] = module._channel_scale_index.cpu()
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        shutil.copy(cfg_path, out_dir)
    torch.save(state, os.path.join(out_dir, "state_dict.pt"))
    size_mb = os.path.getsize(os.path.join(out_dir, "state_dict.pt")) / (1024**2)
    print(f"[SPARK EXPORT] {out_dir} ({size_mb:.1f}MB)")
