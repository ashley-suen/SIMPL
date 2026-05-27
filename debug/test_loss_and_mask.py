"""
Standalone smoke test for:
  1. AV2PromptDataset — verifies train_mask is present and correct shape
  2. LLMMotionLoss    — verifies FDE-WTA + Soft-WTA + absolute-coord loss (no NaN/Inf)
  3. Full model forward + backward — verifies gradient flows without error

Run from SIMPL_back root:
    python debug/test_loss_and_mask.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── 1. Dataset smoke test ───────────────────────────────────────────────────
print("\n[1/3] Dataset — checking train_mask field...")
from data_av2.av2_llm_dataset import AV2PromptDataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data_av2", "features", "train")
ds = AV2PromptDataset(DATA_DIR, max_len_per_agent=256)
print(f"  Dataset size: {len(ds)}")

sample = ds[0]
assert "train_mask" in sample, "FAIL: 'train_mask' key missing from dataset output"
assert "agent_valid" in sample, "FAIL: 'agent_valid' key missing"
tm = sample["train_mask"]
av = sample["agent_valid"]
print(f"  train_mask shape : {tm.shape}  dtype={tm.dtype}")
print(f"  agent_valid shape: {av.shape}  dtype={av.dtype}")
print(f"  train_mask values: {tm.tolist()}")
print(f"  agent_valid values: {av.tolist()}")
assert tm.dtype == torch.bool, "FAIL: train_mask should be bool"
assert (tm & ~av).sum() == 0, "FAIL: train_mask should be subset of agent_valid"
print("  [PASS] Dataset train_mask OK")

# ─── 2. Loss smoke test (no model, mock tensors) ─────────────────────────────
print("\n[2/3] Loss — FDE-WTA + Soft-WTA + absolute coord regression...")
from simpl.av2_llm_loss import LLMMotionLoss

B, N, K, T = 2, 4, 6, 60
loss_fn = LLMMotionLoss(device=DEVICE, soft_wta_alpha=0.1)

torch.manual_seed(0)
pred_trajs = torch.randn(B, N, K, T, 2, device=DEVICE, requires_grad=True)
gt_abs     = torch.randn(B, N, T, 2, device=DEVICE)
anchor     = torch.randn(B, N, 2, device=DEVICE)
gt_masks   = torch.ones(B, N, T, dtype=torch.bool, device=DEVICE)
agent_valid = torch.ones(B, N, dtype=torch.bool, device=DEVICE)
train_mask  = torch.ones(B, N, dtype=torch.bool, device=DEVICE)
# Simulate: last agent slot is unscore — train_mask=False but agent_valid=True
train_mask[:, -1] = False

data = {
    "gt_abs_trajectories": gt_abs,
    "gt_anchor":           anchor,
    "gt_masks":            gt_masks,
    "agent_valid":         agent_valid,
    "train_mask":          train_mask,
}

out = loss_fn(pred_trajs, data)
loss = out["loss"]
print(f"  loss value: {loss.item():.6f}")
assert not torch.isnan(loss), "FAIL: loss is NaN"
assert not torch.isinf(loss), "FAIL: loss is Inf"
assert loss.item() > 0,       "FAIL: loss should be > 0 for random predictions"

# backward
loss.backward()
assert pred_trajs.grad is not None, "FAIL: no gradient on pred_trajs"
grad_norm = pred_trajs.grad.norm().item()
print(f"  grad norm on pred_trajs: {grad_norm:.4f}")
assert not torch.isnan(pred_trajs.grad).any(), "FAIL: NaN in gradients"

# verify unscored agents (index -1) contribute zero to loss
pred_trajs2 = pred_trajs.detach().clone().requires_grad_(True)
pred_trajs2_mod = pred_trajs2.clone()
# perturb the unscored agent slot massively — loss should not change
with torch.no_grad():
    pred_trajs2_mod[:, -1] += 1000.0
out2 = loss_fn(pred_trajs2_mod.requires_grad_(True), data)
loss2 = out2["loss"]
print(f"  loss with perturbed unscore agent: {loss2.item():.6f}")
assert abs(loss2.item() - loss.item()) < 1e-3, \
    f"FAIL: unscore agent affected loss (original={loss.item():.6f}, perturbed={loss2.item():.6f})"
print("  [PASS] Loss function OK — unscore agents correctly excluded")

# ─── 3. Full model forward + backward ─────────────────────────────────────────
print("\n[3/3] Full model forward + backward (SmolLM-135M)...")
from simpl.llm_motion_model import SmolLMMotionPredictor

model = SmolLMMotionPredictor(num_modes=6, device=DEVICE)
model.train()

# Use a real batch from the dataloader
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(dl))

input_ids      = batch["input_ids"].to(DEVICE)       # [B, N, L]
attention_mask = batch["attention_mask"].to(DEVICE)   # [B, N, L]
agent_valid_b  = batch["agent_valid"].to(DEVICE)      # [B, N]

print(f"  input_ids shape: {input_ids.shape}")
print(f"  train_mask: {batch['train_mask'].tolist()}")

with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
    pred = model(input_ids, attention_mask, agent_valid_b)
    print(f"  model output shape: {pred.shape}  expected: {list(input_ids.shape[:2]) + [6, 60, 2]}")
    assert pred.shape == (input_ids.shape[0], input_ids.shape[1], 6, 60, 2), \
        f"FAIL: unexpected output shape {pred.shape}"

    loss_out = loss_fn(pred, {k: v.to(DEVICE) for k, v in batch.items()
                               if isinstance(v, torch.Tensor)})
    loss_val = loss_out["loss"]

print(f"  training loss: {loss_val.item():.6f}")
assert not torch.isnan(loss_val), "FAIL: NaN loss on real batch"
assert not torch.isinf(loss_val), "FAIL: Inf loss on real batch"

loss_val.backward()
total_gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
print(f"  total grad norm: {total_gnorm:.4f}")
assert not torch.isnan(total_gnorm), "FAIL: NaN gradient norm"
print("  [PASS] Full forward + backward OK")

print("\n========== ALL TESTS PASSED ==========")
