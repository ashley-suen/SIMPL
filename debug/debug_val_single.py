"""
Single-sample debug validation script.
Loads the best checkpoint, picks one random val sample, and prints:
  - raw model output (predicted displacements / absolute positions)
  - GT values
  - per-step L2 error
  - minADE / minFDE for this sample
"""

import os, random, sys
import torch
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data_av2.av2_llm_dataset import AV2PromptDataset
from simpl.llm_motion_model import SmolLMMotionPredictor

# ── Config ──────────────────────────────────────────────────────────────────
CKPT_PATH   = "../saved_models/20260501-022610_llm_simpl_best.tar"
VAL_DIR     = "../data_av2/features/val"
SEED        = 46          # change to pick different sample
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── 1. Load dataset, pick one sample ────────────────────────────────────────
print(f"Loading val dataset from {VAL_DIR} ...")
val_set = AV2PromptDataset(VAL_DIR)
idx = random.randint(0, len(val_set) - 1)
print(f"Selected sample index: {idx} / {len(val_set)-1}\n")

sample = val_set[idx]

# Collate into batch-of-1
def to_batch(s):
    out = {}
    for k, v in s.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        else:
            out[k] = [v]
    return out

batch = to_batch(sample)

# ── 2. Print prompt (first 800 chars) ───────────────────────────────────────
print("=" * 70)
print("PROMPT (first 800 chars):")
print(batch["prompt_text"][0][:800])
print("=" * 70)

# ── 3. Load model ───────────────────────────────────────────────────────────
print(f"\nLoading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
print(f"  Checkpoint epoch : {ckpt.get('epoch', 'N/A')}")
print(f"  Checkpoint keys  : {list(ckpt.keys())}")

model = SmolLMMotionPredictor(unfreeze_last_n_layers=1).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("  Model loaded.\n")

# ── 4. Forward pass ─────────────────────────────────────────────────────────
input_ids      = batch["input_ids"].to(DEVICE)      # [1, N, L]
attention_mask = batch["attention_mask"].to(DEVICE)  # [1, N, L]
agent_valid    = batch["agent_valid"].to(DEVICE)     # [1, N]

with torch.no_grad():
    pred_disp = model(input_ids, attention_mask, agent_valid)  # [1, N, 60, 2]

pred_disp = pred_disp.float().cpu()  # [1, N, 60, 2]

# ── 5. Reconstruct absolute positions from predicted displacements ───────────
anchor    = batch["gt_anchor"].float()           # [1, N, 2]
pred_abs  = anchor.unsqueeze(2) + pred_disp.cumsum(dim=2)   # [1, N, 60, 2]

gt_disp   = batch["gt_trajectories"].float()     # [1, N, 60, 2]
gt_abs    = batch["gt_abs_trajectories"].float() # [1, N, 60, 2]
gt_mask   = batch["gt_masks"]                    # [1, N, 60]  bool
ag_valid  = batch["agent_valid"]                 # [1, N]      bool

N = pred_disp.shape[1]

print("=" * 70)
print("PER-AGENT SUMMARY  (focal agent = index 0)")
print("=" * 70)

for a in range(N):
    if not ag_valid[0, a]:
        continue

    mask = gt_mask[0, a]          # [60] bool
    valid_steps = mask.sum().item()
    if valid_steps == 0:
        continue

    # ── displacement check ──────────────────────────────────────────────────
    pred_d = pred_disp[0, a]      # [60, 2]
    gt_d   = gt_disp[0, a]        # [60, 2]
    disp_err = (pred_d - gt_d).norm(dim=-1)  # [60]

    # ── absolute position check ─────────────────────────────────────────────
    pred_p = pred_abs[0, a]       # [60, 2]
    gt_p   = gt_abs[0, a]         # [60, 2]
    pos_err = (pred_p - gt_p).norm(dim=-1)   # [60]

    valid_pos_err = pos_err[mask]
    ade = valid_pos_err.mean().item()
    fde = valid_pos_err[-1].item()

    label = "FOCAL" if a == 0 else f"NBR{a}"
    print(f"\n[Agent {a} | {label}]  valid steps={valid_steps}")

    # Show first 5 steps of GT vs predicted displacement
    print(f"  GT   disp (t=1..5): {gt_d[:5].tolist()}")
    print(f"  PRED disp (t=1..5): {[[ round(x,4) for x in step] for step in pred_d[:5].tolist()]}")

    # Show first 5 steps of GT vs predicted absolute position
    print(f"  GT   abs  (t=1..5): {gt_p[:5].tolist()}")
    print(f"  PRED abs  (t=1..5): {[[ round(x,4) for x in step] for step in pred_p[:5].tolist()]}")

    # Per-step L2 error (absolute positions), print every 10 steps
    print(f"  Pos L2 err at t=1,10,20,30,40,50,60:")
    for t in [0, 9, 19, 29, 39, 49, 59]:
        valid = mask[t].item()
        print(f"    t={t+1:2d}: {pos_err[t]:.4f} m  {'(valid)' if valid else '(masked)'}")

    print(f"  >> ADE={ade:.4f} m | FDE={fde:.4f} m")

# ── 6. Quick sanity: are predictions all the same (model not learning)? ──────
print("\n" + "=" * 70)
print("SANITY CHECKS")
print("=" * 70)

focal_pred = pred_disp[0, 0]  # [60, 2]
print(f"Focal pred disp  — mean: {focal_pred.mean():.6f}  std: {focal_pred.std():.6f}  "
      f"min: {focal_pred.min():.4f}  max: {focal_pred.max():.4f}")

focal_abs = pred_abs[0, 0]
print(f"Focal pred abs   — mean: {focal_abs.mean():.6f}  std: {focal_abs.std():.6f}  "
      f"min: {focal_abs.min():.4f}  max: {focal_abs.max():.4f}")

gt_focal_abs = gt_abs[0, 0]
print(f"Focal GT   abs   — mean: {gt_focal_abs.mean():.6f}  std: {gt_focal_abs.std():.6f}  "
      f"min: {gt_focal_abs.min():.4f}  max: {gt_focal_abs.max():.4f}")

# Check if all agents' predictions are identical (degenerate output)
if N > 1:
    diff_01 = (pred_disp[0, 0] - pred_disp[0, 1]).abs().max().item()
    print(f"\nMax diff between agent0 and agent1 predictions: {diff_01:.6f} "
          f"{'<-- IDENTICAL (degenerate!)' if diff_01 < 1e-5 else '(different, good)'}")

print("\nDone.")
