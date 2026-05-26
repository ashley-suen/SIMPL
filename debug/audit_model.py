"""
Comprehensive model audit script.
Checks: checkpoint, architecture, prompts, GT consistency,
        coordinate frames, prediction quality, and mode collapse.

Usage (from SIMPL_back/):
    python debug/audit_model.py

Output is written to both stdout and debug/audit_report.txt
"""

import os, sys, random, math, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from torch.utils.data import DataLoader

from data_av2.av2_llm_dataset import AV2PromptDataset
from simpl.llm_motion_model import SmolLMMotionPredictor
from simpl.av2_llm_loss import LLMMotionLoss

# ── Config ───────────────────────────────────────────────────────────────────
CKPT_PATH       = "saved_models/20260505-133823_llm_simpl_best2.tar"
VAL_DIR         = "data_av2/features/val"
TRAIN_DIR       = "data_av2/features/train"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
N_AUDIT_SAMPLES = 200      # samples for statistical checks
SEED            = 42
REPORT_PATH     = "debug/audit_report.txt"
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

lines = []
def p(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    lines.append(s)

def section(title):
    bar = "=" * 72
    p(f"\n{bar}\n  {title}\n{bar}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════
section("1. CHECKPOINT INSPECTION")

assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
p(f"Keys        : {list(ckpt.keys())}")
p(f"Saved epoch : {ckpt.get('epoch', 'N/A')}")

sd = ckpt["state_dict"]
p(f"State dict  : {len(sd)} tensors")

# Check for NaN/Inf in weights
nan_keys = [k for k, v in sd.items() if torch.isnan(v).any()]
inf_keys = [k for k, v in sd.items() if torch.isinf(v).any()]
p(f"NaN weights : {len(nan_keys)}  {'PROBLEM: ' + str(nan_keys[:3]) if nan_keys else 'OK'}")
p(f"Inf weights : {len(inf_keys)}  {'PROBLEM: ' + str(inf_keys[:3]) if inf_keys else 'OK'}")

# Weight magnitude stats for key modules
for prefix, label in [("mlp_head", "MLP head"), ("social_attn", "Social attn"),
                       ("llm.norm", "LLM final norm")]:
    vals = torch.cat([v.float().flatten() for k, v in sd.items() if k.startswith(prefix)])
    p(f"  {label:20s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
      f"min={vals.min():.4f}  max={vals.max():.4f}  (n={vals.numel():,})")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
section("2. MODEL ARCHITECTURE & PARAMETER AUDIT")

model = SmolLMMotionPredictor(unfreeze_last_n_layers=1, device=DEVICE,
                               use_flash_attn=False, dtype=torch.float32)
model.load_state_dict(sd)
model.eval()

trainable = [(n, p_) for n, p_ in model.named_parameters() if p_.requires_grad]
frozen    = [(n, p_) for n, p_ in model.named_parameters() if not p_.requires_grad]
p(f"Trainable params : {sum(p_.numel() for _,p_ in trainable):,}")
p(f"Frozen    params : {sum(p_.numel() for _,p_ in frozen):,}")
p(f"Trainable layer names (first 10):")
for n, _ in trainable[:10]:
    p(f"  {n}")

# Dtype consistency
dtypes = {p_.dtype for p_ in model.parameters()}
p(f"Parameter dtypes : {dtypes}  {'OK (uniform)' if len(dtypes)==1 else 'MIXED - may cause issues'}")

# Dummy forward pass
p("\nDummy forward pass (B=2, N=6, L=128)...")
B, N, L = 2, 6, 128
with torch.no_grad():
    dummy_ids  = torch.randint(0, 49000, (B, N, L)).to(DEVICE)
    dummy_mask = torch.ones(B, N, L, dtype=torch.long).to(DEVICE)
    dummy_valid = torch.ones(B, N, dtype=torch.bool).to(DEVICE)
    dummy_valid[0, -1] = False   # one dummy slot

    out = model(dummy_ids, dummy_mask, dummy_valid)

p(f"Output shape     : {out.shape}  (expected [2, 6, 60, 2])")
p(f"Output NaN       : {torch.isnan(out).any().item()}")
p(f"Output Inf       : {torch.isinf(out).any().item()}")
p(f"Output mean/std  : {out.mean():.4f} / {out.std():.4f}")

# Check dummy slot (agent_valid=False) is zeroed by social_attn mask
dummy_slot_norm = out[0, -1].norm().item()
p(f"Dummy slot (invalid agent) output norm: {dummy_slot_norm:.6f} "
  f"{'(near-zero OK)' if dummy_slot_norm < 1.0 else 'WARNING: should be suppressed by mask'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET & PROMPT QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
section("3. DATASET & PROMPT QUALITY")

val_set = AV2PromptDataset(VAL_DIR)
p(f"Val samples : {len(val_set)}")

# Sample N_AUDIT_SAMPLES for statistical analysis
indices = random.sample(range(len(val_set)), min(N_AUDIT_SAMPLES, len(val_set)))

token_lens, n_agents_list, has_truncation = [], [], []
anchor_norms, gt_disp_step_norms = [], []
gt_abs_vs_anchor_cumsum_errors = []
coord_frame_checks = []

for idx in indices:
    s = val_set[idx]

    # Token length analysis
    for ag in range(s["agent_valid"].shape[0]):
        if s["agent_valid"][ag]:
            tlen = s["attention_mask"][ag].sum().item()
            token_lens.append(tlen)
            has_truncation.append(tlen >= val_set.max_len_per_agent)

    n_agents_list.append(s["agent_valid"].sum().item())

    # GT consistency: anchor + cumsum(disp) == abs_pos ?
    for ag in range(s["agent_valid"].shape[0]):
        if not s["agent_valid"][ag]:
            continue
        mask   = s["gt_masks"][ag]                   # [60]
        anchor = s["gt_anchor"][ag]                  # [2]
        disp   = s["gt_trajectories"][ag]            # [60, 2]
        absp   = s["gt_abs_trajectories"][ag]        # [60, 2]

        reconstructed = anchor.unsqueeze(0) + disp.cumsum(0)  # [60, 2]
        err = (reconstructed[mask] - absp[mask]).norm(dim=-1).max().item()
        gt_abs_vs_anchor_cumsum_errors.append(err)

        # Anchor norm (should be near origin in focal frame for focal agent)
        anchor_norms.append(anchor.norm().item())

        # Per-step displacement magnitude (what magnitudes the model must predict)
        step_norms = disp[mask].norm(dim=-1)
        gt_disp_step_norms.extend(step_norms.tolist())

p(f"Token length (per agent):  mean={np.mean(token_lens):.0f}  "
  f"median={np.median(token_lens):.0f}  "
  f"p95={np.percentile(token_lens, 95):.0f}  "
  f"max={np.max(token_lens):.0f} / {val_set.max_len_per_agent}")
p(f"Truncated sequences       : {sum(has_truncation)}/{len(has_truncation)} "
  f"({100*sum(has_truncation)/max(1,len(has_truncation)):.1f}%) "
  f"{'WARNING: losing context!' if sum(has_truncation)/max(1,len(has_truncation)) > 0.1 else 'OK'}")
p(f"Agents per scene           : mean={np.mean(n_agents_list):.1f}  "
  f"min={np.min(n_agents_list)}  max={np.max(n_agents_list)}")

p(f"\nGT consistency (anchor+cumsum==abs): max_err={np.max(gt_abs_vs_anchor_cumsum_errors):.6f} m  "
  f"{'OK' if np.max(gt_abs_vs_anchor_cumsum_errors) < 1e-3 else 'PROBLEM'}")

p(f"Anchor norms (focal frame origin): mean={np.mean(anchor_norms):.3f}  "
  f"max={np.max(anchor_norms):.3f}  "
  f"{'near-zero (correct)' if np.mean(anchor_norms) < 1.0 else 'large - check frame'}")

p(f"GT per-step displacement magnitude:")
p(f"  mean={np.mean(gt_disp_step_norms):.4f} m  "
  f"median={np.median(gt_disp_step_norms):.4f} m  "
  f"p95={np.percentile(gt_disp_step_norms, 95):.4f} m  "
  f"max={np.max(gt_disp_step_norms):.4f} m")
p(f"  => Model must predict steps of this magnitude to achieve low ADE")

# Print one example prompt
sample_0 = val_set[indices[0]]
p(f"\nSample prompt (first 600 chars):\n" + "-"*60)
p(textwrap.fill(sample_0["prompt_text"][:600], width=80))
p("-"*60)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. COORDINATE FRAME CONSISTENCY (focal vs neighbor labels)
# ═══════════════════════════════════════════════════════════════════════════════
section("4. COORDINATE FRAME ANALYSIS")

p("Checking: are neighbor GT labels in the same frame as their prompt?")
p("(focal agent: always consistent; neighbors: prompt=focal frame, GT=own frame)")

# Load a raw pkl to compare
import glob, pandas as pd
pkl_files = glob.glob(os.path.join(VAL_DIR, "**", "*.pkl"), recursive=True) or \
            glob.glob(os.path.join(VAL_DIR, "*.pkl"))
if pkl_files:
    row = pd.read_pickle(pkl_files[0]).iloc[0]
    trajs = row["TRAJS"]
    trajs_pos = trajs["trajs_pos"]
    trajs_ctrs = trajs["trajs_ctrs"]
    trajs_vecs = trajs["trajs_vecs"]
    has_flags = trajs["has_flags"]

    focal_t49 = trajs_pos[0, 49]
    p(f"Focal agent position at t=49 (should be ~origin): {focal_t49}")
    p(f"  => {'CORRECT (focal frame centered at t=49)' if np.linalg.norm(focal_t49) < 2.0 else 'WARNING: not near origin'}")

    if len(trajs_pos) > 1:
        nbr_t49_raw   = trajs_pos[1, 49]       # neighbor in its OWN frame
        nbr_ctr       = trajs_ctrs[1]           # neighbor center in focal frame
        nbr_vec       = trajs_vecs[1]
        theta         = np.arctan2(nbr_vec[1], nbr_vec[0])
        rot           = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta),  np.cos(theta)]])
        nbr_t49_focal = nbr_t49_raw.dot(rot.T) + nbr_ctr

        p(f"\nNeighbor agent 1:")
        p(f"  trajs_pos[1, 49] (own frame) : {nbr_t49_raw}")
        p(f"  ctr (position in focal frame): {nbr_ctr}")
        p(f"  Converted to focal frame     : {nbr_t49_focal}")
        p(f"  GT abs used in training      : {trajs_pos[1, 49]}")
        p(f"\n  FRAME MISMATCH for neighbors: prompt shows focal-frame coords,")
        p(f"  but GT labels use each agent's OWN local frame.")
        p(f"  => Neighbor losses are in wrong frame (focal agent loss is correct).")
        p(f"  => minADE (focal only) is NOT affected, but neighbor training signal is noisy.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PREDICTION QUALITY AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
section("5. INFERENCE: PREDICTION QUALITY (N=" + str(min(N_AUDIT_SAMPLES, len(val_set))) + " samples)")

dl = DataLoader(val_set, batch_size=8, shuffle=False,
                num_workers=0, collate_fn=None)

pred_disp_mags, gt_disp_mags = [], []
pred_abs_mags = []
ade_list, fde_list = [], []
agent_pred_diffs = []     # difference between focal and neighbour prediction
zero_pred_count = 0
total_focal = 0

loss_fn = LLMMotionLoss(device=DEVICE, pos_loss_weight=0.05)

n_done = 0
with torch.no_grad():
    for batch in dl:
        if n_done >= N_AUDIT_SAMPLES:
            break

        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        valid = batch["agent_valid"].to(DEVICE)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(ids, mask, valid)   # [B, N, 60, 2]
        pred = pred.float().cpu()

        anchor  = batch["gt_anchor"].float()          # [B, N, 2]
        gt_disp = batch["gt_trajectories"].float()    # [B, N, 60, 2]
        gt_abs  = batch["gt_abs_trajectories"].float()# [B, N, 60, 2]
        gt_mask = batch["gt_masks"]                   # [B, N, 60]
        ag_v    = batch["agent_valid"]                # [B, N]

        pred_abs = anchor.unsqueeze(2) + pred.cumsum(dim=2)  # [B, N, 60, 2]

        B_ = pred.shape[0]
        for b in range(B_):
            # Focal agent (index 0)
            fmask = gt_mask[b, 0]
            if fmask.sum() == 0:
                continue
            total_focal += 1

            # Predicted displacement magnitudes
            pm = pred[b, 0][fmask].norm(dim=-1)
            gm = gt_disp[b, 0][fmask].norm(dim=-1)
            pred_disp_mags.extend(pm.tolist())
            gt_disp_mags.extend(gm.tolist())

            # Absolute position magnitude (drift from anchor)
            pred_abs_mags.extend(pred_abs[b, 0][fmask].norm(dim=-1).tolist())

            # ADE / FDE
            l2 = (pred_abs[b, 0] - gt_abs[b, 0]).norm(dim=-1)
            valid_l2 = l2[fmask]
            ade_list.append(valid_l2.mean().item())
            fde_list.append(valid_l2[-1].item())

            # Near-zero prediction check
            if pm.mean().item() < 0.01:
                zero_pred_count += 1

            # Mode collapse: compare focal vs first neighbour prediction
            if ag_v[b, 1]:
                diff = (pred[b, 0] - pred[b, 1]).norm(dim=-1).mean().item()
                agent_pred_diffs.append(diff)

        n_done += B_

p(f"Predicted displacement magnitude (per step):")
p(f"  mean={np.mean(pred_disp_mags):.4f}  median={np.median(pred_disp_mags):.4f}  "
  f"std={np.std(pred_disp_mags):.4f}  max={np.max(pred_disp_mags):.4f}")
p(f"GT displacement magnitude (per step):")
p(f"  mean={np.mean(gt_disp_mags):.4f}  median={np.median(gt_disp_mags):.4f}  "
  f"std={np.std(gt_disp_mags):.4f}  max={np.max(gt_disp_mags):.4f}")

scale_ratio = np.mean(pred_disp_mags) / max(np.mean(gt_disp_mags), 1e-9)
p(f"\nPred/GT displacement ratio: {scale_ratio:.3f} "
  f"{'(OK)' if 0.5 < scale_ratio < 2.0 else 'WARNING: model under/over-predicts motion'}")

p(f"\nPredicted absolute position drift from anchor (cumsum):")
p(f"  mean={np.mean(pred_abs_mags):.3f}  max={np.max(pred_abs_mags):.3f} m")

p(f"\nminADE (focal agent, this audit set):")
p(f"  mean={np.mean(ade_list):.4f} m  median={np.median(ade_list):.4f} m  "
  f"p90={np.percentile(ade_list, 90):.4f} m")
p(f"minFDE:")
p(f"  mean={np.mean(fde_list):.4f} m  median={np.median(fde_list):.4f} m")

p(f"\nNear-zero predictions (mean step < 0.01m): {zero_pred_count}/{total_focal} focal agents "
  f"{'WARNING: model not moving!' if zero_pred_count / max(1, total_focal) > 0.3 else 'OK'}")

if agent_pred_diffs:
    p(f"\nMode collapse check (focal vs neighbour pred diff):")
    p(f"  mean diff={np.mean(agent_pred_diffs):.4f}  min={np.min(agent_pred_diffs):.6f}")
    p(f"  {'WARNING: near-zero diff = mode collapse' if np.mean(agent_pred_diffs) < 0.01 else 'OK: agents predict different trajectories'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SPEED-CONDITIONED PREDICTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
section("6. SPEED-CONDITIONED ANALYSIS")
p("Does the model predict more motion for faster agents?")

# Reuse stored val samples
speed_buckets = {"stopped(0-0.5m/s)": [], "slow(0.5-5)": [],
                 "medium(5-15)": [], "fast(>15)": []}

for idx in indices[:100]:
    s = val_set[idx]
    batch_1 = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]
               for k, v in s.items()}

    with torch.no_grad():
        pred_1 = model(
            batch_1["input_ids"].to(DEVICE),
            batch_1["attention_mask"].to(DEVICE),
            batch_1["agent_valid"].to(DEVICE)
        ).float().cpu()

    # Estimate focal agent's observed speed from GT disp
    mask_obs = s["gt_masks"][0]
    if mask_obs.sum() == 0:
        continue
    # use gt displacement to estimate speed (m per 0.1s step → m/s)
    gt_d = s["gt_trajectories"][0][mask_obs]  # [T, 2]
    avg_speed = gt_d.norm(dim=-1).mean().item() / 0.1   # assume 10Hz

    pred_mean_step = pred_1[0, 0].norm(dim=-1).mean().item()

    if avg_speed < 0.5:   speed_buckets["stopped(0-0.5m/s)"].append(pred_mean_step)
    elif avg_speed < 5:   speed_buckets["slow(0.5-5)"].append(pred_mean_step)
    elif avg_speed < 15:  speed_buckets["medium(5-15)"].append(pred_mean_step)
    else:                 speed_buckets["fast(>15)"].append(pred_mean_step)

for label, vals in speed_buckets.items():
    if vals:
        p(f"  {label:25s}: n={len(vals):3d}  pred_step_mean={np.mean(vals):.4f} m")
    else:
        p(f"  {label:25s}: n=0")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SOCIAL ATTENTION CONTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
section("7. SOCIAL ATTENTION CONTRIBUTION")
p("Compare prediction with vs without social attention (by zeroing out other agents).")

sample = val_set[indices[0]]
b1 = {k: v.unsqueeze(0).to(DEVICE) if isinstance(v, torch.Tensor) else [v]
      for k, v in sample.items()}

with torch.no_grad():
    pred_full = model(b1["input_ids"], b1["attention_mask"], b1["agent_valid"]).float().cpu()

    # Run with only focal agent valid (mask all others)
    solo_valid = b1["agent_valid"].clone()
    solo_valid[0, 1:] = False
    pred_solo = model(b1["input_ids"], b1["attention_mask"], solo_valid).float().cpu()

focal_diff = (pred_full[0, 0] - pred_solo[0, 0]).norm(dim=-1)
p(f"Focal agent prediction change when neighbours masked out:")
p(f"  Mean step diff = {focal_diff.mean():.6f}  Max step diff = {focal_diff.max():.6f}")
p(f"  {'Social attention has effect (good)' if focal_diff.mean() > 1e-4 else 'WARNING: social attention has NO effect on focal prediction'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY & DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════════
section("8. SUMMARY & DIAGNOSIS")

issues = []
if nan_keys:                            issues.append("NaN weights detected")
if inf_keys:                            issues.append("Inf weights detected")
if sum(has_truncation)/max(1,len(has_truncation)) > 0.1:
                                        issues.append(f">{10}% sequences truncated — losing prompt context")
if np.max(gt_abs_vs_anchor_cumsum_errors) > 1e-3:
                                        issues.append("GT data inconsistency: anchor+cumsum != abs_pos")
if scale_ratio < 0.3:                  issues.append(f"Severe under-prediction: pred/GT ratio={scale_ratio:.2f}")
if scale_ratio > 3.0:                  issues.append(f"Severe over-prediction: pred/GT ratio={scale_ratio:.2f}")
if zero_pred_count / max(1,total_focal) > 0.3:
                                        issues.append("Model predicting near-zero displacement (not learning motion)")
if agent_pred_diffs and np.mean(agent_pred_diffs) < 0.01:
                                        issues.append("Mode collapse: all agents same prediction")
if focal_diff.mean() < 1e-4:           issues.append("Social attention has no effect on focal agent")

if issues:
    p("ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        p(f"  [{i}] {issue}")
else:
    p("No critical issues found. Model architecture and data pipeline appear healthy.")

p(f"\nKey metric summary:")
p(f"  Focal minADE  (this audit): {np.mean(ade_list):.4f} m")
p(f"  Pred/GT disp ratio        : {scale_ratio:.3f}")
p(f"  Token truncation rate     : {100*sum(has_truncation)/max(1,len(has_truncation)):.1f}%")
p(f"  Social attn effect        : {focal_diff.mean():.6f} m")

# ── Write report ──────────────────────────────────────────────────────────────
os.makedirs("debug", exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nReport saved to {REPORT_PATH}")
