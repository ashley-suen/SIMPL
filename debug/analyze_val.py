
"""
Validation diagnostic for Exp3 best checkpoint.
Analyzes:
  1. Agent coverage   — are focal/scored/unscored agents all being processed?
  2. Prediction sanity — NaN/Inf, coordinate range, displacement plausibility
  3. Mode diversity   — are K=6 modes actually diverse or collapsed?
  4. Per-type ADE/FDE — breakdown by train_mask vs non-train_mask agents
  5. Trajectory plots — GT vs predicted for sampled scenes

Run from SIMPL_back root:
    python debug/analyze_val.py --ckpt saved_models/20260526-184549_llm_simpl_best.tar
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader

from data_av2.av2_llm_dataset import AV2PromptDataset
from simpl.llm_motion_model import SmolLMMotionPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "debug/val_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",       required=True, type=str)
parser.add_argument("--data_dir",   default="data_av2/features/val")
parser.add_argument("--num_scenes", default=200,  type=int,
                    help="Scenes to evaluate for statistics")
parser.add_argument("--plot_scenes",default=12,   type=int,
                    help="Scenes to visualise")
parser.add_argument("--batch_size", default=4,    type=int)
parser.add_argument("--num_modes",  default=6,    type=int)
args = parser.parse_args()

# ── Load model ───────────────────────────────────────────────────────────────
print(f"\n[1/5] Loading checkpoint: {args.ckpt}")
model = SmolLMMotionPredictor(num_modes=args.num_modes, device=DEVICE)
ckpt  = torch.load(args.ckpt, map_location=DEVICE)
state = ckpt.get("state_dict", ckpt)
# strip DDP "module." and torch.compile "_orig_mod." prefixes
state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.eval()
print(f"   Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")

# ── Collate with dynamic padding ──────────────────────────────────────────────
def dynamic_pad_collate(batch):
    max_len = max(item["attention_mask"].sum(dim=-1).max().item() for item in batch)
    max_len = int(max_len)
    for item in batch:
        item["input_ids"]      = item["input_ids"][:, :max_len]
        item["attention_mask"] = item["attention_mask"][:, :max_len]
    return torch.utils.data.dataloader.default_collate(batch)

# ── Dataset ───────────────────────────────────────────────────────────────────
print(f"[2/5] Loading val dataset from: {args.data_dir}")
ds = AV2PromptDataset(args.data_dir, max_len_per_agent=1536)
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                num_workers=0, collate_fn=dynamic_pad_collate)
print(f"   Val samples: {len(ds)}")

# ── Inference loop ────────────────────────────────────────────────────────────
print(f"[3/5] Running inference on {args.num_scenes} scenes …")

records = []   # list of dicts, one per valid agent slot
scene_preds = []  # for plotting

n_scenes = 0
with torch.no_grad():
    for batch in dl:
        if n_scenes >= args.num_scenes:
            break

        ids   = batch["input_ids"].to(DEVICE)           # [B, N, L]
        amask = batch["attention_mask"].to(DEVICE)       # [B, N, L]
        av    = batch["agent_valid"].to(DEVICE)          # [B, N]
        tm    = batch["train_mask"].to(DEVICE)           # [B, N]
        gt_abs= batch["gt_abs_trajectories"].to(DEVICE)  # [B, N, 60, 2]
        gt_m  = batch["gt_masks"].to(DEVICE)             # [B, N, 60]
        anchor= batch["gt_anchor"].to(DEVICE)            # [B, N, 2]

        pred_disp = model(ids, amask, av)                # [B, N, K, 60, 2]
        B, N, K, T, _ = pred_disp.shape

        # absolute predictions
        pred_abs = anchor.unsqueeze(2).unsqueeze(2) + pred_disp.cumsum(dim=-2)
        # [B, N, K, 60, 2]

        for b in range(B):
            scene_rec = {"pred_abs": [], "gt_abs": [], "gt_mask": [],
                         "train_mask": [], "agent_valid": []}
            for n in range(N):
                is_valid = av[b, n].item()
                is_train = tm[b, n].item()
                has_gt   = gt_m[b, n].any().item()

                # ── sanity checks ─────────────────────────────
                p = pred_abs[b, n]   # [K, 60, 2]
                has_nan = torch.isnan(p).any().item()
                has_inf = torch.isinf(p).any().item()

                # ── mode diversity: std across K modes ────────
                mode_std = p.std(dim=0).mean().item()  # avg std per timestep

                # ── ADE / FDE for train_mask agents ──────────
                ade = fde = None
                if is_train and has_gt:
                    g  = gt_abs[b, n]            # [60, 2]
                    mk = gt_m[b, n].bool()        # [60]
                    if mk.any():
                        err = (p[:, mk, :] - g[mk].unsqueeze(0)).norm(dim=-1)
                        # err: [K, valid_T]
                        ade = err.mean(dim=-1).min().item()
                        fde_idx = mk.nonzero(as_tuple=False)[-1].item()
                        fde = (p[:, fde_idx, :] - g[fde_idx]).norm(dim=-1).min().item()

                # ── displacement magnitude check ───────────────
                # largest single-step jump in best mode (mode with highest avg xy std over time)
                best_mode = p.std(dim=1).mean(dim=-1).argmax().item()  # std over T → [K,2], mean xy → [K], argmax → int in [0,K)
                step_disp = pred_disp[b, n, best_mode].norm(dim=-1)   # [60]
                max_jump  = step_disp.max().item()

                rec = dict(
                    is_valid=is_valid, is_train=is_train, has_gt=has_gt,
                    has_nan=has_nan,   has_inf=has_inf,
                    mode_std=mode_std, ade=ade, fde=fde,
                    max_jump=max_jump,
                )
                records.append(rec)
                scene_rec["pred_abs"].append(p.cpu().numpy())
                scene_rec["gt_abs"].append(gt_abs[b, n].cpu().numpy())
                scene_rec["gt_mask"].append(gt_m[b, n].cpu().numpy())
                scene_rec["train_mask"].append(is_train)
                scene_rec["agent_valid"].append(is_valid)

            if n_scenes < args.plot_scenes:
                scene_preds.append(scene_rec)
            n_scenes += 1

print(f"   Processed {n_scenes} scenes, {len(records)} agent slots total.")

# ── Statistics ────────────────────────────────────────────────────────────────
print("\n[4/5] Computing statistics …\n")

def filt(key, cond):
    return [r[key] for r in records if cond(r) and r[key] is not None]

valid_recs  = [r for r in records if r["is_valid"]]
train_recs  = [r for r in records if r["is_train"]]
nontrn_recs = [r for r in records if r["is_valid"] and not r["is_train"]]

print("=" * 60)
print("AGENT SLOT COVERAGE")
print(f"  Total slots        : {len(records)}")
print(f"  agent_valid=True   : {len(valid_recs)}")
print(f"  train_mask=True    : {len(train_recs)}")
print(f"  valid but non-train: {len(nontrn_recs)}")

# NaN / Inf
nan_cnt = sum(r["has_nan"] for r in valid_recs)
inf_cnt = sum(r["has_inf"] for r in valid_recs)
print(f"\nSANITY (valid agents)")
print(f"  NaN in predictions : {nan_cnt} / {len(valid_recs)}")
print(f"  Inf in predictions : {inf_cnt} / {len(valid_recs)}")

# Mode diversity
std_train    = [r["mode_std"] for r in train_recs]
std_nontrain = [r["mode_std"] for r in nontrn_recs]
print(f"\nMODE DIVERSITY (std across K=6 modes, per-step avg)")
print(f"  train_mask agents  : mean={np.mean(std_train):.4f}  "
      f"median={np.median(std_train):.4f}  p5={np.percentile(std_train,5):.4f}")
if std_nontrain:
    print(f"  non-train agents   : mean={np.mean(std_nontrain):.4f}  "
          f"median={np.median(std_nontrain):.4f}  p5={np.percentile(std_nontrain,5):.4f}")

# ADE / FDE
ade_vals = [r["ade"] for r in train_recs if r["ade"] is not None]
fde_vals = [r["fde"] for r in train_recs if r["fde"] is not None]
print(f"\nPREDICTION QUALITY (train_mask agents with GT)")
print(f"  minADE: mean={np.mean(ade_vals):.4f}  "
      f"median={np.median(ade_vals):.4f}  p90={np.percentile(ade_vals,90):.4f}")
print(f"  minFDE: mean={np.mean(fde_vals):.4f}  "
      f"median={np.median(fde_vals):.4f}  p90={np.percentile(fde_vals,90):.4f}")

# Max single-step jump
jumps = [r["max_jump"] for r in valid_recs]
print(f"\nDISPLACEMENT SANITY (max single-step jump in best mode)")
print(f"  mean={np.mean(jumps):.3f}  p50={np.median(jumps):.3f}  "
      f"p95={np.percentile(jumps,95):.3f}  max={np.max(jumps):.3f}")
large_jumps = sum(j > 5.0 for j in jumps)
print(f"  Agents with jump > 5m/step: {large_jumps} / {len(jumps)}")

# ── Visualise scenes ───────────────────────────────────────────────────────────
print(f"\n[5/5] Plotting {len(scene_preds)} scenes to {OUT_DIR}/ …")

COLORS_K = cm.tab10(np.linspace(0, 0.6, 6))

for s_idx, scene in enumerate(scene_preds):
    N = len(scene["pred_abs"])
    fig, axes = plt.subplots(1, N, figsize=(4 * N, 5))
    if N == 1:
        axes = [axes]
    fig.suptitle(f"Scene {s_idx:03d}", fontsize=11)

    for n, ax in enumerate(axes):
        is_train = scene["train_mask"][n]
        is_valid = scene["agent_valid"][n]
        pred     = scene["pred_abs"][n]    # [K, 60, 2]
        gt       = scene["gt_abs"][n]      # [60, 2]
        mask     = scene["gt_mask"][n].astype(bool)  # [60]

        label = "focal/score" if is_train else ("unscore/av" if is_valid else "PADDING")
        ax.set_title(f"Agent {n}\n[{label}]", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        if not is_valid:
            ax.text(0.5, 0.5, "PADDING\n(dummy slot)", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            continue

        # GT trajectory
        if mask.any():
            ax.plot(gt[mask, 0], gt[mask, 1], "k-", lw=2, label="GT", zorder=5)
            ax.plot(gt[mask, 0][-1], gt[mask, 1][-1], "k*", ms=8, zorder=6)

        # K predicted modes
        for k in range(pred.shape[0]):
            alpha = 0.8 if k == 0 else 0.4
            ax.plot(pred[k, :, 0], pred[k, :, 1],
                    color=COLORS_K[k], lw=1.2, alpha=alpha,
                    label=f"mode {k}" if n == 0 else "")

        # anchor (start point)
        if mask.any():
            ax.plot(gt[mask, 0][0], gt[mask, 1][0], "go", ms=6, zorder=7, label="start")

        ax.tick_params(labelsize=6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"scene_{s_idx:03d}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

print(f"   Saved to {OUT_DIR}/scene_*.png")
print("\n========== ANALYSIS COMPLETE ==========")
