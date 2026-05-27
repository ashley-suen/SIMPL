"""
Export one val scene's predicted displacements (all 6 modes, all 60 steps)
and GT absolute positions to a human-readable text file.

Run from SIMPL_back root:
    python debug/export_prediction_report.py
"""
import sys, os, torch, numpy as np, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_av2.av2_llm_dataset import AV2PromptDataset
from simpl.llm_motion_model import SmolLMMotionPredictor

CKPT     = "saved_models/20260526-184549_llm_simpl_best.tar"
DATA_DIR = "data_av2/features/val"
OUT_FILE = "debug/exp3_val_prediction_report2.txt"
SEED     = 7890

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
model = SmolLMMotionPredictor(num_modes=6, device=DEVICE)
ckpt  = torch.load(CKPT, map_location=DEVICE)
state = ckpt.get("state_dict", ckpt)
state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.eval()
print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")

# ── Dataset ───────────────────────────────────────────────────────────────────
ds = AV2PromptDataset(DATA_DIR, max_len_per_agent=1536)
random.seed(SEED)
idx = random.randint(0, len(ds) - 1)
print(f"Selected scene index: {idx}  (total val scenes: {len(ds)})")

sample = ds[idx]
ids    = sample["input_ids"].unsqueeze(0).to(DEVICE)
amask  = sample["attention_mask"].unsqueeze(0).to(DEVICE)
av     = sample["agent_valid"].unsqueeze(0).to(DEVICE)
tm     = sample["train_mask"]           # [N]
gt_abs = sample["gt_abs_trajectories"]  # [N, 60, 2]
gt_m   = sample["gt_masks"]             # [N, 60]
anchor = sample["gt_anchor"]            # [N, 2]

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    pred_disp = model(ids, amask, av)[0].cpu()  # [N, 6, 60, 2]

pred_abs = anchor.unsqueeze(1).unsqueeze(1) + pred_disp.cumsum(dim=-2)  # [N, 6, 60, 2]
N = pred_disp.shape[0]

# ── Build report ──────────────────────────────────────────────────────────────
SEP = "=" * 90
sep = "-" * 90

lines = []
lines.append(SEP)
lines.append("Exp3 Best Checkpoint — Val Prediction Report")
lines.append(f"Checkpoint : {CKPT}  (epoch {ckpt.get('epoch', '?')})")
lines.append(f"Val scene  : index {idx} of {len(ds)}")
lines.append(f"Device     : {DEVICE}")
lines.append("Timestep   : 0.1 s per step  |  Horizon: 60 steps = 6.0 s")
lines.append("K          : 6 prediction modes")
lines.append("Coords     : agent-centric (anchor = agent position at observation end)")
lines.append(SEP)

valid_count = int(av[0].sum().item())
lines.append(f"Total agent slots: {N}  |  Valid (agent_valid=True): {valid_count}")
lines.append("")

for n in range(N):
    if not av[0, n].item():
        continue

    is_train = bool(tm[n].item())
    label    = "FOCAL / SCORE  (supervised by loss)" if is_train else "UNSCORED / AV  (context only, no loss)"
    anc_x, anc_y = anchor[n].tolist()
    mask = gt_m[n].bool()
    n_valid_gt = int(mask.sum().item())

    lines.append("")
    lines.append(SEP)
    lines.append(f"AGENT {n}   [{label}]")
    lines.append(f"Anchor (t=0 absolute position): x={anc_x:.4f}  y={anc_y:.4f}")
    lines.append(f"GT valid future timesteps: {n_valid_gt} / 60")
    lines.append(SEP)

    # ── GT absolute positions ─────────────────────────────────────────────────
    lines.append("")
    lines.append("[ Ground Truth — Absolute Positions ]")
    lines.append(f"  {'step':>4}  {'GT_x':>10}  {'GT_y':>10}  {'valid':>5}")
    lines.append(f"  {'----':>4}  {'----------':>10}  {'----------':>10}  {'-----':>5}")
    for t in range(60):
        vld = "YES" if mask[t].item() else "NO "
        gx, gy = gt_abs[n, t].tolist()
        lines.append(f"  {t:>4}  {gx:>10.4f}  {gy:>10.4f}  {vld:>5}")

    # ── Predicted per-step displacements + cumulative absolute ────────────────
    lines.append("")
    lines.append("[ Predicted Per-Step Displacements and Cumulative Absolute Positions ]")
    lines.append("  (disp_x, disp_y = model output at each step;")
    lines.append("   abs_x, abs_y   = anchor + cumsum of displacements up to this step)")
    lines.append("")

    for k in range(6):
        lines.append(sep)
        lines.append(f"  Mode {k}")
        lines.append(sep)
        lines.append(f"  {'step':>4}  {'disp_x':>10}  {'disp_y':>10}  {'abs_x':>10}  {'abs_y':>10}")
        lines.append(f"  {'----':>4}  {'----------':>10}  {'----------':>10}  {'----------':>10}  {'----------':>10}")
        for t in range(60):
            dx, dy = pred_disp[n, k, t].tolist()
            ax, ay = pred_abs[n, k, t].tolist()
            lines.append(f"  {t:>4}  {dx:>10.4f}  {dy:>10.4f}  {ax:>10.4f}  {ay:>10.4f}")

    # ── Final position comparison + metrics ───────────────────────────────────
    lines.append("")
    lines.append("[ Final Position Comparison — t=59 ]")
    last_valid_t = int(mask.nonzero(as_tuple=False)[-1].item()) if mask.any() else 59
    gx59, gy59 = gt_abs[n, last_valid_t].tolist()
    lines.append(f"  GT  (last valid t={last_valid_t}): ({gx59:.4f}, {gy59:.4f})")
    for k in range(6):
        px, py = pred_abs[n, k, last_valid_t].tolist()
        lines.append(f"  Mode {k}            : ({px:.4f}, {py:.4f})")

    if mask.any() and is_train:
        lines.append("")
        lines.append("[ Per-Mode Error Metrics (train_mask agents only) ]")
        gt_valid_pts = gt_abs[n][mask]   # [valid_T, 2]
        for k in range(6):
            pred_valid_pts = pred_abs[n, k][mask]   # [valid_T, 2]
            err = (pred_valid_pts - gt_valid_pts).norm(dim=-1)
            ade = err.mean().item()
            fde = (pred_abs[n, k, last_valid_t] - gt_abs[n, last_valid_t]).norm().item()
            lines.append(f"  Mode {k}:  ADE = {ade:.4f} m   FDE = {fde:.4f} m")
        # minADE / minFDE
        ades = []
        fdes = []
        for k in range(6):
            pred_valid_pts = pred_abs[n, k][mask]
            err = (pred_valid_pts - gt_valid_pts).norm(dim=-1)
            ades.append(err.mean().item())
            fdes.append((pred_abs[n, k, last_valid_t] - gt_abs[n, last_valid_t]).norm().item())
        lines.append(f"  minADE (best of 6): {min(ades):.4f} m")
        lines.append(f"  minFDE (best of 6): {min(fdes):.4f} m")

lines.append("")
lines.append(SEP)
lines.append("END OF REPORT")
lines.append(SEP)

# ── Write file ────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nReport saved to: {OUT_FILE}")
print(f"Total lines: {len(lines)}")
