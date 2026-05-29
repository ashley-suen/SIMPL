"""
Export one val scene's full diagnostic report for Exp4.
Captures intermediate representations at each stage of the pipeline:

  input_ids (tokens)
      ↓
  SmolLM-135M  →  last_hidden_state [B*N, L, H]
      ↓  masked mean-pool
  pooled_repr  [N, H=576]          ← LLM output fed into GRU (shown here)
      ↓  Social Attention
  agent_repr   [N, H=576]          ← after cross-agent context
      ↓  GRU Decoder
  pred_disp    [N, K=6, T=60, 2]

Run from SIMPL_back root:
    python debug/export_exp4_report.py
"""
import sys, os, torch, numpy as np, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_av2.av2_llm_dataset import AV2PromptDataset
from simpl.llm_motion_model import SmolLMMotionPredictor

CKPT     = "saved_models/exp4_20260528-120014_llm_simpl_best.tar"
DATA_DIR = "data_av2/features/val"
OUT_FILE = "debug/exp4_val_prediction_report.txt"
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
ids    = sample["input_ids"].unsqueeze(0).to(DEVICE)       # [1, N, L]
amask  = sample["attention_mask"].unsqueeze(0).to(DEVICE)  # [1, N, L]
av     = sample["agent_valid"].unsqueeze(0).to(DEVICE)     # [1, N]
tm     = sample["train_mask"]
gt_abs = sample["gt_abs_trajectories"]  # [N, 60, 2]
gt_m   = sample["gt_masks"]             # [N, 60]
anchor = sample["gt_anchor"]            # [N, 2]

# ── Step-by-step forward to capture intermediates ─────────────────────────────
with torch.no_grad():
    B, N, L = ids.shape
    ids_flat  = ids.view(B * N, L)
    mask_flat = amask.view(B * N, L)

    # Stage 1: LLM
    llm_out = model.llm(input_ids=ids_flat, attention_mask=mask_flat)
    hidden  = llm_out.last_hidden_state.float()          # [B*N, L, H]
    H       = hidden.shape[-1]

    # Stage 2: masked mean-pool → pooled LLM output
    mask_exp  = mask_flat.unsqueeze(-1).float()
    pooled    = (hidden * mask_exp).sum(dim=1) / \
                mask_exp.sum(dim=1).clamp(min=1e-9)      # [B*N, H]
    pooled_N  = pooled.view(B, N, H)[0].cpu()            # [N, H]  batch=1

    # Stage 3: apply agent_valid mask
    pooled_masked = pooled_N * av[0].cpu().unsqueeze(-1).float()   # [N, H]

    # Stage 4: social attention
    agent_repr_raw = model.social_attn(
        pooled_masked.unsqueeze(0).to(DEVICE),
        av[0].unsqueeze(0)
    )[0].cpu().float()                                   # [N, H]

    # Stage 5: GRU decoder
    pred_disp = model.gru_decoder(
        agent_repr_raw.to(DEVICE)
    ).cpu()                                              # [N, K, T, 2]

pred_abs = anchor.unsqueeze(1).unsqueeze(1) + pred_disp.cumsum(dim=-2)  # [N, K, 60, 2]
tokenizer = ds.tokenizer

# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_prompt(ids_1d, mask_1d):
    vlen = int(mask_1d.sum().item())
    return tokenizer.decode(ids_1d[:vlen].tolist(), skip_special_tokens=False)

def vec_stats(v):
    """Return stat dict for a 1-D float tensor."""
    return {
        "norm":   v.norm().item(),
        "mean":   v.mean().item(),
        "std":    v.std().item(),
        "min":    v.min().item(),
        "max":    v.max().item(),
        "nonzero": (v.abs() > 1e-6).sum().item(),
    }

def cosine_sim(a, b):
    return (a * b).sum() / (a.norm() * b.norm() + 1e-9)

# ── Build report ──────────────────────────────────────────────────────────────
SEP  = "=" * 90
sep  = "-" * 90
sep2 = "~" * 90

lines = []
lines.append(SEP)
lines.append("Exp4 Best Checkpoint — Full Diagnostic Val Report")
lines.append("Pipeline: tokens → SmolLM-135M → mean-pool → SocialAttn → GRU Decoder")
lines.append(f"Checkpoint : {CKPT}  (epoch {ckpt.get('epoch', '?')})")
lines.append(f"Val scene  : index {idx} of {len(ds)}")
lines.append(f"Device     : {DEVICE}  |  Hidden dim H = {H}")
lines.append(SEP)

valid_agents = [n for n in range(N) if av[0, n].item()]
lines.append(f"Total slots: {N}  |  Valid agents: {len(valid_agents)}")
lines.append("")

# ════════════════════════════════════════════════════════════════════════════
# SECTION A: Cross-agent representation summary (collapse diagnosis)
# ════════════════════════════════════════════════════════════════════════════
lines.append(SEP)
lines.append("SECTION A — LLM REPRESENTATION COLLAPSE DIAGNOSIS")
lines.append("(If all cosine similarities > 0.99 → LLM gives identical reps → model can't")
lines.append(" differentiate agents → all predictions will be similar)")
lines.append(SEP)

# Pooled repr norms
lines.append("")
lines.append("[ A1: Pooled LLM output (before social attention) — L2 norm per agent ]")
lines.append(f"  {'agent':>5}  {'L2 norm':>10}  {'mean':>10}  {'std':>10}  {'nonzero/576':>12}")
lines.append(f"  {'-----':>5}  {'----------':>10}  {'----------':>10}  {'----------':>10}  {'------------':>12}")
for n in valid_agents:
    s = vec_stats(pooled_N[n])
    lines.append(f"  {n:>5}  {s['norm']:>10.4f}  {s['mean']:>10.6f}  "
                 f"{s['std']:>10.4f}  {s['nonzero']:>12}")

# Cosine similarity matrix
lines.append("")
lines.append("[ A2: Pairwise cosine similarity of pooled representations ]")
lines.append("  (1.000 = identical; expected < 0.90 for good differentiation)")
header = "        " + "".join(f"  ag{n:>2}" for n in valid_agents)
lines.append(header)
for i in valid_agents:
    row = f"  ag{i:>2} |"
    for j in valid_agents:
        sim = cosine_sim(pooled_N[i], pooled_N[j]).item()
        row += f"  {sim:>5.3f}"
    lines.append(row)

# Social attention delta
lines.append("")
lines.append("[ A3: Social attention effect — ||agent_repr_post - pooled_pre|| per agent ]")
lines.append("  (near 0 → social attention not changing representations)")
lines.append(f"  {'agent':>5}  {'delta_norm':>12}  {'cos_sim(pre,post)':>18}")
lines.append(f"  {'-----':>5}  {'------------':>12}  {'------------------':>18}")
for n in valid_agents:
    delta = (agent_repr_raw[n] - pooled_masked[n]).norm().item()
    sim   = cosine_sim(agent_repr_raw[n], pooled_masked[n]).item()
    lines.append(f"  {n:>5}  {delta:>12.4f}  {sim:>18.6f}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION B: Per-agent full detail
# ════════════════════════════════════════════════════════════════════════════
for n in valid_agents:
    is_train   = bool(tm[n].item())
    label      = "FOCAL/SCORE (loss supervised)" if is_train else "UNSCORED/AV (context only)"
    anc_x, anc_y = anchor[n].tolist()
    mask       = gt_m[n].bool()
    valid_len  = int(sample["attention_mask"][n].sum().item())

    lines.append("")
    lines.append(SEP)
    lines.append(f"AGENT {n}   [{label}]")
    lines.append(f"Anchor: ({anc_x:.4f}, {anc_y:.4f})  |  GT valid steps: {mask.sum().item()}/60")
    lines.append(SEP)

    # ── B1. Prompt ───────────────────────────────────────────────────────────
    lines.append("")
    lines.append("[ B1: INPUT PROMPT ]")
    lines.append(sep2)
    prompt_text = decode_prompt(sample["input_ids"][n], sample["attention_mask"][n])
    for line in prompt_text.split("\n"):
        while len(line) > 88:
            lines.append("  " + line[:88])
            line = line[88:]
        lines.append("  " + line)
    lines.append(sep2)
    lines.append(f"  Valid tokens: {valid_len} / {sample['input_ids'][n].shape[0]}  "
                 f"({valid_len / sample['input_ids'][n].shape[0] * 100:.1f}% utilization)")

    # ── B2. LLM output (pooled) ──────────────────────────────────────────────
    lines.append("")
    lines.append("[ B2: LLM OUTPUT — pooled representation (mean-pool over valid tokens) ]")
    lines.append(f"  Vector shape: [{H}]  (SmolLM-135M hidden dim)")
    s = vec_stats(pooled_N[n])
    lines.append(f"  L2 norm  : {s['norm']:.4f}")
    lines.append(f"  Mean     : {s['mean']:.6f}")
    lines.append(f"  Std      : {s['std']:.4f}")
    lines.append(f"  Min      : {s['min']:.4f}")
    lines.append(f"  Max      : {s['max']:.4f}")
    lines.append(f"  Nonzero  : {s['nonzero']} / {H}")
    lines.append(f"  First 16 dims: {pooled_N[n, :16].tolist()}")
    lines.append(f"  Last  16 dims: {pooled_N[n, -16:].tolist()}")

    # ── B3. After social attention ───────────────────────────────────────────
    lines.append("")
    lines.append("[ B3: AFTER SOCIAL ATTENTION — representation fed into GRU ]")
    s2 = vec_stats(agent_repr_raw[n])
    lines.append(f"  L2 norm  : {s2['norm']:.4f}  (vs pooled: {s['norm']:.4f})")
    lines.append(f"  Mean     : {s2['mean']:.6f}")
    lines.append(f"  Std      : {s2['std']:.4f}")
    delta = (agent_repr_raw[n] - pooled_masked[n]).norm().item()
    sim   = cosine_sim(agent_repr_raw[n], pooled_masked[n]).item()
    lines.append(f"  Delta from pooled: ||diff|| = {delta:.4f}  cos_sim = {sim:.6f}")
    lines.append(f"  First 16 dims: {agent_repr_raw[n, :16].tolist()}")

    # ── B4. GT ───────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("[ B4: GROUND TRUTH — Absolute Positions ]")
    lines.append(f"  {'step':>4}  {'GT_x':>10}  {'GT_y':>10}  {'valid':>5}")
    lines.append(f"  {'----':>4}  {'----------':>10}  {'----------':>10}  {'-----':>5}")
    for t in range(60):
        vld = "YES" if mask[t].item() else "NO "
        gx, gy = gt_abs[n, t].tolist()
        lines.append(f"  {t:>4}  {gx:>10.4f}  {gy:>10.4f}  {vld:>5}")

    # ── B5. Predictions ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("[ B5: PREDICTED DISPLACEMENTS & ABSOLUTE POSITIONS ]")
    for k in range(6):
        lines.append(sep)
        lines.append(f"  Mode {k}")
        lines.append(f"  {'step':>4}  {'disp_x':>10}  {'disp_y':>10}  {'abs_x':>10}  {'abs_y':>10}")
        lines.append(f"  {'----':>4}  {'----------':>10}  {'----------':>10}  {'----------':>10}  {'----------':>10}")
        for t in range(60):
            dx, dy = pred_disp[n, k, t].tolist()
            ax, ay = pred_abs[n, k, t].tolist()
            lines.append(f"  {t:>4}  {dx:>10.5f}  {dy:>10.5f}  {ax:>10.4f}  {ay:>10.4f}")

    # ── B6. Metrics + displacement stats ─────────────────────────────────────
    last_valid_t = int(mask.nonzero(as_tuple=False)[-1].item()) if mask.any() else 59
    gx_l, gy_l  = gt_abs[n, last_valid_t].tolist()

    lines.append("")
    lines.append("[ B6: METRICS + DISPLACEMENT STATISTICS ]")
    lines.append(f"  GT final pos (t={last_valid_t}): ({gx_l:.4f}, {gy_l:.4f})")
    lines.append("")
    lines.append(f"  {'mode':>4}  {'mean|d|':>9}  {'max|d|':>9}  {'sum_x':>9}  {'sum_y':>9}  "
                 f"{'ADE':>8}  {'FDE':>8}  {'verdict':>16}")
    lines.append(f"  {'----':>4}  {'---------':>9}  {'---------':>9}  {'---------':>9}  {'---------':>9}  "
                 f"{'--------':>8}  {'--------':>8}  {'----------------':>16}")

    gt_valid_pts = gt_abs[n][mask] if mask.any() else None
    ades, fdes   = [], []
    for k in range(6):
        d      = pred_disp[n, k]
        d_norm = d.norm(dim=-1)
        sum_x  = d[:, 0].sum().item()
        sum_y  = d[:, 1].sum().item()
        mean_d = d_norm.mean().item()
        max_d  = d_norm.max().item()

        if gt_valid_pts is not None and is_train:
            pred_valid = pred_abs[n, k][mask]
            ade = (pred_valid - gt_valid_pts).norm(dim=-1).mean().item()
            fde = (pred_abs[n, k, last_valid_t] - gt_abs[n, last_valid_t]).norm().item()
        else:
            ade = fde = float("nan")
        ades.append(ade)
        fdes.append(fde)

        verdict = "NEAR-ZERO" if mean_d < 0.02 else "low"  if mean_d < 0.10 else "OK"
        lines.append(f"  {k:>4}  {mean_d:>9.5f}  {max_d:>9.5f}  {sum_x:>9.4f}  {sum_y:>9.4f}  "
                     f"{ade:>8.4f}  {fde:>8.4f}  {verdict:>16}")

    if is_train and not np.isnan(ades[0]):
        lines.append(f"  minADE = {min(ades):.4f} m   minFDE = {min(fdes):.4f} m")

    # Mode diversity
    mode_std = pred_disp[n].std(dim=0).mean().item()
    lines.append(f"  Mode diversity (std across 6 modes): {mode_std:.6f} m"
                 f"  {'← collapsed!' if mode_std < 0.001 else ''}")

lines.append("")
lines.append(SEP)
lines.append("END OF REPORT")
lines.append(SEP)

# ── Write ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nReport saved to: {OUT_FILE}")
print(f"Total lines: {len(lines)}")
