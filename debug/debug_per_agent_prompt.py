"""
Inspect the Hybrid prompt format (AV2HybridDataset).

Shows for each sample:
  1. Text portion  — decoded semantic text (no coordinate steps)
  2. Agent features summary — shape, type distribution, position/velocity range
  3. Lane features summary  — shape, coverage
  4. inputs_embeds sequence layout  — token count breakdown
  5. Tensor shapes from __getitem__
  6. Agent valid / train_mask table

Run from SIMPL_back root:
    python debug/debug_per_agent_prompt.py
"""
import os, sys, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from data_av2.av2_hybrid_dataset import AV2HybridDataset

TRAIN_DIR  = "data_av2/features/train"
VAL_DIR    = "data_av2/features/val"
OUT_FILE   = "debug/hybrid_prompt_inspect.txt"
N_SAMPLES  = 5
SEED       = 4265

# Dataset config (must match training)
MAX_AGENTS   = 6
MAX_LANES    = 20
MAX_TEXT_LEN = 384
MODEL_NAME   = "Qwen/Qwen3-0.6B-Base"

random.seed(SEED)
torch.manual_seed(SEED)

SEP = "=" * 80
sep = "-" * 80

AGENT_TYPE_NAMES = {
    0: "VEHICLE",
    1: "PEDESTRIAN",
    2: "MOTORCYCLIST",
    3: "CYCLIST",
    4: "BUS",
    5: "UNKNOWN",
    6: "STATIC",
}

lines = []

def w(s=""):
    lines.append(str(s))


def summarise_agent_features(agent_features, agent_valid, train_mask):
    """
    agent_features: [N, T_obs, 13]
    Returns multi-line summary string.
    """
    out = []
    N = agent_features.shape[0]
    for i in range(N):
        if not agent_valid[i]:
            out.append(f"  Slot {i}: (padding)")
            continue
        feat  = agent_features[i]           # [T, 13]
        typ   = feat[:, 6:13]               # [T, 7]
        # Determine agent type from first valid frame
        type_sum = typ.sum(axis=0)
        dominant = int(type_sum.argmax()) if type_sum.max() > 0 else 6
        type_name = AGENT_TYPE_NAMES.get(dominant, "UNKNOWN")

        pos   = feat[:, 0:2]
        vel   = feat[:, 2:4]
        valid_frames = np.any(feat != 0, axis=1).sum()
        speed = np.linalg.norm(vel, axis=1)
        valid_speed = speed[speed > 0]
        speed_str = (f"{valid_speed.min():.1f}–{valid_speed.max():.1f} m/s"
                     if len(valid_speed) > 0 else "stationary")
        x_range = f"x:[{pos[:,0].min():.1f},{pos[:,0].max():.1f}]"
        y_range = f"y:[{pos[:,1].min():.1f},{pos[:,1].max():.1f}]"
        tr = "train" if train_mask[i] else "context"
        out.append(f"  Slot {i}: {type_name:<12} | {tr:<7} | "
                   f"{valid_frames}/50 valid frames | speed={speed_str} | "
                   f"{x_range} {y_range}")
    return out


def summarise_lane_features(lane_features, lane_valid):
    """
    lane_features: [L, 10, 8]
    Returns summary string.
    """
    n_valid = int(lane_valid.sum())
    if n_valid == 0:
        return ["  (no valid lanes)"]
    # Compute spatial range of all valid lane points
    valid_lanes = lane_features[lane_valid.astype(bool)]   # [n_valid, 10, 8]
    pts = valid_lanes[:, :, 0:2].reshape(-1, 2)             # [n_valid*10, 2]
    intersect_pts = valid_lanes[:, :, 4]                    # [n_valid, 10]
    n_intersect = int((intersect_pts > 0).any(axis=1).sum())
    # Lane type distribution (vehicle/bike/bus)
    lt = valid_lanes[:, :, 5:8]
    lt_sum = lt.reshape(-1, 3).sum(axis=0)
    total_pts = lt_sum.sum()
    type_str = (f"vehicle={int(lt_sum[0])} bike={int(lt_sum[1])} bus={int(lt_sum[2])}"
                if total_pts > 0 else "n/a")
    return [
        f"  Valid lanes: {n_valid}/{lane_valid.shape[0]} | "
        f"Intersection lanes: {n_intersect} | Lane types: {type_str}",
        f"  Spatial range: x:[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
        f"y:[{pts[:,1].min():.1f},{pts[:,1].max():.1f}]",
    ]


def sequence_layout(L_text, N, L_lanes):
    """Print the inputs_embeds sequence structure."""
    total = L_text + N + L_lanes + N
    lines_out = [
        f"  inputs_embeds layout  (total = {total} tokens)",
        f"  {'Region':<28}  {'Start':>6}  {'Length':>7}  {'End':>6}",
        f"  {'-'*28}  {'------':>6}  {'-------':>7}  {'------':>6}",
    ]
    regions = [
        ("Text (semantic)",       0,          L_text),
        ("Agent embeddings",      L_text,     N),
        ("Lane embeddings",       L_text + N, L_lanes),
        ("[AGT] summary tokens",  L_text + N + L_lanes, N),
    ]
    for name, start, length in regions:
        lines_out.append(f"  {name:<28}  {start:>6}  {length:>7}  {start+length-1:>6}")
    lines_out.append(f"  {'─'*53}")
    lines_out.append(f"  {'TOTAL':<28}  {'0':>6}  {total:>7}  {total-1:>6}")
    # AGT positions
    agt_positions = [L_text + N + L_lanes + i for i in range(N)]
    lines_out.append(f"  AGT token positions: {agt_positions}")
    return lines_out


def inspect_sample(ds, idx, sample_no, split_name):
    w(SEP)
    w(f"SPLIT: {split_name}  |  Sample {sample_no}/{N_SAMPLES}  |  Dataset index: {idx}")
    w(SEP)

    # ── Raw data ──────────────────────────────────────────────────────────────
    row = pd.read_pickle(ds.file_list[idx]).iloc[0]

    # Get text + neighbor ids directly
    text_ids, text_mask, neighbor_ids = ds._build_scene_text(row)
    agent_feats, agent_valid, train_mask = ds._get_agent_features(row, neighbor_ids)
    lane_feats,  lane_valid              = ds._get_lane_features(row)

    valid_text_len = int(text_mask.sum().item())
    n_valid_agents = int(agent_valid.sum().item())
    n_valid_lanes  = int(lane_valid.sum().item())

    w(f"Agents in scene: {n_valid_agents}/{MAX_AGENTS}  "
      f"(focal + {len(neighbor_ids)} neighbors: ids={[0]+list(neighbor_ids)})")
    w(f"Valid lanes    : {n_valid_lanes}/{MAX_LANES}")
    w()

    # ── 1. Text portion ───────────────────────────────────────────────────────
    w(sep)
    w("1. TEXT PORTION  (semantic, no coordinate steps):")
    w(sep)
    decoded = ds.tokenizer.decode(
        text_ids[:valid_text_len].tolist(), skip_special_tokens=False)
    for line in decoded.split("\n"):
        w(line)
    w(sep)
    w(f"Text tokens: {valid_text_len} / {MAX_TEXT_LEN}  "
      f"({valid_text_len / MAX_TEXT_LEN * 100:.1f}% of max_text_len)")

    # ── 2. Agent features ─────────────────────────────────────────────────────
    w()
    w(sep)
    w(f"2. AGENT NUMERICAL FEATURES  [shape: {list(agent_feats.shape)}  "
      f"= (N={MAX_AGENTS}, T_obs=50, feat=13)]")
    w(f"   Features per timestep: x, y, vx, vy, cos_θ, sin_θ,  type_one_hot(7)")
    w(sep)
    for line in summarise_agent_features(
            agent_feats.numpy(), agent_valid.numpy(), train_mask.numpy()):
        w(line)

    # ── 3. Lane features ──────────────────────────────────────────────────────
    w()
    w(sep)
    w(f"3. LANE NUMERICAL FEATURES   [shape: {list(lane_feats.shape)}  "
      f"= (L={MAX_LANES}, pts=10, feat=8)]")
    w(f"   Features per point: x, y, dx, dy, is_intersect, lane_type(3)")
    w(sep)
    for line in summarise_lane_features(lane_feats.numpy(), lane_valid.numpy()):
        w(line)

    # ── 4. inputs_embeds layout ───────────────────────────────────────────────
    w()
    w(sep)
    w("4. inputs_embeds SEQUENCE LAYOUT  (all in LLM hidden space H):")
    w(sep)
    for line in sequence_layout(valid_text_len, MAX_AGENTS, n_valid_lanes):
        w(line)

    # ── 5. Tensor shapes (first sample per split only) ────────────────────────
    if sample_no == 1:
        w()
        w(SEP)
        w("5. __getitem__ TENSOR SHAPES  (shown once per split)")
        w(SEP)
        sample = ds[idx]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                w(f"  {k:<28}: shape={str(tuple(v.shape)):<25}  dtype={v.dtype}")
            else:
                w(f"  {k:<28}: (non-tensor)")
    else:
        sample = ds[idx]

    # ── 6. Agent slot table ───────────────────────────────────────────────────
    w()
    w(sep)
    w("6. AGENT SLOT TABLE")
    w(f"  {'Slot':>4}  {'traj_id':>8}  {'Type':<13}  {'valid':>6}  {'train':>6}  "
      f"{'[AGT] pos':>10}")
    w(f"  {'----':>4}  {'-------':>8}  {'-----':<13}  {'------':>6}  {'------':>6}  "
      f"{'----------':>10}")

    agent_ids = [0] + list(neighbor_ids)
    while len(agent_ids) < MAX_AGENTS:
        agent_ids.append(-1)

    n_valid_l = int(lane_valid.sum())
    L_text_real = valid_text_len

    for slot in range(MAX_AGENTS):
        traj_id   = agent_ids[slot] if slot < len(agent_ids) else -1
        valid     = bool(agent_valid[slot])
        train     = bool(train_mask[slot])
        agt_pos   = L_text_real + MAX_AGENTS + n_valid_l + slot
        feat      = agent_feats[slot].numpy()
        typ       = feat[:, 6:13].sum(axis=0)
        dom       = int(typ.argmax()) if typ.max() > 0 else 6
        type_name = AGENT_TYPE_NAMES.get(dom, "?") if valid else "(padding)"
        w(f"  {slot:>4}  {str(traj_id):>8}  {type_name:<13}  "
          f"{str(valid):>6}  {str(train):>6}  {agt_pos:>10}")
    w()


def inspect_split(split_name, data_dir):
    ds      = AV2HybridDataset(
        data_dir,
        tokenizer_name=MODEL_NAME,
        max_agents=MAX_AGENTS,
        max_lanes=MAX_LANES,
        max_text_len=MAX_TEXT_LEN,
    )
    indices = random.sample(range(len(ds)), min(N_SAMPLES, len(ds)))

    w(f"\n{'#'*80}")
    w(f"# SPLIT: {split_name}  --  {N_SAMPLES} samples  (dataset size: {len(ds)})")
    w(f"{'#'*80}\n")

    for i, idx in enumerate(indices, 1):
        inspect_sample(ds, idx, i, split_name)


# ── Run ────────────────────────────────────────────────────────────────────────
inspect_split("train", TRAIN_DIR)
inspect_split("val",   VAL_DIR)
w(SEP)
w("Done.")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Output written to: {OUT_FILE}  ({len(lines)} lines)")
