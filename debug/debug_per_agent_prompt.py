"""
Inspect the unified prompt format.
Loads one sample from train and one from val, writes full output to a txt file.

Run from SIMPL_back root:
    python debug/debug_per_agent_prompt.py
"""
import os, sys, random, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from data_av2.av2_llm_dataset import AV2PromptDataset

TRAIN_DIR = "data_av2/features/train"
VAL_DIR   = "data_av2/features/val"
OUT_FILE   = "debug/unified_prompt_inspect.txt"
N_SAMPLES  = 5
SEED       = 4265

random.seed(SEED)
torch.manual_seed(SEED)

SEP  = "=" * 80
sep  = "-" * 80

lines = []

def w(s=""):
    lines.append(s)

def inspect_sample(ds, idx, sample_no, split_name):
    w(SEP)
    w(f"SPLIT: {split_name}  |  Sample {sample_no}/{N_SAMPLES}  |  Dataset index: {idx}")
    w(SEP)

    row = pd.read_pickle(ds.file_list[idx]).iloc[0]

    # ── Raw unified prompt ────────────────────────────────────────────────────
    input_ids, attention_mask, agent_token_positions, written_ids = \
        ds.generate_unified_prompt(row)

    w(f"Agents written: {len(written_ids)}  (traj indices: {written_ids})")
    w(f"Agent [PREDICT]: token positions: {agent_token_positions}")
    w()

    tokenizer = ds.tokenizer
    valid_len = int(attention_mask.sum().item())
    prompt_text = tokenizer.decode(input_ids[:valid_len].tolist(), skip_special_tokens=False)

    w(sep)
    w("UNIFIED PROMPT:")
    w(sep)
    for line in prompt_text.split("\n"):
        w(line)
    w(sep)
    w(f"Valid tokens: {valid_len} / {input_ids.shape[0]}  "
      f"({valid_len / input_ids.shape[0] * 100:.1f}% utilization)")

    # ── Tensor shapes (first sample only to avoid repetition) ────────────────
    if sample_no == 1:
        w()
        w(SEP)
        w("__getitem__ tensor shapes (shown once)")
        w(SEP)
        sample = ds[idx]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                w(f"  {k:25s}: {str(tuple(v.shape)):20s}  dtype={v.dtype}")
            else:
                w(f"  {k:25s}: (non-tensor)")
    else:
        sample = ds[idx]

    # ── Agent positions table ─────────────────────────────────────────────────
    w()
    w(f"Agent positions — hidden state at this token index → GRU Decoder")
    w(f"  {'slot':>4}  {'traj_id':>8}  {'token_pos':>10}  {'valid':>6}  {'train':>6}  decoded token")
    w(f"  {'----':>4}  {'-------':>8}  {'---------':>10}  {'------':>6}  {'------':>6}  -------------")
    ag_pos  = sample["agent_positions"]
    ag_val  = sample["agent_valid"]
    tr_mask = sample["train_mask"]
    ids_tok = sample["input_ids"]
    for a in range(ag_pos.shape[0]):
        pos   = ag_pos[a].item()
        valid = ag_val[a].item()
        train = tr_mask[a].item()
        tid   = written_ids[a] if a < len(written_ids) else "-"
        tok   = tokenizer.decode([ids_tok[pos].item()]) if valid else "(padding slot)"
        w(f"  {a:>4}  {str(tid):>8}  {pos:>10}  {str(bool(valid)):>6}  "
          f"{str(bool(train)):>6}  '{tok}'")
    w()


def inspect_split(split_name, data_dir):
    ds      = AV2PromptDataset(data_dir)
    indices = random.sample(range(len(ds)), min(N_SAMPLES, len(ds)))
    w(f"\n{'#'*80}")
    w(f"# SPLIT: {split_name}  —  {N_SAMPLES} samples  (dataset size: {len(ds)})")
    w(f"{'#'*80}\n")
    for i, idx in enumerate(indices, 1):
        inspect_sample(ds, idx, i, split_name)


inspect_split("train", TRAIN_DIR)
inspect_split("val",   VAL_DIR)
w(SEP)
w("Done.")

# ── Write to file ──────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Output written to: {OUT_FILE}  ({len(lines)} lines)")
