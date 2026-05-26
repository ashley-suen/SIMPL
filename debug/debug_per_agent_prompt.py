"""
Inspect the new per-agent prompt format (Plan A).
Loads one sample from train and one from val, prints all agent prompts
and key tensor shapes to verify the new dataset pipeline.
"""
import os, sys, random, torch
sys.stdout.reconfigure(encoding='utf-8')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data_av2.av2_llm_dataset import AV2PromptDataset

TRAIN_DIR = "../data_av2/features/train"
VAL_DIR   = "../data_av2/features/val"
SEED      = 42

random.seed(SEED)
torch.manual_seed(SEED)

SEP = "=" * 70

def inspect_split(split_name, data_dir):
    print(f"\n{SEP}")
    print(f"SPLIT: {split_name}  ({data_dir})")
    print(SEP)

    ds  = AV2PromptDataset(data_dir)
    idx = random.randint(0, len(ds) - 1)
    print(f"Dataset size: {len(ds)}   |   Selected index: {idx}\n")

    # --- Raw per-agent prompts (before tokenization) ---
    import pandas as pd
    row = pd.read_pickle(ds.file_list[idx]).iloc[0]
    agent_prompts, written_ids = ds.generate_per_agent_prompts(row)

    print(f"Agents written: {len(agent_prompts)}  (traj indices: {written_ids})")
    for a_idx, (prompt, traj_id) in enumerate(agent_prompts):
        label = "FOCAL" if a_idx == 0 else f"NBR{a_idx}"
        n_chars = len(prompt)
        print(f"\n{'─'*60}")
        print(f"[Agent {a_idx} | {label} | traj_id={traj_id} | {n_chars} chars]")
        print(prompt)

    # --- __getitem__ output shapes ---
    print(f"\n{SEP}")
    print("__getitem__ tensor shapes")
    print(SEP)
    sample = ds[idx]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:25s}: {tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {'prompt_text':25s}: str, {len(v)} chars")

    # --- Token usage per agent ---
    print(f"\n{SEP}")
    print("Token usage per agent  (non-pad tokens out of max_len_per_agent)")
    print(SEP)
    ids  = sample["input_ids"]        # [N, L]
    mask = sample["attention_mask"]   # [N, L]
    av   = sample["agent_valid"]      # [N]
    N    = ids.shape[0]
    for a in range(N):
        used  = mask[a].sum().item()
        valid = av[a].item()
        label = "FOCAL" if a == 0 else (f"NBR{a}" if a < len(written_ids) else "EMPTY")
        print(f"  Agent {a} | {label:6s} | valid={valid} | tokens used: {used}/{ids.shape[1]}")


inspect_split("train", TRAIN_DIR)
inspect_split("val",   VAL_DIR)
print(f"\n{SEP}")
print("Done.")
