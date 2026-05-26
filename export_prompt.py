"""
Export per-agent prompts (Plan A) from train and val splits to a markdown file
for manual inspection. Samples N_SAMPLES random scenes from each split and
writes every agent's individual prompt as a separate fenced block.
"""
import sys, os, random
sys.stdout.reconfigure(encoding='utf-8')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from data_av2.av2_llm_dataset import AV2PromptDataset

TRAIN_DIR   = "data_av2/features/train"
VAL_DIR     = "data_av2/features/val"
OUTPUT_PATH = "debug/sample_prompt_output_planA.md"
N_SAMPLES   = 5   # scenes per split
SEED        = 42


def export_split(f, split_name, data_dir, n_samples, rng):
    ds = AV2PromptDataset(data_dir)
    indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))

    f.write(f"# Split: {split_name}  (`{data_dir}`, {len(ds)} scenes total)\n\n")

    for scene_no, idx in enumerate(indices, 1):
        row = pd.read_pickle(ds.file_list[idx]).iloc[0]
        agent_prompts, written_ids = ds.generate_per_agent_prompts(row)

        # Also get token counts from __getitem__
        sample = ds[idx]
        mask   = sample["attention_mask"]   # [N, L]
        av     = sample["agent_valid"]      # [N]
        L      = mask.shape[1]

        seq_id = row.get("SEQ_ID", f"scene_{idx}")
        f.write(f"---\n\n")
        f.write(f"## Scene {scene_no}/{n_samples} — `{seq_id}` (dataset index {idx})\n\n")
        f.write(f"**Agents written:** {len(agent_prompts)}  |  "
                f"**max_len_per_agent:** {L}\n\n")

        # Token summary table
        f.write("| Agent | Role | traj_id | valid | tokens used |\n")
        f.write("|-------|------|---------|-------|-------------|\n")
        for a_idx in range(mask.shape[0]):
            used  = mask[a_idx].sum().item()
            valid = av[a_idx].item()
            if a_idx < len(written_ids):
                role = "FOCAL" if a_idx == 0 else f"NBR{a_idx}"
                tid  = written_ids[a_idx]
            else:
                role, tid = "EMPTY", "—"
            f.write(f"| {a_idx} | {role} | {tid} | {valid} | {used}/{L} |\n")
        f.write("\n")

        # Individual agent prompts
        for a_idx, (prompt_text, traj_id) in enumerate(agent_prompts):
            role = "FOCAL" if a_idx == 0 else f"NBR{a_idx}"
            used = mask[a_idx].sum().item()
            f.write(f"### Agent {a_idx} — {role} (traj_id={traj_id}, {used} tokens)\n\n")
            f.write("```text\n")
            f.write(prompt_text)
            f.write("\n```\n\n")


def main():
    rng = random.Random(SEED)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Plan A — Per-Agent Prompt Inspection\n\n")
        f.write("> Each agent receives its own independent prompt: "
                "**shared scene header + that agent's own trajectory**.\n\n")
        export_split(f, "train", TRAIN_DIR, N_SAMPLES, rng)
        export_split(f, "val",   VAL_DIR,   N_SAMPLES, rng)

    print(f"Exported to: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
