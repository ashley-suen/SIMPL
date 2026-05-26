import sys
sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoConfig, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM-360M"
# model_name = "HuggingFaceTB/SmolLM-135M"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("=== Model Config ===")
print(f"max_position_embeddings : {config.max_position_embeddings}")
print(f"hidden_size             : {config.hidden_size}")
print(f"num_hidden_layers       : {config.num_hidden_layers}")
print(f"vocab_size              : {config.vocab_size}")
print(f"model_type              : {config.model_type}")

print("\n=== Tokenizer ===")
print(f"model_max_length        : {tokenizer.model_max_length}")

# ── Token cost measurement ───────────────────────────────────────────────────
# Measure how many tokens one agent-step costs in the current format
sample_focal_step  = "t=0:(-40.3,1.3)|V:9.2m/s -> "
sample_nbr_step    = "t=0:(40.4,2.6)|V:6.1m/s -> "
sample_stationary  = "t=0~49 Stationary at (19.6, 0.3)"

toks_focal  = tokenizer(sample_focal_step,  add_special_tokens=False)["input_ids"]
toks_nbr    = tokenizer(sample_nbr_step,    add_special_tokens=False)["input_ids"]
toks_stat   = tokenizer(sample_stationary,  add_special_tokens=False)["input_ids"]

print("\n=== Token cost per element (current format) ===")
print(f"One focal step  '{sample_focal_step}' → {len(toks_focal)} tokens")
print(f"One nbr   step  '{sample_nbr_step}'   → {len(toks_nbr)} tokens")
print(f"Stationary line '{sample_stationary}' → {len(toks_stat)} tokens")

tokens_per_step_focal = len(toks_focal)
tokens_per_step_nbr   = len(toks_nbr)

print("\n=== Projection: full 50-step history per agent ===")
focal_50 = tokens_per_step_focal * 50
nbr_50   = tokens_per_step_nbr   * 50
header   = 120   # rough estimate for scene header

print(f"Focal agent 50 steps : {focal_50} tokens")
print(f"Neighbor  50 steps   : {nbr_50} tokens")
print(f"Header (fixed)       : ~{header} tokens")
print(f"6 agents full history: {header} + {focal_50} + 5*{nbr_50} = "
      f"{header + focal_50 + 5*nbr_50} tokens total")

print("\n=== What fits in max_position_embeddings? ===")
max_ctx = config.max_position_embeddings
budget  = max_ctx - header
focal_steps_fit  = budget // tokens_per_step_focal
nbr_budget       = budget - focal_50
nbr_steps_each   = nbr_budget // (5 * tokens_per_step_nbr) if nbr_budget > 0 else 0
print(f"Context window       : {max_ctx} tokens")
print(f"Budget after header  : {budget} tokens")
print(f"Max focal steps      : {focal_steps_fit} (full 50 needs {focal_50})")
print(f"With focal@50 steps, remaining for 5 neighbors: {nbr_budget} tokens")
print(f"  → each neighbor gets ~{nbr_budget // 5} tokens → ~{nbr_steps_each} steps")
