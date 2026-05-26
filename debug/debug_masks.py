"""
Diagnose agent_token_masks: check how many tokens each agent is assigned.
"""
import os, random, sys, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.stdout.reconfigure(encoding='utf-8')

from data_av2.av2_llm_dataset import AV2PromptDataset
from transformers import AutoTokenizer

SEED    = 0
VAL_DIR = "../data_av2/features/val"

torch.manual_seed(SEED); random.seed(SEED)

val_set = AV2PromptDataset(VAL_DIR)
idx     = random.randint(0, len(val_set) - 1)
print(f"Sample idx: {idx}\n")
sample  = val_set[idx]

masks      = sample["agent_token_masks"]   # [N, SeqLen]
ag_valid   = sample["agent_valid"]         # [N]
input_ids  = sample["input_ids"]           # [SeqLen]
prompt     = sample["prompt_text"]

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

print("=" * 60)
print("TOKEN MASK STATS PER AGENT")
print("=" * 60)
N = masks.shape[0]
for a in range(N):
    n_tokens = masks[a].sum().item()
    valid    = ag_valid[a].item()
    # find which token positions belong to this agent
    positions = masks[a].nonzero(as_tuple=True)[0].tolist()
    if positions:
        span = f"pos {positions[0]}..{positions[-1]}"
        # decode those tokens to see what text they cover
        token_text = tokenizer.decode(input_ids[positions[0]:positions[-1]+1])
        span_text  = repr(token_text[:80])
    else:
        span = "EMPTY"
        span_text = "N/A"
    print(f"  Agent {a} | valid={valid} | mask_tokens={int(n_tokens):3d} | {span} | text={span_text}")

print(f"\nSequence length : {input_ids.shape[0]}")
print(f"Non-pad tokens  : {sample['attention_mask'].sum().item()}")

print("\n" + "=" * 60)
print("FULL PROMPT")
print("=" * 60)
print(prompt)
