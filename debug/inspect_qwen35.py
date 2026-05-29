"""
Inspect Qwen3.5-0.8B-Base (language_model sub-module) and Qwen2.5-0.5B
to determine which is easier to integrate into SmolLMMotionPredictor.
"""
from transformers import AutoModel, AutoConfig
import torch


def inspect_model(model_id):
    print(f"\n{'='*70}")
    print(f"MODEL: {model_id}")
    print(f"{'='*70}")

    config = AutoConfig.from_pretrained(model_id)
    model  = AutoModel.from_pretrained(model_id, dtype=torch.float32,
                                       low_cpu_mem_usage=True)
    model.eval()

    # ── Top-level children ────────────────────────────────────────────────────
    print("\n[ Top-level children ]")
    for name, _ in model.named_children():
        print(f"  {name}")

    # ── Locate the text encoder ───────────────────────────────────────────────
    # Could be: model itself / model.model / model.language_model / model.language_model.model
    candidates = {
        "model":                          model,
        "model.model":                    getattr(model, "model", None),
        "model.language_model":           getattr(model, "language_model", None),
    }
    if hasattr(model, "language_model"):
        candidates["model.language_model.model"] = getattr(model.language_model, "model", None)

    print("\n[ Compatibility check — looking for embed_tokens / layers / norm ]")
    for path, obj in candidates.items():
        if obj is None:
            continue
        has_et = hasattr(obj, "embed_tokens")
        has_ly = hasattr(obj, "layers")
        has_no = hasattr(obj, "norm")
        marker = "✓ COMPATIBLE" if (has_et and has_ly and has_no) else ""
        print(f"  {path:40s}  embed_tokens={has_et}  layers={has_ly}  norm={has_no}  {marker}")

    # ── Hidden size ───────────────────────────────────────────────────────────
    print("\n[ Hidden size ]")
    for cfg_obj, label in [(config, "config"),
                            (getattr(config, "text_config", None), "config.text_config")]:
        if cfg_obj is None:
            continue
        for attr in ["hidden_size", "d_model", "embed_dim", "n_embd", "dim"]:
            val = getattr(cfg_obj, attr, None)
            if val is not None:
                print(f"  {label}.{attr} = {val}")

    # ── Context length ────────────────────────────────────────────────────────
    print("\n[ Context length ]")
    for cfg_obj, label in [(config, "config"),
                            (getattr(config, "text_config", None), "config.text_config")]:
        if cfg_obj is None:
            continue
        for attr in ["max_position_embeddings", "max_seq_len", "n_positions"]:
            val = getattr(cfg_obj, attr, None)
            if val is not None:
                print(f"  {label}.{attr} = {val}")

    # ── Forward pass ─────────────────────────────────────────────────────────
    print("\n[ Forward pass ]")
    dummy_ids  = torch.randint(0, 1000, (1, 32))
    dummy_mask = torch.ones(1, 32, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=dummy_ids, attention_mask=dummy_mask)
    H = out.last_hidden_state.shape[-1]
    print(f"  last_hidden_state: {out.last_hidden_state.shape}  →  H = {H}")

    # ── Layer count ───────────────────────────────────────────────────────────
    text_enc = (getattr(model, "language_model", None) or
                getattr(model, "model", None) or model)
    layers = getattr(text_enc, "layers", None)
    if layers is None and hasattr(text_enc, "model"):
        layers = getattr(text_enc.model, "layers", None)
    if layers is not None:
        print(f"  num layers = {len(layers)}")


# ── Run both ──────────────────────────────────────────────────────────────────
inspect_model("Qwen/Qwen3-0.6B-Base")
inspect_model("Qwen/Qwen3.5-0.8B-Base")

print("\n\nDone.")

