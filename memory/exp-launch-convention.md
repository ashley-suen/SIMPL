---
name: exp-launch-convention
description: How to name and launch hybrid-anchor experiments (exp counter, batch/LR, resume flags)
metadata:
  type: project
---

Convention for launching `train_hybrid.py` experiments on the 4× RTX PRO 6000 (96GB) box.

**Naming (sequential counter):** use `--exp_name expN_<desc>` (e.g. `exp7_anchor_ema`).
Counter reached exp6 before the anchor-codebook work; continue incrementing.
`exp_name` now drives BOTH the log dir (`log/{exp_name}/`) and the checkpoint
filenames (`saved_models/{exp_name}_{timestamp}_hybrid_best.tar` / `_last.tar`),
so logs and weights pair up — this prefixing was added in train_hybrid.py.

**Backend:** `--dist_backend gloo` — see [[cloud-ddp-use-gloo]] (nccl is buggy on the
cloud platform).

**Batch / workers:** 96GB is far more than this model (Qwen3-0.6B + LoRA, agents=6,
lanes=20, text≤500) needs. Start `--train_batch_size 16 --val_batch_size 24
--num_workers 8` (global batch 64 across 4 GPUs); can push to 24–32 to fill VRAM.

**LR with larger batch:** default `--gru_lr 1e-4 --llm_lr 5e-5` is tuned for batch 4.
After scaling batch 4→16, if warmup-phase loss drops too slowly, bump conservatively
to `--gru_lr 2e-4 --llm_lr 1e-4`. Run defaults first, decide from the post-warmup curve.

**Resume across the anchor-codebook architecture change:** MUST pass
`--reset_optimizer`. The anchors stopped being an nn.Parameter (moved out of the
optimizer) and `anchor_mlp` was added, so the old `opt_state` param groups mismatch.
`load_state_dict` uses `strict=False`, so new buffers/layers cold-start cleanly.
Latest hybrid checkpoint to resume from: `saved_models/20260603-020731_hybrid_local_best.tar`.

**Anchor diagnostics:** each epoch the val block prints `[Anchors]` — per-anchor
(x,y) + ema_count + min pairwise sep — plus TensorBoard `anchor/*` scalars. Watch
min_sep stays well above 0 (no mode collapse) and no anchor's ema_count sticks near
reset_thresh=1.0 (zombie anchor).
