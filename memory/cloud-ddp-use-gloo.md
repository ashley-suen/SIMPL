---
name: cloud-ddp-use-gloo
description: On the cloud GPU platform, DDP must use gloo backend — nccl errors out (platform bug)
metadata:
  type: project
---

On the user's cloud GPU platform (4× RTX PRO 6000 96GB), launch DDP training with
`--dist_backend gloo`. NCCL throws an error there due to a platform bug, even though
NCCL is normally the correct choice for multi-GPU.

**Why:** The cloud platform has a bug that breaks NCCL initialization/communication;
gloo works around it. (gloo is slower for GPU collectives but functional.)

**How to apply:** In every `torchrun ... train_hybrid.py` command on the cloud box,
pass `--dist_backend gloo` (NOT nccl). See [[exp-launch-convention]] for the full
launch command.
