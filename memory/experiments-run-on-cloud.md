---
name: experiments-run-on-cloud
description: All training/experiments run on the cloud box; local machine is code-editing/debug only
metadata:
  type: project
---

ALL experiments (training runs, exp7/exp8/exp9…) are executed on the **cloud** box
(4× RTX PRO 6000 96GB). The local Windows machine (`E:\Msc\new\code\simpl_new\SIMPL_back`)
is used **only** for writing and debugging code — never for real training runs.

**Why:** matters for how I reason about checkpoints and reproducibility. Local
`saved_models/*.tar` are stale/old; the authoritative checkpoints live on the cloud
and I usually don't have their exact filenames. When giving a `--resume_path`, ask
the user for the current cloud ckpt name rather than assuming a local file.

**How to apply:** when changing code, remember files must be SYNCED to the cloud
before a run (see the stale-model-file incident). Provide launch commands targeting
the cloud setup (gloo backend, 4 GPUs). See [[exp-launch-convention]] and
[[cloud-ddp-use-gloo]].
