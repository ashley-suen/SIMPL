---
name: todo-remove-detach-end2end
description: Planned (not yet done) — remove agent_emb.detach() into LLM for end-to-end encoder+LLM co-training
metadata:
  type: project
---

待办（高风险高回报，最后做）：去掉 `HybridLLMPredictor.forward` 中 `agent_emb.detach()`（hybrid_llm_model.py line ~470），让 encoder 与 LLM 端到端联合优化。

**Why:** 当前 encoder 输出 detach 后才进 LLM，LLM 只能看冻结快照，修正只能经单层 `llm_correction_proj`，这是架构根本天花板。之前 detach 是为防 28 层梯度爆炸，但**真正的爆炸根源是 torch.compile（已确认并永久关闭）**，所以前提已不成立。预期单项最大改进（5–10% minADE）。

**How to apply:** 分两步降风险——
1. 软放开：给 LLM→encoder 路径加可学习缩放 α（init=0），warmup 后再放大，配合 gn_enc 监控。
2. 或保留 detach 但加宽修正路径：`llm_correction_proj` 单层 Linear → 2 层 MLP+LayerNorm，先低风险提 capacity。

顺序：在 [[todo-model-improvements]] 的 #1/#3/#4 之后再做。
