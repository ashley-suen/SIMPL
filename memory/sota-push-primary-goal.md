---
name: sota-push-primary-goal
description: Current primary research goal is pushing minADE toward SOTA, not ablations
metadata:
  type: project
---

As of 2026-06-12 the user's top priority is **improving the final minADE metric toward
SOTA**, NOT ablation studies. exp12-B (LLM scene guidance) plateaued at joint
avgMinADE ≈ 1.94m (ep7) — ~1.5–2× off AV2 SOTA. The exp12-A ablation and the
LoRA-vs-Partial-FT comparison are **deprioritized**; user keeps LoRA (Partial-FT
analysis showed it won't reliably improve accuracy, consistent with early exps).

**Why:** absolute SOTA performance now matters more than clean attribution of the
LLM's contribution.

**How to apply:** propose changes that move minADE, ordered by ROI. Key insight:
scene scaling is FREE (no re-preprocessing) — dataset reads full agents/lanes from
pkl and truncates to max_agents/max_lanes; sequence length is dominated by the 600
text tokens, so raising max_agents/max_lanes adds only ~10% sequence. Highest-ROI
next step (exp13) is pure-CLI scene scaling: max_agents 6→20, max_lanes 20→64.
Keep num_modes=6 — AV2 evaluates at minADE_1/minADE_6, so K must match the
reportable metric; do NOT raise num_modes to game oracle coverage (user rejected
K=10). See [[exp-launch-convention]] and analysis/metric_improvement_plan.md.
