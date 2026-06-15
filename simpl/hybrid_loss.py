"""
Hybrid Motion Loss — applies LLMMotionLoss at each Level-k output,
weighted so the final level receives higher supervision.

Level weights (n_levels=2):  [0.25, 0.25, 0.50]
                               L0     L1     L2
"""

import torch
import torch.nn as nn
from simpl.av2_llm_loss import LLMMotionLoss


class HybridMotionLoss(nn.Module):
    def __init__(self, n_levels=2, device="cuda",
                 soft_wta_alpha=0.0, cls_weight=0.5,
                 interaction_weight=0.0, collision_margin=2.0):
        """
        n_levels: number of interaction refinement levels (NOT counting level-0).
                  Total prediction outputs = n_levels + 1.
        exp12: anchor_cls term removed (see LLMMotionLoss docstring).
        exp16: interaction_weight enables GameFormer's repulsion loss (Eq.4). At each
               level k>=1 it penalises agent i's predicted trajectory for getting
               within collision_margin of any OTHER agent's level-(k-1) trajectory
               (the "leader" response, detached). This is what makes the level-k
               self-attention game functionally collision-aware; it directly targets
               actorCR. Default 0.0 keeps the term off (pre-exp16 behaviour).
        """
        super().__init__()
        self.device = device
        self.base_loss = LLMMotionLoss(
            device=device,
            soft_wta_alpha=soft_wta_alpha,
            cls_weight=cls_weight,
        )
        self.interaction_weight = interaction_weight
        self.collision_margin   = collision_margin
        # Weights: early levels get half the weight of the final level
        n_total = n_levels + 1
        raw = [0.5] * n_levels + [1.0]
        total = sum(raw)
        self.level_weights = [w / total for w in raw]

    def _repulsion_loss(self, all_preds, data):
        """
        GameFormer Eq.4 repulsion. For each level k>=1, agent i (level k) is pushed
        away from every other agent j's level-(k-1) trajectory (detached = fixed
        leader). Per joint-world convention we pair same mode m for i and j.

        all_preds: list of [B,N,K,T,2] offsets from anchor. Returns scalar.
        """
        anchor     = data["gt_anchor"].to(self.device, dtype=all_preds[0].dtype)   # [B,N,2]
        agent_valid= data["agent_valid"].to(self.device)                           # [B,N]
        train_mask = data["train_mask"].to(self.device)                            # [B,N]

        a = anchor.unsqueeze(2).unsqueeze(2)                                        # [B,N,1,1,2]
        B, N = anchor.shape[0], anchor.shape[1]
        margin = self.collision_margin
        eye    = torch.eye(N, dtype=torch.bool, device=self.device)                # [N,N]

        total, n_terms = all_preds[0].new_zeros(()), 0
        for k in range(1, len(all_preds)):
            cur  = a + all_preds[k]                       # [B,N,K,T,2]   (level k, agent i)
            prev = (a + all_preds[k - 1]).detach()        # [B,N,K,T,2]   (level k-1, leader j)
            # pairwise i (dim1) vs j (dim2), same mode/timestep
            d = (cur.unsqueeze(2) - prev.unsqueeze(1)).norm(dim=-1)   # [B,Ni,Nj,K,T]
            # mask self-pairs and padding leaders → +inf (excluded by min)
            j_pad = (~agent_valid).view(B, 1, N, 1, 1) | eye.view(1, N, N, 1, 1)
            d = d.masked_fill(j_pad, float("inf"))
            min_d = d.min(dim=2).values                  # [B,Ni,K,T]  nearest leader
            # 1/(d+1) gated to a safety margin (GameFormer Eq.4)
            pen = (1.0 / (min_d + 1.0)) * (min_d < margin).to(min_d.dtype)
            # only supervise scored agents i
            pen = pen * train_mask.float().view(B, N, 1, 1)
            total   = total + pen.sum()
            n_terms = n_terms + train_mask.float().sum() * pen.shape[2] * pen.shape[3]

        if n_terms == 0:
            return total
        return total / n_terms.clamp(min=1e-9)

    def forward(self, all_preds, data, all_scores=None, codebook=None):
        """
        all_preds:  list of [B, N, K, T, 2], length = n_levels + 1
        all_scores: list of [B, N, K],        length = n_levels + 1 (or None)
        codebook:   AnchorCodebook (diagnostic endpoint tracker, or None)
        data:       batch dict (same format as LLMMotionLoss expects)
        Returns:    dict with 'loss' (total) + per-level 'loss_l{k}'

        The codebook's online EMA update (diagnostic only) is triggered exactly
        ONCE per step, on the FINAL level.
        """
        total_loss = None
        loss_dict  = {}
        last_k     = len(all_preds) - 1

        for k, pred in enumerate(all_preds):
            scores_k = all_scores[k] if all_scores is not None else None
            out = self.base_loss(pred, data, scores_k, codebook,
                                 update_codebook=(k == last_k))
            w   = self.level_weights[k]
            loss_dict[f"loss_l{k}"]     = out["loss"]
            loss_dict[f"reg_loss_l{k}"] = out["reg_loss"]
            loss_dict[f"cls_loss_l{k}"] = out["cls_loss"]
            if total_loss is None:
                total_loss = w * out["loss"]
            else:
                total_loss = total_loss + w * out["loss"]

        # ── GameFormer Eq.4 repulsion (exp16) ────────────────────────────────
        if self.interaction_weight > 0.0 and len(all_preds) > 1:
            inter_loss = self._repulsion_loss(all_preds, data)
            loss_dict["inter_loss"] = inter_loss
            total_loss = total_loss + self.interaction_weight * inter_loss

        loss_dict["loss"] = total_loss
        return loss_dict
