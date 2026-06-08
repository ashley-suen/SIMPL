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
                 soft_wta_alpha=0.1, endpoint_weight=0.2, cls_weight=0.5,
                 anchor_cls_weight=0.5, pos_weight=0.1):
        """
        n_levels: number of interaction refinement levels (NOT counting level-0).
                  Total prediction outputs = n_levels + 1.
        """
        super().__init__()
        self.base_loss = LLMMotionLoss(
            device=device,
            soft_wta_alpha=soft_wta_alpha,
            endpoint_weight=endpoint_weight,
            cls_weight=cls_weight,
            anchor_cls_weight=anchor_cls_weight,
            pos_weight=pos_weight,
        )
        # Weights: early levels get half the weight of the final level
        n_total = n_levels + 1
        raw = [0.5] * n_levels + [1.0]
        total = sum(raw)
        self.level_weights = [w / total for w in raw]

    def forward(self, all_preds, data, all_scores=None, codebook=None):
        """
        all_preds:  list of [B, N, K, T, 2], length = n_levels + 1
        all_scores: list of [B, N, K],        length = n_levels + 1 (or None)
        codebook:   AnchorCodebook (shared across levels, or None)
        data:       batch dict (same format as LLMMotionLoss expects)
        Returns:    dict with 'loss' (total) + per-level 'loss_l{k}'

        The codebook's online EMA update is triggered exactly ONCE per step, on the
        FINAL level — so every level's anchor-cls target uses the same (pre-update)
        anchors, and the update sees a single set of GT endpoints per batch.
        """
        total_loss = None
        loss_dict  = {}
        last_k     = len(all_preds) - 1

        for k, pred in enumerate(all_preds):
            scores_k = all_scores[k] if all_scores is not None else None
            out = self.base_loss(pred, data, scores_k, codebook,
                                 update_codebook=(k == last_k))
            w   = self.level_weights[k]
            loss_dict[f"loss_l{k}"]            = out["loss"]
            loss_dict[f"reg_loss_l{k}"]        = out["reg_loss"]
            loss_dict[f"pos_loss_l{k}"]        = out["pos_loss"]
            loss_dict[f"ep_loss_l{k}"]         = out["ep_loss"]
            loss_dict[f"cls_loss_l{k}"]        = out["cls_loss"]
            loss_dict[f"anchor_cls_loss_l{k}"] = out["anchor_cls_loss"]
            if total_loss is None:
                total_loss = w * out["loss"]
            else:
                total_loss = total_loss + w * out["loss"]

        loss_dict["loss"] = total_loss
        return loss_dict
