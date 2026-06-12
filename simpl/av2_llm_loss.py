import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMMotionLoss(nn.Module):
    def __init__(self, device, soft_wta_alpha=0.0, cls_weight=0.5):
        """
        soft_wta_alpha: weight for non-winner modes in soft WTA. Default 0.0
                        (pure scene-level WTA — exp12): alpha>0 pulls ALL modes
                        toward every scene's GT, squeezing them toward the
                        conditional mean and fighting mode diversity. With
                        scenario queries there is no anchor-cls regulariser, so
                        the winner-only gradient IS the diversity mechanism.
        cls_weight:     weight for the scene-level mode-classification loss.
                        Supervises the per-mode confidence scores so the K joint
                        "worlds" can be ranked at inference (needed for
                        brierMinFDE). Set to 0.0 to disable.

        exp12: the per-agent anchor-classification term (B2) is REMOVED — its
        per-agent anchor semantics conflicted with the scene-shared winner mode
        of the joint protocol. The AnchorCodebook is kept as a pure diagnostic:
        this loss still triggers its EMA update so the endpoint-distribution
        logging keeps working, but anchors no longer affect any loss term.
        """
        super().__init__()
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="none")
        self.soft_wta_alpha = soft_wta_alpha
        self.cls_weight     = cls_weight

    def forward(self, predicted_trajs, data, scores=None, codebook=None,
                update_codebook=False):
        """
        predicted_trajs: [B, N, K, T, 2]  position offsets from anchor (Bézier decoded)
        data: dict with:
            gt_abs_trajectories [B, N, T, 2]  absolute GT positions
            gt_anchor           [B, N, 2]     agent position at t=0
            gt_masks            [B, N, T]     valid future timestep mask
            agent_valid         [B, N]        all present agents
            train_mask          [B, N]        focal + scored agents only (loss supervision)

        Loss design (JOINT / scene-level):
          - Primary:   SmoothL1 in position space. pred_abs = anchor + predicted_trajs
                       (no cumsum). Training loss and minADE are in the SAME space.
          - Winner selection: SCENE-LEVEL FDE-based (no gradient through prediction).
          - Soft WTA with the shared per-scene winner applied to the position reg_loss.
        """
        B, N, K, T, _ = predicted_trajs.shape

        gt_abs     = data["gt_abs_trajectories"].to(self.device, dtype=predicted_trajs.dtype)  # [B, N, T, 2]
        anchor     = data["gt_anchor"].to(self.device, dtype=predicted_trajs.dtype)            # [B, N, 2]
        gt_masks   = data["gt_masks"].to(self.device)                                          # [B, N, T]
        train_mask = data["train_mask"].to(self.device)                                        # [B, N]

        valid_mask = gt_masks & train_mask.unsqueeze(-1)  # [B, N, T]

        # Absolute predicted positions (Bézier — no cumsum)
        pred_abs_all = anchor.unsqueeze(2).unsqueeze(2) + predicted_trajs   # [B, N, K, T, 2]

        # ── FDE-based winner selection (no gradient) ──────────────────────────
        with torch.no_grad():
            pred_abs_det = anchor.unsqueeze(2).unsqueeze(2) + \
                           predicted_trajs.detach()                          # [B, N, K, T, 2]

            valid_float    = valid_mask.float()
            last_valid_idx = (valid_float.cumsum(dim=-1) * valid_float).argmax(dim=-1)  # [B, N]
            has_any_valid  = valid_mask.any(dim=-1)
            last_valid_idx = torch.where(has_any_valid, last_valid_idx,
                                         torch.full_like(last_valid_idx, T - 1))

            last_exp  = last_valid_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, N, K, 1, 2)
            pred_last = pred_abs_det.gather(3, last_exp).squeeze(3)         # [B, N, K, 2]

            gt_last_exp = last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, 2)
            gt_last     = gt_abs.gather(2, gt_last_exp).squeeze(2)          # [B, N, 2]

            fde = (pred_last - gt_last.unsqueeze(2)).norm(dim=-1)           # [B, N, K]

            # ── SCENE-LEVEL winner: average each agent's FDE over scored
            #    agents, then pick the single mode minimising the joint error.
            scored        = train_mask.float()                              # [B, N]
            scored_cnt    = scored.sum(dim=1, keepdim=True).clamp(min=1e-9) # [B, 1]
            scene_fde     = (fde * scored.unsqueeze(-1)).sum(dim=1) / scored_cnt  # [B, K]
            best_idx_scene = scene_fde.argmin(dim=-1)                       # [B]

        # ── Soft WTA mode weights (shared across all agents in a scene) ────────
        mode_weights = torch.full((B, K), self.soft_wta_alpha,
                                  device=predicted_trajs.device,
                                  dtype=predicted_trajs.dtype)              # [B, K]
        mode_weights.scatter_(1, best_idx_scene.unsqueeze(-1), 1.0)
        mode_weights = mode_weights.unsqueeze(1)                            # [B, 1, K] → broadcast over N

        num_train_agents = train_mask.float().sum().clamp(min=1e-9)

        # ── Primary loss: position-space SmoothL1 ────────────────────────────
        gt_abs_exp = gt_abs.unsqueeze(2).expand(-1, -1, K, -1, -1)          # [B, N, K, T, 2]
        valid_exp  = valid_mask.unsqueeze(2).expand(-1, -1, K, -1)           # [B, N, K, T]

        per_step = self.reg_loss(pred_abs_all, gt_abs_exp).sum(dim=-1)       # [B, N, K, T]
        per_step = per_step * valid_exp.float()

        mode_cnt     = valid_exp.float().sum(dim=-1).clamp(min=1e-9)        # [B, N, K]
        per_mode_ade = per_step.sum(dim=-1) / mode_cnt                      # [B, N, K]

        agent_loss = (per_mode_ade * mode_weights).sum(dim=-1)              # [B, N]
        reg_loss   = (agent_loss * train_mask.float()).sum() / num_train_agents

        # ── Scene-level mode classification ───────────────────────────────────
        # Aggregate per-agent per-mode logits into a single scene-level logit
        # (masked mean over scored agents), then CE against the joint winner mode.
        # This teaches the model which "world" is most likely — needed to rank
        # modes at inference and to compute brierMinFDE.
        if scores is not None and self.cls_weight > 0.0:
            scene_logit = (scores * scored.unsqueeze(-1)).sum(dim=1) / scored_cnt  # [B, K]
            cls_loss    = F.cross_entropy(scene_logit.float(), best_idx_scene)
        else:
            cls_loss = reg_loss.new_zeros(())

        # ── Endpoint-distribution diagnostic (exp12: no anchor-cls term) ──────
        # The codebook is a VQ-VAE-style EMA tracker of GT endpoints, kept ONLY
        # for logging (anchor positions / ema_count / min pairwise sep). Train
        # only, once per step (update_codebook flag set on the final level).
        if codebook is not None and update_codebook and torch.is_grad_enabled():
            with torch.no_grad():
                gt_ep = gt_last - anchor                                    # [B, N, 2]
                codebook.ema_update(gt_ep[train_mask.bool()])

        total_loss = reg_loss + self.cls_weight * cls_loss

        return {"loss": total_loss, "reg_loss": reg_loss, "cls_loss": cls_loss}
