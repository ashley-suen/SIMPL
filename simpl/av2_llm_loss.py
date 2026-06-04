import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMMotionLoss(nn.Module):
    def __init__(self, device, soft_wta_alpha=0.1, endpoint_weight=0.2,
                 cls_weight=0.5):
        """
        soft_wta_alpha:  weight for non-winner modes in soft WTA.
        endpoint_weight: weight for endpoint-consistency auxiliary loss.
                         Penalises predicting stationary motion for moving agents.
                         Gradient per step is O(1) — no cumsum amplification.
                         Set to 0.0 to disable.
        cls_weight:      weight for the scene-level mode-classification loss.
                         Supervises the per-mode confidence scores so the K joint
                         "worlds" can be ranked at inference (needed for brierMinFDE).
                         Set to 0.0 to disable.
        """
        super().__init__()
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="none")
        self.soft_wta_alpha  = soft_wta_alpha
        self.endpoint_weight = endpoint_weight
        self.cls_weight      = cls_weight

    def forward(self, predicted_trajs, data, scores=None):
        """
        predicted_trajs: [B, N, K, T, 2]  per-step displacement predictions
        data: dict with:
            gt_abs_trajectories [B, N, T, 2]  absolute GT positions
            gt_anchor           [B, N, 2]     agent position at t=0
            gt_masks            [B, N, T]     valid future timestep mask
            agent_valid         [B, N]        all present agents
            train_mask          [B, N]        focal + scored agents only (loss supervision)

        Loss design (JOINT / scene-level):
          - Primary:   SmoothL1 in displacement space (no cumsum in gradient path).
          - Auxiliary: endpoint consistency — SmoothL1(Σ pred_disp, Σ gt_disp).
                       Gradient per step = d(SmoothL1)/d(sum) × valid[t], O(1), no amplification.
                       Directly penalises "predict stationary" for moving agents.
          - Winner selection: SCENE-LEVEL. The K modes are K joint futures
                       ("worlds"). For each scene we pick ONE winning mode that
                       minimises the joint FDE averaged over all scored agents.
                       That single mode is shared by every agent in the scene
                       (vs. the old marginal scheme where each agent picked its own).
          - Soft WTA applied to both loss terms with the shared per-scene winner.
        """
        B, N, K, T, _ = predicted_trajs.shape

        gt_abs     = data["gt_abs_trajectories"].to(self.device, dtype=predicted_trajs.dtype)  # [B, N, T, 2]
        anchor     = data["gt_anchor"].to(self.device, dtype=predicted_trajs.dtype)            # [B, N, 2]
        gt_masks   = data["gt_masks"].to(self.device)                                          # [B, N, T]
        train_mask = data["train_mask"].to(self.device)                                        # [B, N]

        valid_mask = gt_masks & train_mask.unsqueeze(-1)  # [B, N, T]

        # ── GT per-step displacements ─────────────────────────────────────────
        anchor_exp = anchor.unsqueeze(2)                                    # [B, N, 1, 2]
        gt_prev    = torch.cat([anchor_exp, gt_abs[:, :, :-1, :]], dim=2)   # [B, N, T, 2]
        gt_disp    = gt_abs - gt_prev                                       # [B, N, T, 2]

        # ── FDE-based winner selection (no gradient through cumsum) ───────────
        with torch.no_grad():
            pred_abs_det = anchor.unsqueeze(2).unsqueeze(2) + \
                           predicted_trajs.detach().cumsum(dim=-2)          # [B, N, K, T, 2]

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

        # ── Primary loss: per-step displacement regression ────────────────────
        gt_disp_exp = gt_disp.unsqueeze(2).expand(-1, -1, K, -1, -1)       # [B, N, K, T, 2]
        valid_exp   = valid_mask.unsqueeze(2).expand(-1, -1, K, -1)         # [B, N, K, T]

        per_step = self.reg_loss(predicted_trajs, gt_disp_exp).sum(dim=-1)  # [B, N, K, T]
        per_step = per_step * valid_exp.float()

        mode_cnt     = valid_exp.float().sum(dim=-1).clamp(min=1e-9)        # [B, N, K]
        per_mode_ade = per_step.sum(dim=-1) / mode_cnt                      # [B, N, K]

        agent_loss = (per_mode_ade * mode_weights).sum(dim=-1)              # [B, N]
        reg_loss   = (agent_loss * train_mask.float()).sum() / num_train_agents

        # ── Auxiliary loss: endpoint consistency ──────────────────────────────
        # pred_sum[k] = Σ_t pred_disp[k,t] * valid[t]  ≈ predicted total displacement
        # gt_sum      = Σ_t gt_disp[t]     * valid[t]  = gt_abs[last_valid] - anchor
        #
        # Gradient: d(ep_loss)/d pred_disp[k,t] = d(SmoothL1)/d pred_sum[k] * valid[t]
        # Constant across t — no cumulative amplification (O(1) per step).
        if self.endpoint_weight > 0.0:
            valid_f    = valid_mask.unsqueeze(-1).float()                       # [B, N, T, 1]
            pred_sum   = (predicted_trajs * valid_f.unsqueeze(2)).sum(dim=-2)   # [B, N, K, 2]
            gt_sum     = (gt_disp * valid_f).sum(dim=-2)                        # [B, N, 2]
            gt_sum_exp = gt_sum.unsqueeze(2).expand(-1, -1, K, -1)              # [B, N, K, 2]

            ep_per_mode = self.reg_loss(pred_sum, gt_sum_exp).sum(dim=-1)       # [B, N, K]
            ep_agent    = (ep_per_mode * mode_weights).sum(dim=-1)              # [B, N]
            ep_loss     = (ep_agent * train_mask.float()).sum() / num_train_agents
        else:
            ep_loss = reg_loss.new_zeros(())

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

        total_loss = reg_loss + self.endpoint_weight * ep_loss + self.cls_weight * cls_loss

        return {"loss": total_loss, "reg_loss": reg_loss,
                "ep_loss": ep_loss, "cls_loss": cls_loss}
