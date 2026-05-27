import torch
import torch.nn as nn


class LLMMotionLoss(nn.Module):
    def __init__(self, device, soft_wta_alpha=0.1):
        """
        soft_wta_alpha: weight for non-winner modes in soft WTA.
            Winner mode always gets weight 1.0.
            0.0 = hard WTA (not recommended for K>1, causes mode collapse).
        """
        super().__init__()
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="none")
        self.soft_wta_alpha = soft_wta_alpha

    def forward(self, predicted_trajs, data):
        """
        predicted_trajs: [B, N, K, T, 2]  per-step displacement predictions
        data: dict with:
            gt_abs_trajectories [B, N, T, 2]  absolute GT positions
            gt_anchor           [B, N, 2]     agent position at t=0
            gt_masks            [B, N, T]     valid future timestep mask
            agent_valid         [B, N]        all present agents (social attention context)
            train_mask          [B, N]        focal + scored agents only (loss supervision)

        Winner selection: FDE-based (L2 at last valid timestep).
        All K modes receive gradient; winner weight=1.0, others weight=soft_wta_alpha.
        Loss is averaged over train_mask agents only (focal + score categories).
        """
        B, N, K, T, _ = predicted_trajs.shape

        gt_abs      = data["gt_abs_trajectories"].to(self.device, dtype=predicted_trajs.dtype)  # [B, N, T, 2]
        anchor      = data["gt_anchor"].to(self.device, dtype=predicted_trajs.dtype)            # [B, N, 2]
        gt_masks    = data["gt_masks"].to(self.device)                                          # [B, N, T]
        train_mask  = data["train_mask"].to(self.device)                                        # [B, N] focal+score only

        valid_mask = gt_masks & train_mask.unsqueeze(-1)  # [B, N, T]

        # ── Displacement → absolute coordinates ──────────────────────────────
        # anchor: [B, N, 2] → [B, N, 1, 1, 2] for broadcasting
        pred_abs = anchor.unsqueeze(2).unsqueeze(2) + predicted_trajs.cumsum(dim=-2)  # [B, N, K, T, 2]

        # ── FDE-based winner selection ────────────────────────────────────────
        # last valid timestep index per agent [B, N]
        valid_float = valid_mask.float()
        last_valid_idx = (valid_float.cumsum(dim=-1) * valid_float).argmax(dim=-1)  # [B, N]
        has_any_valid  = valid_mask.any(dim=-1)
        last_valid_idx = torch.where(has_any_valid, last_valid_idx,
                                     torch.full_like(last_valid_idx, T - 1))

        # pred at last valid step: [B, N, K, 2]
        last_exp  = last_valid_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, N, K, 1, 2)
        pred_last = pred_abs.gather(3, last_exp).squeeze(3)   # [B, N, K, 2]

        # GT at last valid step: [B, N, 2]
        gt_last_exp = last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, 2)
        gt_last     = gt_abs.gather(2, gt_last_exp).squeeze(2)  # [B, N, 2]

        fde      = (pred_last - gt_last.unsqueeze(2)).norm(dim=-1)  # [B, N, K]
        best_idx = fde.argmin(dim=-1)                               # [B, N]

        # ── Soft WTA mode weights ─────────────────────────────────────────────
        mode_weights = torch.full((B, N, K), self.soft_wta_alpha, device=predicted_trajs.device)
        mode_weights.scatter_(2, best_idx.unsqueeze(-1), 1.0)  # winner → 1.0

        # ── Regression loss on absolute coordinates ───────────────────────────
        gt_abs_exp = gt_abs.unsqueeze(2).expand(-1, -1, K, -1, -1)   # [B, N, K, T, 2]
        valid_exp  = valid_mask.unsqueeze(2).expand(-1, -1, K, -1)   # [B, N, K, T]

        per_step = self.reg_loss(pred_abs, gt_abs_exp).sum(dim=-1)   # [B, N, K, T]
        per_step = per_step * valid_exp.float()

        mode_cnt     = valid_exp.float().sum(dim=-1).clamp(min=1e-9)  # [B, N, K]
        per_mode_ade = per_step.sum(dim=-1) / mode_cnt                # [B, N, K]  (SmoothL1 ADE)

        # weighted sum over modes, then average over focal+scored agents only
        agent_loss  = (per_mode_ade * mode_weights).sum(dim=-1)       # [B, N]
        agent_loss  = agent_loss * train_mask.float()

        num_train_agents = train_mask.float().sum().clamp(min=1e-9)
        reg_loss = agent_loss.sum() / num_train_agents

        return {"loss": reg_loss, "reg_loss": reg_loss}
