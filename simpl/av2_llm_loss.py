import torch
import torch.nn as nn

class LLMMotionLoss(nn.Module):
    def __init__(self, device, pos_loss_weight=1.0, wta_epsilon=0.2):
        """
        pos_loss_weight: weight for cumulative position reconstruction loss.
            Set to 0.0 to use displacement loss only.
        wta_epsilon: probability of selecting a random mode instead of the best
            mode during training (epsilon-greedy WTA). Prevents mode collapse
            when K>1. Automatically disabled during validation (no_grad context).
            Set to 0.0 to use pure WTA (not recommended for K>1).
        """
        super().__init__()
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="none")
        self.pos_loss_weight = pos_loss_weight
        self.wta_epsilon = wta_epsilon

    def forward(self, predicted_trajs, data):
        """
        predicted_trajs: [B, N, K, T, 2]  K=num_modes, T=num_future_steps
        data: dict with:
            gt_trajectories     [B, N, T, 2]
            gt_masks            [B, N, T]
            agent_valid         [B, N]
            gt_abs_trajectories [B, N, T, 2]
            gt_anchor           [B, N, 2]

        Training: epsilon-greedy WTA over K modes (pure WTA when epsilon=0).
        Validation: pure WTA (torch.is_grad_enabled() == False → epsilon ignored).
        """
        B, N, K, T, _ = predicted_trajs.shape

        gt         = data["gt_trajectories"].to(self.device, dtype=predicted_trajs.dtype)
        gt_masks   = data["gt_masks"].to(self.device)       # [B, N, T]
        agent_valid = data["agent_valid"].to(self.device)   # [B, N]

        valid_mask = gt_masks & agent_valid.unsqueeze(-1)   # [B, N, T]
        num_valid  = valid_mask.float().sum()

        # ── Per-mode average displacement loss ───────────────────────────────
        gt_exp    = gt.unsqueeze(2).expand(-1, -1, K, -1, -1)            # [B, N, K, T, 2]
        valid_exp = valid_mask.unsqueeze(2).expand(-1, -1, K, -1)        # [B, N, K, T]

        per_step = self.reg_loss(predicted_trajs, gt_exp).sum(dim=-1)    # [B, N, K, T]
        per_step = per_step * valid_exp.float()
        mode_cnt = valid_exp.float().sum(dim=-1).clamp(min=1e-9)         # [B, N, K]
        per_mode = per_step.sum(dim=-1) / mode_cnt                       # [B, N, K]

        # ── Winner selection (epsilon-greedy during training) ─────────────────
        best_idx = per_mode.argmin(dim=-1)                               # [B, N]

        if torch.is_grad_enabled() and self.wta_epsilon > 0 and K > 1:
            # With prob epsilon, override with a uniformly random mode
            rand_mask = torch.rand(B, N, device=predicted_trajs.device) < self.wta_epsilon
            rand_idx  = torch.randint(0, K, (B, N), device=predicted_trajs.device)
            best_idx  = torch.where(rand_mask, rand_idx, best_idx)

        # ── Gather winning-mode predictions ───────────────────────────────────
        idx_exp   = best_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) \
                             .expand(-1, -1, 1, T, 2)                    # [B, N, 1, T, 2]
        best_pred = predicted_trajs.gather(2, idx_exp).squeeze(2)       # [B, N, T, 2]

        # ── Displacement loss (winning mode) ──────────────────────────────────
        disp_raw    = self.reg_loss(best_pred, gt).sum(dim=-1)           # [B, N, T]
        disp_masked = disp_raw * valid_mask.float()

        if num_valid > 0:
            disp_loss = disp_masked.sum() / num_valid
        else:
            disp_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # ── Position reconstruction loss (winning mode) ───────────────────────
        use_pos_loss = (
            self.pos_loss_weight > 0.0
            and "gt_abs_trajectories" in data
            and "gt_anchor" in data
        )

        if use_pos_loss:
            gt_abs = data["gt_abs_trajectories"].to(self.device, dtype=predicted_trajs.dtype)
            anchor = data["gt_anchor"].to(self.device, dtype=predicted_trajs.dtype)  # [B, N, 2]

            pred_abs   = anchor.unsqueeze(-2) + best_pred.cumsum(dim=-2)  # [B, N, T, 2]
            pos_raw    = self.reg_loss(pred_abs, gt_abs).sum(dim=-1)       # [B, N, T]
            pos_masked = pos_raw * valid_mask.float()

            if num_valid > 0:
                pos_loss = pos_masked.sum() / num_valid
            else:
                pos_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            total_loss = disp_loss + self.pos_loss_weight * pos_loss
            return {"loss": total_loss, "disp_loss": disp_loss, "pos_loss": pos_loss}
        else:
            return {"loss": disp_loss, "disp_loss": disp_loss}
