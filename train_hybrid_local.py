"""
Single-GPU local training script for the Hybrid LLM Motion Predictor.
Use this for local development / smoke-testing on Windows.

For multi-GPU Linux training, use train_hybrid.py with torchrun.

Launch:
    python train_hybrid_local.py --features_dir data_av2/features [options]
"""
import os
import sys
import time
import argparse
import contextlib
import math
from datetime import datetime
from collections import deque
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data_av2.av2_hybrid_dataset import AV2HybridDataset
from simpl.hybrid_llm_model import HybridLLMPredictor
from simpl.hybrid_loss import HybridMotionLoss
from utils.logger import Logger
from utils.utils import AverageMeterForDict, set_seed, save_ckpt


# ── Collate ───────────────────────────────────────────────────────────────────

def hybrid_collate_fn(batch):
    max_len = max(item["text_input_ids"].shape[0] for item in batch)
    for item in batch:
        cur = item["text_input_ids"].shape[0]
        pad = max_len - cur
        if pad > 0:
            item["text_input_ids"]      = F.pad(item["text_input_ids"],      (0, pad), value=0)
            item["text_attention_mask"] = F.pad(item["text_attention_mask"], (0, pad), value=0)
    return torch.utils.data.dataloader.default_collate(batch)


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_dir",      required=True, type=str)
    parser.add_argument("--ckpt_dir",          type=str, default="saved_models/")

    # Dataset
    parser.add_argument("--max_text_len",  type=int, default=500)
    parser.add_argument("--max_agents",    type=int, default=6)
    parser.add_argument("--max_lanes",     type=int, default=20)

    # Training
    parser.add_argument("--train_batch_size",  type=int,   default=8)
    parser.add_argument("--val_batch_size",    type=int,   default=8)
    parser.add_argument("--train_epoches",     type=int,   default=20)
    parser.add_argument("--val_interval",      type=int,   default=20)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--num_workers",       type=int,   default=0,
                        help="Keep 0 on Windows to avoid spawn issues.")
    parser.add_argument("--max_train_samples", type=int,   default=50,
                        help="Limit training set for smoke-test. 0 = full dataset.")
    parser.add_argument("--no_pbar",           action="store_true")
    parser.add_argument("--logger_writer",     action="store_true")

    # Model
    parser.add_argument("--model_name",    type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--num_modes",     type=int, default=6)
    parser.add_argument("--n_levels",      type=int, default=2)
    parser.add_argument("--n_bezier_ctrl", type=int, default=6,
                        help="Number of Bezier control points per mode (degree K-1).")

    # LoRA
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Loss (Bezier: reg_loss is position-space SmoothL1, no cumsum; pos/endpoint
    #        weights removed — they were displacement-era terms)
    parser.add_argument("--soft_wta_alpha",    type=float, default=0.1)
    parser.add_argument("--cls_weight",        type=float, default=0.5)
    parser.add_argument("--anchor_cls_weight", type=float, default=0.5)

    # Optimiser
    parser.add_argument("--llm_lr",           type=float, default=5e-5)
    parser.add_argument("--gru_lr",           type=float, default=1e-4)
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int,   default=1)

    # Scheduler
    parser.add_argument("--T_0",           type=int,   default=20)
    parser.add_argument("--warmup_epochs", type=int,   default=2)
    parser.add_argument("--eta_min_ratio", type=float, default=0.1)

    # Logging
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--running_window",  type=int, default=20)

    return parser.parse_args()


def format_lr(optimizer):
    parts = []
    names = ["llm", "enc+dec"]
    for i, pg in enumerate(optimizer.param_groups):
        n = names[i] if i < len(names) else f"g{i}"
        parts.append(f"{n}_lr={pg['lr']:.2e}")
    return ", ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_arguments()
    set_seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger   = Logger(date_str=date_str,
                      log_dir=f"log/{date_str}_hybrid_local",
                      enable_flags={"writer": args.logger_writer})
    logger.log_basics(args=args, datetime=date_str)
    logger.print(f"Device: {device}")
    if device.type == "cuda":
        logger.print(f"GPU: {torch.cuda.get_device_name()} | "
                     f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 2**20} MB")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_set = AV2HybridDataset(
        os.path.join(args.features_dir, "train"),
        tokenizer_name=args.model_name,
        max_agents=args.max_agents, max_lanes=args.max_lanes,
        max_text_len=args.max_text_len)
    val_set = AV2HybridDataset(
        os.path.join(args.features_dir, "val"),
        tokenizer_name=args.model_name,
        max_agents=args.max_agents, max_lanes=args.max_lanes,
        max_text_len=args.max_text_len)

    if args.max_train_samples > 0:
        train_set = torch.utils.data.Subset(
            train_set, range(min(args.max_train_samples, len(train_set))))

    logger.print(f"Train: {len(train_set)} | Val: {len(val_set)} | "
                 f"max_text_len={args.max_text_len}")

    dl_train = DataLoader(train_set, batch_size=args.train_batch_size,
                          shuffle=True, num_workers=args.num_workers,
                          pin_memory=(device.type == "cuda"),
                          collate_fn=hybrid_collate_fn)
    dl_val   = DataLoader(val_set,   batch_size=args.val_batch_size,
                          shuffle=False, num_workers=args.num_workers,
                          pin_memory=(device.type == "cuda"),
                          collate_fn=hybrid_collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HybridLLMPredictor(
        model_name=args.model_name,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_targets if args.lora_targets == "all-linear"
                            else args.lora_targets.split(","),
        lora_dropout=args.lora_dropout,
        n_levels=args.n_levels,
        max_agents=args.max_agents, max_lanes=args.max_lanes,
        num_modes=args.num_modes, n_bezier_ctrl=args.n_bezier_ctrl,
        use_flash_attn=False,       # disabled for local compatibility
        dtype=torch.bfloat16)
    model = model.to(device).to(torch.bfloat16)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.print(f"Params: trainable={n_train:,} / total={n_total:,} "
                 f"({100.*n_train/max(n_total,1):.2f}%)")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = HybridMotionLoss(n_levels=args.n_levels, device=device,
                               soft_wta_alpha=args.soft_wta_alpha,
                               cls_weight=args.cls_weight,
                               anchor_cls_weight=args.anchor_cls_weight)
    logger.print(f"Loss: HybridMotionLoss n_levels={args.n_levels} | "
                 f"level_weights={loss_fn.level_weights}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    _slow_keywords = ("llm.",  "llm_correction_proj")
    llm_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and any(k in n for k in _slow_keywords)]
    enc_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and not any(k in n for k in _slow_keywords)]
    optimizer = AdamW([
        {"params": llm_params, "lr": args.llm_lr},
        {"params": enc_params, "lr": args.gru_lr},
    ], weight_decay=1e-4)
    logger.print(f"Optimiser: llm={sum(p.numel() for p in llm_params):,} "
                 f"| enc+dec={sum(p.numel() for p in enc_params):,}")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    _T0, _wu, _r = args.T_0, args.warmup_epochs, args.eta_min_ratio
    def _lr_lambda(epoch):
        cp = epoch % _T0
        if cp < _wu:
            return _r + (1.0 - _r) * cp / max(1, _wu)
        prog = (cp - _wu) / max(1, _T0 - _wu)
        return _r + (1.0 - _r) * 0.5 * (1.0 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_ckpt     = f"{date_str}_hybrid_local_best.tar"
    best_val_loss = float("inf")

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(args.train_epoches):
        logger.print(f"\n{'='*70}")
        logger.print(f"Epoch {epoch+1}/{args.train_epoches}  |  {format_lr(optimizer)}")
        logger.print(f"{'='*70}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        meter       = AverageMeterForDict()
        loss_window = deque(maxlen=args.running_window)
        epoch_start = time.time()
        accum       = args.grad_accum_steps
        grad_norm   = torch.tensor(0.0)
        optimizer.zero_grad()

        pbar = tqdm(dl_train, disable=args.no_pbar, ncols=120,
                    desc=f"[Train ep {epoch+1}]")

        for i, data in enumerate(pbar):
            text_ids  = data["text_input_ids"].to(device)
            text_mask = data["text_attention_mask"].to(device)
            ag_feat   = data["agent_features"].to(device)
            ag_valid  = data["agent_valid"].to(device)
            lane_feat = data["lane_features"].to(device)
            lane_valid= data["lane_valid"].to(device)
            gt_anchor = data["gt_anchor"].to(device)

            is_last = ((i + 1) % accum == 0) or ((i + 1) == len(dl_train))

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                all_preds, all_scores, codebook = model(text_ids, text_mask, ag_feat, ag_valid,
                                                        lane_feat, lane_valid, gt_anchor)
                loss_out  = loss_fn(all_preds, data, all_scores, codebook)

            (loss_out["loss"] / accum).backward()

            if is_last:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            scalars = {k: v.item() for k, v in loss_out.items()}
            meter.update(scalars)
            loss_window.append(scalars["loss"])

            gnorm     = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            run_avg   = sum(loss_window) / len(loss_window)
            pbar.set_postfix({
                "loss": f"{scalars['loss']:.4f}",
                f"avg{len(loss_window)}": f"{run_avg:.4f}",
                "gnorm": f"{gnorm:.2f}",
            })

            if (i + 1) % args.print_interval == 0 or (i + 1) == len(dl_train):
                elapsed = time.time() - epoch_start
                ips     = (i + 1) / max(elapsed, 1e-6)
                mem     = torch.cuda.memory_allocated(device) // 2**20
                peak    = torch.cuda.max_memory_allocated(device) // 2**20
                logger.print(
                    f"  [ep {epoch+1} | {i+1}/{len(dl_train)}] "
                    f"loss={scalars['loss']:.4f} avg={run_avg:.4f} | "
                    f"gnorm={gnorm:.2f} | {ips:.2f} it/s | "
                    f"mem={mem}MB peak={peak}MB"
                )

        pbar.close()
        scheduler.step()
        peak = torch.cuda.max_memory_allocated(device) // 2**20
        logger.print(
            f"[Train] ep {epoch+1} avg_loss={meter.metrics['loss'].avg:.4f} | "
            f"time={(time.time()-epoch_start)/60:.2f}min | peak_mem={peak}MB"
        )

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % args.val_interval != 0:
            continue

        model.eval()
        val_meter = AverageMeterForDict()
        ade_meter = AverageMeterForDict()
        val_start = time.time()

        pbar_v = tqdm(dl_val, disable=args.no_pbar, ncols=120,
                      desc=f"[Val   ep {epoch+1}]")
        with torch.no_grad():
            for data in pbar_v:
                text_ids  = data["text_input_ids"].to(device)
                text_mask = data["text_attention_mask"].to(device)
                ag_feat   = data["agent_features"].to(device)
                ag_valid  = data["agent_valid"].to(device)
                lane_feat = data["lane_features"].to(device)
                lane_valid= data["lane_valid"].to(device)
                gt_anchor = data["gt_anchor"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    all_preds, all_scores, codebook = model(text_ids, text_mask, ag_feat, ag_valid,
                                                            lane_feat, lane_valid, gt_anchor)
                    loss_out  = loss_fn(all_preds, data, all_scores, codebook)

                val_meter.update({k: v.item() for k, v in loss_out.items()})

                # minADE / minFDE (focal agent, final level)
                # Bezier: all_preds[-1] are direct position offsets (no cumsum)
                pred_pos  = all_preds[-1][:, 0].float()
                anchor_f  = gt_anchor[:, 0].float()
                pred_abs  = anchor_f.unsqueeze(1).unsqueeze(1) + pred_pos
                gt_abs_f  = data["gt_abs_trajectories"][:, 0].to(device).float()
                mask_f    = data["gt_masks"][:, 0].to(device)
                B_        = pred_abs.shape[0]
                last_t    = (mask_f.long().cumsum(1) * mask_f.long()).argmax(1)
                gt_last   = gt_abs_f[torch.arange(B_), last_t]
                pred_last = pred_abs[torch.arange(B_), :, last_t]
                fde_k     = (pred_last - gt_last.unsqueeze(1)).norm(dim=-1)
                best_k    = fde_k.argmin(dim=1)
                pred_best = pred_abs[torch.arange(B_), best_k]
                l2        = (pred_best - gt_abs_f).norm(dim=-1)
                valid_cnt = mask_f.float().sum(1).clamp(min=1)
                min_ade   = ((l2 * mask_f.float()).sum(1) / valid_cnt).mean()
                min_fde   = l2[torch.arange(B_), last_t].mean()
                ade_meter.update({"minADE": min_ade.item(), "minFDE": min_fde.item()})
                pbar_v.set_postfix({
                    "loss":   f"{val_meter.metrics['loss'].avg:.4f}",
                    "minADE": f"{ade_meter.metrics['minADE'].avg:.4f}",
                })

        pbar_v.close()
        val_loss = val_meter.metrics["loss"].avg
        min_ade  = ade_meter.metrics["minADE"].avg
        min_fde  = ade_meter.metrics["minFDE"].avg
        logger.print(
            f"[Val] ep {epoch+1} | loss={val_loss:.4f} | "
            f"minADE={min_ade:.4f}m | minFDE={min_fde:.4f}m | "
            f"time={(time.time()-val_start)/60:.2f}min"
        )

        # ── LLM / lane branch contribution (is the LLM actually working?) ─────
        _corr = getattr(model, "diag_corr_ratio", float("nan"))
        _lane = getattr(model, "diag_lane_ratio", float("nan"))
        logger.print(f"[Branch] ep {epoch+1} | LLM corr/enc: {_corr:.4f} "
                     f"| lane_cross/input: {_lane:.4f}")

        # ── Anchor codebook diagnostics ──────────────────────────────────────
        cb      = model.traj_decoder.codebook
        anc     = cb.anchors.float().cpu()                          # [K, 2]
        cnt     = cb.ema_count.float().cpu()                        # [K]
        pdist   = torch.cdist(anc, anc)
        pdist.fill_diagonal_(float("inf"))
        min_sep = pdist.min().item()
        logger.print(f"[Anchors] ep {epoch+1} | min pairwise sep: {min_sep:.3f} m")
        for j in range(anc.shape[0]):
            logger.print(f"    anchor[{j}] = ({anc[j,0]:+7.2f}, {anc[j,1]:+7.2f}) "
                         f"| ema_count = {cnt[j]:8.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(model, optimizer, epoch, args.ckpt_dir, best_ckpt)
            logger.print(f"  >> New best ({val_loss:.4f}) — saved {best_ckpt}")
        else:
            logger.print(f"  -- No improvement (best={best_val_loss:.4f})")

    logger.print(f"\nDone. Best val loss = {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
