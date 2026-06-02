"""
Single-GPU training script for the unified-prompt LLM motion predictor.
Use this for local development and testing on Windows / single-GPU machines.

For multi-GPU Linux training, use train_ddp.py with torchrun.

Launch:
    python train_llm.py --features_dir data_av2/features [options]
"""
import os
import glob
import sys
import time
import argparse
from datetime import datetime
from collections import deque
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data_av2.av2_llm_dataset import AV2PromptDataset
from simpl.llm_motion_model import UnifiedLLMMotionPredictor
from simpl.av2_llm_loss import LLMMotionLoss
from utils.logger import Logger
from utils.utils import AverageMeterForDict, set_seed, save_ckpt


def dynamic_pad_collate(batch):
    """Trim unified input_ids / attention_mask to the longest real token in the batch."""
    max_len = int(max(item["attention_mask"].sum().item() for item in batch))
    for item in batch:
        item["input_ids"]       = item["input_ids"][:max_len]
        item["attention_mask"]  = item["attention_mask"][:max_len]
        item["agent_positions"] = item["agent_positions"].clamp(0, max_len - 1)
    return torch.utils.data.dataloader.default_collate(batch)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True, type=str)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size",   type=int, default=1)
    parser.add_argument("--train_epoches",    type=int, default=10)
    parser.add_argument("--val_interval",     type=int, default=1)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--logger_writer",    action="store_true")
    parser.add_argument("--no_pbar",          action="store_true")
    parser.add_argument("--num_workers",      type=int, default=0,
                        help="DataLoader workers. Keep 0 on Windows to avoid spawn issues.")

    # Model
    parser.add_argument("--model_name",    type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--num_modes",     type=int, default=6)
    parser.add_argument("--max_seq_len",   type=int, default=1024,
                        help="Reduce to 1024-2048 on 8GB GPUs to keep activation memory in check.")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="If >0, only use first N training scenes (quick smoke-test).")
    # LoRA
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Loss
    parser.add_argument("--soft_wta_alpha",  type=float, default=0.1)
    parser.add_argument("--endpoint_weight", type=float, default=0.1)

    # Optimiser
    parser.add_argument("--llm_lr",    type=float, default=1e-5)
    parser.add_argument("--gru_lr",    type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Scheduler
    parser.add_argument("--T_0",    type=int,   default=20)
    parser.add_argument("--T_mult", type=int,   default=1)
    parser.add_argument("--warmup_epochs",   type=int,   default=2)
    parser.add_argument("--eta_min_ratio",   type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--print_interval",      type=int, default=10)
    parser.add_argument("--running_window",       type=int, default=20)
    parser.add_argument("--ckpt_dir",             type=str, default="saved_models/")
    parser.add_argument("--save_last_every_epoch",action="store_true", default=True)

    return parser.parse_args()


def format_lr(optimizer):
    parts = []
    for i, pg in enumerate(optimizer.param_groups):
        name = ["llm", "gru"][i] if i < 2 else f"g{i}"
        parts.append(f"{name}_lr={pg['lr']:.2e}")
    return ", ".join(parts)


def main():
    args = parse_arguments()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = Logger(date_str=date_str, log_dir=f"log/{date_str}_llm",
                    enable_flags={"writer": args.logger_writer})
    logger.log_basics(args=args, datetime=date_str)
    logger.print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_set = AV2PromptDataset(
        os.path.join(args.features_dir, "train"),
        tokenizer_name=args.model_name,
        max_len_per_agent=args.max_seq_len,
    )
    val_set = AV2PromptDataset(
        os.path.join(args.features_dir, "val"),
        tokenizer_name=args.model_name,
        max_len_per_agent=args.max_seq_len,
    )
    if args.max_train_samples > 0:
        train_set = torch.utils.data.Subset(train_set, range(min(args.max_train_samples, len(train_set))))
    logger.print(f"Train: {len(train_set)} scenes | Val: {len(val_set)} scenes | "
                 f"max_seq_len={args.max_seq_len}")

    dl_train = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                          collate_fn=dynamic_pad_collate)
    dl_val   = DataLoader(val_set,   batch_size=args.val_batch_size,   shuffle=False,
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                          collate_fn=dynamic_pad_collate)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UnifiedLLMMotionPredictor(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_targets.split(","),
        lora_dropout=args.lora_dropout,
        num_modes=args.num_modes,
        device=device,
        use_flash_attn=False,
        dtype=torch.bfloat16,
    )
    model = model.to(device).to(torch.bfloat16)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.print(f"Params: trainable={n_train:,} / total={n_total:,} "
                 f"({100.*n_train/max(n_total,1):.2f}%)")

    loss_fn = LLMMotionLoss(device=device, soft_wta_alpha=args.soft_wta_alpha,
                            endpoint_weight=args.endpoint_weight)
    logger.print(f"Loss: SmoothL1(disp) + ep_loss(w={args.endpoint_weight}) | "
                 f"soft_wta_alpha={args.soft_wta_alpha}")

    # ── Optimiser (differential LR) ───────────────────────────────────────────
    llm_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "llm." in n]
    gru_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "llm." not in n]
    optimizer = AdamW([
        {"params": llm_params, "lr": args.llm_lr},
        {"params": gru_params, "lr": args.gru_lr},
    ], weight_decay=1e-4)
    logger.print(f"Optimiser: llm_params={sum(p.numel() for p in llm_params):,} "
                 f"| gru_params={sum(p.numel() for p in gru_params):,}")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_ckpt = f"{date_str}_llm_simpl_best.tar"
    last_ckpt = f"{date_str}_llm_simpl_last.tar"
    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.train_epoches):
        logger.print(f"\n{'='*70}")
        logger.print(f"Epoch {epoch+1}/{args.train_epoches}  |  {format_lr(optimizer)}")
        logger.print(f"{'='*70}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_meter = AverageMeterForDict()
        loss_window = deque(maxlen=args.running_window)
        epoch_start = time.time()

        pbar = tqdm(dl_train, disable=args.no_pbar, ncols=110,
                    desc=f"[Train ep {epoch+1}]")
        for i, data in enumerate(pbar):
            input_ids       = data["input_ids"].to(device)
            attention_mask  = data["attention_mask"].to(device)
            agent_positions = data["agent_positions"].to(device)
            agent_valid     = data["agent_valid"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(input_ids, attention_mask, agent_positions, agent_valid)
                loss_out = loss_fn(pred, data)

            loss_out["loss"].backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            scalars = {k: v.item() for k, v in loss_out.items()}
            train_meter.update(scalars)
            loss_window.append(scalars["loss"])

            pbar.set_postfix({
                "loss": f"{scalars['loss']:.4f}",
                f"avg{args.running_window}": f"{sum(loss_window)/len(loss_window):.4f}",
                "gnorm": f"{gnorm.item():.2f}",
            })

            if (i + 1) % args.print_interval == 0 or (i + 1) == len(dl_train):
                mem = torch.cuda.memory_allocated(device) // 2**20
                peak = torch.cuda.max_memory_allocated(device) // 2**20
                elapsed = time.time() - epoch_start
                its = (i + 1) / max(elapsed, 1e-6)
                logger.print(
                    f"  [ep {epoch+1} | {i+1}/{len(dl_train)}] "
                    f"loss={scalars['loss']:.4f} "
                    f"run_avg={sum(loss_window)/len(loss_window):.4f} | "
                    f"gnorm={gnorm.item():.2f} | "
                    f"{its:.2f} it/s | "
                    f"mem={mem}MB peak={peak}MB"
                )

        pbar.close()
        scheduler.step()
        peak_mem = torch.cuda.max_memory_allocated(device) // 2**20
        logger.print(
            f"[Train] ep {epoch+1} avg_loss={train_meter.metrics['loss'].avg:.4f} | "
            f"time={( time.time()-epoch_start)/60:.2f}min | peak_mem={peak_mem}MB"
        )

        if args.save_last_every_epoch:
            save_ckpt(model, optimizer, epoch, args.ckpt_dir, last_ckpt)

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % args.val_interval != 0:
            continue

        model.eval()
        val_meter  = AverageMeterForDict()
        ade_meter  = AverageMeterForDict()
        val_start  = time.time()

        val_pbar = tqdm(dl_val, disable=args.no_pbar, ncols=110,
                        desc=f"[Val   ep {epoch+1}]")
        with torch.no_grad():
            for data in val_pbar:
                input_ids       = data["input_ids"].to(device)
                attention_mask  = data["attention_mask"].to(device)
                agent_positions = data["agent_positions"].to(device)
                agent_valid     = data["agent_valid"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred     = model(input_ids, attention_mask, agent_positions, agent_valid)
                    loss_out = loss_fn(pred, data)

                val_meter.update({k: v.item() for k, v in loss_out.items()})

                # minADE / minFDE — multi-modal, focal agent (index 0)
                pred_disp = pred[:, 0].float()                               # [B, K, T, 2]
                anchor    = data["gt_anchor"][:, 0].to(device).float()       # [B, 2]
                pred_abs  = anchor.unsqueeze(1).unsqueeze(1) + \
                            pred_disp.cumsum(dim=-2)                         # [B, K, T, 2]
                gt_abs    = data["gt_abs_trajectories"][:, 0].to(device).float()  # [B, T, 2]
                mask      = data["gt_masks"][:, 0].to(device)               # [B, T]
                B         = pred_abs.shape[0]

                last_t = (mask.long().cumsum(1) * mask.long()).argmax(1)
                gt_last = gt_abs[torch.arange(B), last_t]                   # [B, 2]
                pred_last = pred_abs[torch.arange(B), :, last_t]            # [B, K, 2]
                fde_k = (pred_last - gt_last.unsqueeze(1)).norm(dim=-1)     # [B, K]
                best_k = fde_k.argmin(dim=1)                                # [B]

                pred_best = pred_abs[torch.arange(B), best_k]               # [B, T, 2]
                l2 = (pred_best - gt_abs).norm(dim=-1)                      # [B, T]
                valid_cnt = mask.float().sum(1).clamp(min=1)
                min_ade = ((l2 * mask.float()).sum(1) / valid_cnt).mean()
                min_fde = l2[torch.arange(B), last_t].mean()

                ade_meter.update({"minADE": min_ade.item(), "minFDE": min_fde.item()})
                val_pbar.set_postfix({
                    "loss":   f"{val_meter.metrics['loss'].avg:.4f}",
                    "minADE": f"{ade_meter.metrics['minADE'].avg:.4f}",
                })

        val_pbar.close()
        val_loss = val_meter.metrics["loss"].avg
        min_ade  = ade_meter.metrics["minADE"].avg
        min_fde  = ade_meter.metrics["minFDE"].avg
        logger.print(
            f"[Val] ep {epoch+1} | loss={val_loss:.4f} | "
            f"minADE={min_ade:.4f}m | minFDE={min_fde:.4f}m | "
            f"time={( time.time()-val_start)/60:.2f}min"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(model, optimizer, epoch, args.ckpt_dir, best_ckpt)
            logger.print(f"  >> New best ({val_loss:.4f}) — saved {best_ckpt}")
        else:
            logger.print(f"  -- No improvement (best={best_val_loss:.4f})")

    logger.print(f"\nDone. Best val loss = {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
