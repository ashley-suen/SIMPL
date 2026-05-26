import os
import glob
import time
import argparse
from datetime import datetime
from collections import deque
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data_av2.av2_llm_dataset import AV2PromptDataset
from simpl.llm_motion_model import SmolLMMotionPredictor
from simpl.av2_llm_loss import LLMMotionLoss
from utils.logger import Logger
from utils.utils import AverageMeterForDict, set_seed, save_ckpt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Val batch size")
    parser.add_argument("--train_epoches", type=int, default=10, help="Number of epoches for training")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation intervals")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", default=True, help="Use CUDA for acceleration")
    parser.add_argument("--logger_writer", action="store_true", help="Enable tensorboard")
    parser.add_argument("--no_pbar", action="store_true", help="Hide progress bar")

    # LLM Specific Args
    parser.add_argument("--llm_lr", type=float, default=1e-5, help="Learning rate for LLM layers")
    parser.add_argument("--mlp_lr", type=float, default=1e-4, help="Learning rate for MLP head")
    parser.add_argument("--unfreeze_layers", type=int, default=1, help="Number of LLM layers to unfreeze")

    # Loss
    parser.add_argument("--pos_loss_weight", type=float, default=1.0,
                        help="[Exp2] Weight for position reconstruction loss. "
                             "Set to 0.0 to disable (displacement-only, Exp1 behaviour).")

    # LR Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine_restart",
                        choices=["cosine", "cosine_restart"],
                        help="LR scheduler: cosine (no restart) or cosine_restart (warm restarts)")
    parser.add_argument("--T_0", type=int, default=20,
                        help="[cosine_restart] Epochs per restart cycle")
    parser.add_argument("--T_mult", type=int, default=1,
                        help="[cosine_restart] Cycle length multiplier after each restart")

    # Logging frequency control
    parser.add_argument("--print_interval", type=int, default=50,
                        help="Print intermediate training stats every N iterations")
    parser.add_argument("--running_window", type=int, default=50,
                        help="Window size (in iters) for running average loss display")

    # Saving control
    parser.add_argument("--ckpt_dir", type=str, default="saved_models/",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_last_every_epoch", action="store_true", default=True,
                        help="Save a 'last' checkpoint at the end of every epoch")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (increase on multi-core servers)")

    return parser.parse_args()


def format_lr(optimizer):
    """Format current LRs of each param group as a readable string."""
    parts = []
    names = ["llm", "social", "mlp"]
    for i, pg in enumerate(optimizer.param_groups):
        name = names[i] if i < len(names) else f"g{i}"
        parts.append(f"{name}_lr={pg['lr']:.2e}")
    return ", ".join(parts)


def count_trainable_params(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def safe_save(logger, model, optimizer, epoch, dirpath, name):
    """Wrap save_ckpt so a failure can never silently swallow our checkpoint."""
    try:
        save_ckpt(model, optimizer, epoch, dirpath, name)
        full_path = os.path.join(dirpath, name)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 ** 2)
            logger.print(f'  >> Saved checkpoint: {full_path} ({size_mb:.1f} MB)')
        else:
            logger.print(f'  !! save_ckpt returned without error but file missing: {full_path}')
        return True
    except Exception as e:
        logger.print(f'  !! save_ckpt FAILED for {name}: {type(e).__name__}: {e}')
        return False


def main():
    args = parse_arguments()
    set_seed(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str + "_llm"
    logger = Logger(date_str=date_str, log_dir=log_dir, enable_flags={'writer': args.logger_writer})
    logger.log_basics(args=args, datetime=date_str)

    # ------ Sanity checks for saving (FAIL FAST instead of after 26 minutes) ------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    probe = os.path.join(args.ckpt_dir, f".write_probe_{date_str}")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except Exception as e:
        raise RuntimeError(
            f"Cannot write to ckpt_dir={args.ckpt_dir!r}: {type(e).__name__}: {e}. "
            f"Aborting before training starts."
        )
    if args.val_interval > args.train_epoches:
        raise ValueError(
            f"val_interval ({args.val_interval}) > train_epoches ({args.train_epoches}); "
            f"validation and best-checkpoint saving would never trigger."
        )
    logger.print(f"Checkpoints will be saved to: {os.path.abspath(args.ckpt_dir)}")

    # 1. Dataset & DataLoader
    train_dir = os.path.join(args.features_dir, 'train')
    val_dir = os.path.join(args.features_dir, 'val')

    logger.print(f"Loading datasets from {train_dir} and {val_dir}...")
    train_set = AV2PromptDataset(train_dir)
    val_set = AV2PromptDataset(val_dir)
    logger.print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    if len(val_set) == 0:
        raise RuntimeError(
            f"Validation set is empty (path={val_dir}). "
            f"Best-checkpoint logic cannot work — aborting."
        )

    dl_train = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    dl_val   = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    iters_per_epoch = len(dl_train)
    val_iters = len(dl_val)
    logger.print(f"Iterations per epoch: train={iters_per_epoch}, val={val_iters}")

    # 2. Model & Loss
    logger.print("Initializing SmolLMMotionPredictor...")
    model = SmolLMMotionPredictor(
        unfreeze_last_n_layers=args.unfreeze_layers,
        device=device,
        use_flash_attn=True,   # Enable Flash Attention 2
        dtype=torch.bfloat16   # Flash Attention 2 works best with bfloat16
    )
    logger.print("Compiling model with torch.compile (first iter will be slower)...")
    model = torch.compile(model)
    loss_fn = LLMMotionLoss(device=device, pos_loss_weight=args.pos_loss_weight)
    logger.print(f"Loss: disp_loss + {args.pos_loss_weight} × pos_loss "
                 f"({'hybrid' if args.pos_loss_weight > 0 else 'displacement-only'})")

    n_trainable, n_total = count_trainable_params(model)
    logger.print(f"Model params: trainable={n_trainable:,} / total={n_total:,} "
                 f"({100.0 * n_trainable / max(n_total, 1):.2f}% trainable)")

    # 3. Optimizer (Differential Learning Rates)
    llm_params    = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("llm.")]
    mlp_params    = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("mlp_head.")]
    social_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("social_attn.")]

    logger.print(f"Trainable groups: llm_params={sum(p.numel() for p in llm_params):,}, "
                 f"social_params={sum(p.numel() for p in social_params):,}, "
                 f"mlp_params={sum(p.numel() for p in mlp_params):,}")

    optimizer = AdamW([
        {'params': llm_params,    'lr': args.llm_lr},
        {'params': social_params, 'lr': args.mlp_lr},
        {'params': mlp_params,    'lr': args.mlp_lr},
    ], weight_decay=1e-4)

    # bf16 autocast: flash-attn requires half precision; bf16 has enough dynamic range
    # so GradScaler is not needed (and breaks torch.compile graph capture).

    # --- [Exp1] Fixed cosine decay, LR → 0 by final epoch ---
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoches)

    # --- [Exp2+] Cosine with warm restarts: LR resets every T_0 epochs so it never stays near zero ---
    if args.scheduler == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=args.T_mult
        )
        logger.print(f"Scheduler: CosineAnnealingWarmRestarts | T_0={args.T_0}, T_mult={args.T_mult}")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoches)
        logger.print(f"Scheduler: CosineAnnealingLR | T_max={args.train_epoches}")

    # 4. Training Loop
    niter = 0
    best_val_loss = float('inf')
    train_start = time.time()
    last_ckpt_name = f'{date_str}_llm_simpl_last.tar'
    best_ckpt_name = f'{date_str}_llm_simpl_best.tar'
    saved_any = False

    for epoch in range(args.train_epoches):
        # ------ Print epoch banner with hyperparameters ------
        logger.print('\n' + '=' * 80)
        logger.print(f'Epoch {epoch + 1}/{args.train_epoches}')
        logger.print(f'  - LR : {format_lr(optimizer)}')
        logger.print(f'  - Batch size (train/val): {args.train_batch_size}/{args.val_batch_size}')
        logger.print(f'  - Unfrozen LLM layers   : {args.unfreeze_layers}')
        logger.print(f'  - AMP (bf16)            : True')
        logger.print(f'  - Grad clip max norm    : 1.0')
        logger.print('=' * 80)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # --- TRAIN ---
        epoch_start = time.time()
        train_loss_meter = AverageMeterForDict()
        loss_window = deque(maxlen=args.running_window)

        model.train()

        pbar = tqdm(dl_train, disable=args.no_pbar, ncols=110,
                    desc=f"[Train ep {epoch + 1}]")

        for i, data in enumerate(pbar):
            input_ids      = data["input_ids"].to(device)       # [B, N, L]
            attention_mask = data["attention_mask"].to(device)   # [B, N, L]
            agent_valid    = data["agent_valid"].to(device)      # [B, N]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predicted_trajs = model(input_ids, attention_mask, agent_valid)
                loss_out = loss_fn(predicted_trajs, data)

            loss = loss_out["loss"]

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scalar_losses = {k: v.item() for k, v in loss_out.items()}
            train_loss_meter.update(scalar_losses)
            loss_window.append(scalar_losses["loss"])

            niter += args.train_batch_size
            logger.add_dict(scalar_losses, niter, prefix='train/')

            running_loss = sum(loss_window) / len(loss_window)
            gnorm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            postfix = {
                "loss": f"{scalar_losses['loss']:.4f}",
                f"loss_avg{len(loss_window)}": f"{running_loss:.4f}",
                "gnorm": f"{gnorm_val:.2f}",
            }
            pbar.set_postfix(postfix)

            # Periodic stdout print so progress is visible even when redirected to a log file.
            if (i + 1) % args.print_interval == 0 or (i + 1) == iters_per_epoch:
                elapsed = time.time() - epoch_start
                it_per_sec = (i + 1) / max(elapsed, 1e-6)
                eta_sec = (iters_per_epoch - (i + 1)) / max(it_per_sec, 1e-6)
                cur_mem = torch.cuda.memory_allocated(device=device) // 2 ** 20

                extra_str = ""
                for k, v in scalar_losses.items():
                    if k == "loss":
                        continue
                    extra_str += f" | {k}={v:.4f}"

                logger.print(
                    f"  [ep {epoch + 1} | iter {i + 1}/{iters_per_epoch}] "
                    f"loss={scalar_losses['loss']:.4f} "
                    f"(run-avg={running_loss:.4f})"
                    f"{extra_str} | "
                    f"gnorm={gnorm_val:.2f} | "
                    f"{it_per_sec:.2f} it/s | "
                    f"eta={eta_sec / 60.0:.1f} min | "
                    f"mem={cur_mem} MB"
                )

        pbar.close()
        scheduler.step()
        max_memory = torch.cuda.max_memory_allocated(device=device) // 2 ** 20
        loss_avg = train_loss_meter.metrics['loss'].avg
        logger.print(
            f'[Training] Epoch {epoch + 1} done | avg loss: {loss_avg:.6f} | '
            f'time: {(time.time() - epoch_start) / 60.0:.3f} mins | '
            f'max mem: {max_memory} MB | next-epoch {format_lr(optimizer)}'
        )

        # ------ Always save a 'last' checkpoint, regardless of val ------
        if args.save_last_every_epoch:
            ok = safe_save(logger, model, optimizer, epoch, args.ckpt_dir, last_ckpt_name)
            saved_any = saved_any or ok

        # --- VALIDATION ---
        if (epoch + 1) % args.val_interval == 0:
            val_start = time.time()
            val_loss_meter = AverageMeterForDict()
            ade_meter = AverageMeterForDict()
            model.eval()

            val_pbar = tqdm(dl_val, disable=args.no_pbar, ncols=110,
                            desc=f"[Val   ep {epoch + 1}]")

            with torch.no_grad():
                for i, data in enumerate(val_pbar):
                    input_ids      = data["input_ids"].to(device)       # [B, N, L]
                    attention_mask = data["attention_mask"].to(device)   # [B, N, L]
                    agent_valid    = data["agent_valid"].to(device)      # [B, N]

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        predicted_trajs = model(input_ids, attention_mask, agent_valid)
                        loss_out = loss_fn(predicted_trajs, data)

                    val_loss_meter.update({k: v.item() for k, v in loss_out.items()})

                    # --- [Exp1] Absolute coordinate metrics ---
                    # pred_focal = predicted_trajs[:, 0].float()               # [B, 60, 2]
                    # gt_focal   = data["gt_trajectories"][:, 0].to(device)    # [B, 60, 2]
                    # mask_focal = data["gt_masks"][:, 0].to(device)           # [B, 60]
                    # B_val      = pred_focal.shape[0]
                    # l2_dist    = (pred_focal - gt_focal).norm(dim=-1)        # [B, 60]

                    # --- [Exp2] Displacement metrics: cumsum predicted displacements → absolute, then compute L2 ---
                    pred_disp_focal = predicted_trajs[:, 0].float()                         # [B, 60, 2] predicted displacements
                    anchor          = data["gt_anchor"][:, 0].to(device).float()            # [B, 2] last valid obs position
                    pred_abs_focal  = anchor.unsqueeze(1) + pred_disp_focal.cumsum(dim=1)   # [B, 60, 2] reconstructed absolute
                    gt_focal        = data["gt_abs_trajectories"][:, 0].to(device).float()  # [B, 60, 2] absolute GT
                    mask_focal      = data["gt_masks"][:, 0].to(device)                     # [B, 60]
                    B_val           = pred_abs_focal.shape[0]
                    l2_dist         = (pred_abs_focal - gt_focal).norm(dim=-1)              # [B, 60]

                    valid_per_sample = mask_focal.float().sum(dim=1).clamp(min=1)
                    min_ade = ((l2_dist * mask_focal.float()).sum(dim=1) / valid_per_sample).mean()

                    last_valid_idx = (mask_focal.long().cumsum(dim=1) * mask_focal.long()).argmax(dim=1)
                    min_fde = l2_dist[torch.arange(B_val, device=device), last_valid_idx].mean()

                    ade_meter.update({"minADE": min_ade.item(), "minFDE": min_fde.item()})

                    val_pbar.set_postfix({
                        "loss": f"{val_loss_meter.metrics['loss'].avg:.4f}",
                        "minADE": f"{ade_meter.metrics['minADE'].avg:.4f}",
                        "minFDE": f"{ade_meter.metrics['minFDE'].avg:.4f}",
                    })

            val_pbar.close()

            val_loss_avg = val_loss_meter.metrics['loss'].avg
            min_ade_avg  = ade_meter.metrics['minADE'].avg
            min_fde_avg  = ade_meter.metrics['minFDE'].avg
            logger.print(
                f'[Validation] ep {epoch + 1} | loss: {val_loss_avg:.4f} | '
                f'minADE: {min_ade_avg:.4f} m | minFDE: {min_fde_avg:.4f} m | '
                f'time: {(time.time() - val_start) / 60.0:.3f} mins'
            )
            logger.add_scalar('val/loss',   val_loss_avg, it=epoch)
            logger.add_scalar('val/minADE', min_ade_avg,  it=epoch)
            logger.add_scalar('val/minFDE', min_fde_avg,  it=epoch)

            if val_loss_avg < best_val_loss:
                improvement = best_val_loss - val_loss_avg
                best_val_loss = val_loss_avg
                logger.print(
                    f'  >> Validation improved by {improvement:.4f}; saving best checkpoint...'
                )
                ok = safe_save(logger, model, optimizer, epoch, args.ckpt_dir, best_ckpt_name)
                saved_any = saved_any or ok
            else:
                logger.print(
                    f'  -- No improvement over best ({best_val_loss:.4f}); '
                    f'current val loss = {val_loss_avg:.4f}'
                )

    # ------ Final summary: prove (or warn) that something landed on disk ------
    elapsed_min = (time.time() - train_start) / 60.0
    logger.print(f"\nTraining completed in {elapsed_min:.2f} mins | best val loss = {best_val_loss:.4f}")

    saved_files = sorted(glob.glob(os.path.join(args.ckpt_dir, f'{date_str}_llm_simpl_*.tar')))
    if not saved_files:
        logger.print(
            "!! WARNING: No checkpoint from this run was found on disk.\n"
            f"   Looked in: {os.path.abspath(args.ckpt_dir)}\n"
            f"   Pattern  : {date_str}_llm_simpl_*.tar\n"
            "   Check the [Validation] / save_ckpt log lines above for errors."
        )
    else:
        logger.print("Checkpoints written this run:")
        for p in saved_files:
            size_mb = os.path.getsize(p) / (1024 ** 2)
            logger.print(f"  - {p} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()