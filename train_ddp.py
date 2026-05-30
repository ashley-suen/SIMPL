"""
Multi-GPU training script (DDP).  Structure mirrors train_llm.py exactly.
Launch: torchrun --nproc_per_node=<N> train_ddp.py --features_dir data_av2/features ...

DDP-only changes vs train_llm.py:
  1. init_process_group / destroy_process_group
  2. device = cuda:{local_rank}
  3. Model wrapped with DDP; safe_save uses model.module
  4. DataLoader uses DistributedSampler instead of shuffle=True
  5. train_sampler.set_epoch(epoch) each epoch
  6. Logging / saving / sanity checks only on rank 0
  7. Validation metrics all_reduced across ranks before logging
  8. dist.barrier() at epoch start and before/after validation to keep ranks in sync
"""
import os
os.environ.setdefault("USE_LIBUV", "0")   # Windows: TCPStore built without libuv support
import sys
import glob
import time
import argparse
import datetime
from datetime import datetime as dt
from collections import deque
from tqdm import tqdm

import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW


# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data_av2.av2_llm_dataset import AV2PromptDataset
from simpl.llm_motion_model import SmolLMMotionPredictor
from simpl.av2_llm_loss import LLMMotionLoss
from utils.logger import Logger
from utils.utils import AverageMeterForDict, set_seed, save_ckpt, distributed_mean


def dynamic_pad_collate(batch):
    """Trim unified input_ids / attention_mask to the longest real token in this batch.
    input_ids is now [L] per sample (unified scene prompt), not [N, L].
    agent_positions are clamped to the trimmed length to stay in bounds.
    """
    max_len = int(max(item["attention_mask"].sum().item() for item in batch))
    for item in batch:
        item["input_ids"]       = item["input_ids"][:max_len]
        item["attention_mask"]  = item["attention_mask"][:max_len]
        item["agent_positions"] = item["agent_positions"].clamp(0, max_len - 1)
    return torch.utils.data.dataloader.default_collate(batch)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Per-GPU training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Per-GPU val batch size")
    parser.add_argument("--train_epoches", type=int, default=10, help="Number of epoches for training")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation intervals")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--logger_writer", action="store_true", help="Enable tensorboard")
    parser.add_argument("--no_pbar", action="store_true", help="Hide progress bar")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="If >0, randomly subsample training set to this many samples "
                             "(fixed by --seed for reproducibility). 0 = use full dataset.")

    # LLM Specific Args
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B-Base",
                        help="HuggingFace model ID for the LLM backbone and tokenizer.")
    parser.add_argument("--llm_lr", type=float, default=1e-5, help="Learning rate for LoRA parameters")
    parser.add_argument("--gru_lr", type=float, default=1e-4, help="Learning rate for GRU decoder")
    parser.add_argument("--lora_r",       type=int,   default=16,
                        help="LoRA rank (default=16)")
    parser.add_argument("--lora_alpha",   type=int,   default=32,
                        help="LoRA scaling factor (default=32, i.e. 2×r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout (default=0.05)")
    parser.add_argument("--lora_targets", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of linear module names to apply LoRA to")

    # Loss
    parser.add_argument("--soft_wta_alpha", type=float, default=0.1,
                        help="Soft WTA weight for non-winner modes (0.0 = hard WTA, not recommended).")
    parser.add_argument("--endpoint_weight", type=float, default=0.2,
                        help="Weight for endpoint-consistency auxiliary loss. "
                             "Penalises predicting stationary motion for moving agents. "
                             "0.0 disables it. Recommended: 0.1-0.3.")

    # LR Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine_warmup_restart",
                        choices=["cosine", "cosine_restart", "cosine_warmup_restart"],
                        help="LR scheduler: cosine | cosine_restart | cosine_warmup_restart "
                             "(warmup at start of each cycle + non-zero floor)")
    parser.add_argument("--T_0", type=int, default=20,
                        help="Epochs per restart cycle")
    parser.add_argument("--T_mult", type=int, default=1,
                        help="[cosine_restart] Cycle length multiplier after each restart")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="[cosine_warmup_restart] Linear warmup epochs at start of each cycle")
    parser.add_argument("--eta_min_ratio", type=float, default=0.1,
                        help="[cosine_warmup_restart] LR floor as fraction of max LR (e.g. 0.1 = 10%%)")

    # Logging frequency control
    parser.add_argument("--print_interval", type=int, default=100,
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

    # Resume
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to checkpoint (.tar) to resume from")
    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="Epoch index (1-based display) already completed in the checkpoint. "
                             "Used to fast-forward the LR scheduler.")
    parser.add_argument("--reset_optimizer", action="store_true",
                        help="When resuming, load model weights only — skip optimizer state. "
                             "Required when changing unfreeze_layers or other param-group structure.")
    parser.add_argument("--num_modes", type=int, default=6,
                        help="Number of prediction modes K (default=6). "
                             "K=1 is single-modal (original behaviour). "
                             "K>1 uses Winner-Takes-All training loss; "
                             "minADE/minFDE take the best mode per sample.")

    # Gradient clipping
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm for clip_grad_norm_")

    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="Stop after this many epochs with no minADE improvement. 0=disabled.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001,
                        help="Minimum minADE improvement (m) to count as progress for early stopping.")

    # Sequence length
    parser.add_argument("--max_seq_len", type=int, default=8192,
                        help="Max token length for the unified scene prompt. "
                             "Worst case (6 agents × 50 steps) ~5050 tokens; 8192 is safe. "
                             "Qwen3-0.6B supports 40k context.")

    # torch.compile
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model with torch.compile after DDP for extra speed. "
                             "First iteration will be slow (compilation).")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune",
                                 "max-autotune-no-cudagraphs"],
                        help="torch.compile mode. "
                             "'max-autotune-no-cudagraphs' (default): full kernel autotuning "
                             "without CUDA Graphs — required when using Flash Attention 2, "
                             "which uses torch.nonzero (data-dependent output) incompatible with CUDA Graphs. "
                             "'max-autotune': adds CUDA Graphs on top, crashes with Flash Attention. "
                             "'reduce-overhead': moderate speedup, still uses CUDA Graphs. "
                             "'default': no CUDA Graphs, safe but least optimized.")

    # Flash Attention
    parser.add_argument("--flash_attn", action="store_true",
                        help="Enable Flash Attention 2 in the LLM backbone. "
                             "Requires flash-attn package (pip install flash-attn --no-build-isolation). "
                             "Recommended for Hopper/Ampere GPUs (H800, A100). "
                             "Do NOT use on Blackwell (RTX 5090/sm_120) — conflicts with NCCL.")

    # DDP backend
    parser.add_argument("--dist_backend", type=str, default="gloo",
                        choices=["gloo", "nccl"],
                        help="DDP backend. Use 'nccl' on Linux/Hopper/Ampere GPUs for best performance. "
                             "Default 'gloo' for Windows/Blackwell (sm_120) compatibility.")

    return parser.parse_args()


def format_lr(optimizer):
    """Format current LRs of each param group as a readable string."""
    parts = []
    names = ["llm", "gru"]
    for i, pg in enumerate(optimizer.param_groups):
        name = names[i] if i < len(names) else f"g{i}"
        parts.append(f"{name}_lr={pg['lr']:.2e}")
    return ", ".join(parts)


def detect_gpu_config():
    """
    Detect GPU architecture and return compatibility info.

    SM version map (relevant):
      sm80  → Ampere  (A100, A30)                — flash attn ✅
      sm86  → Ampere  (RTX 3090, A40)            — flash attn ✅
      sm89  → Ada     (RTX 4090, RTX Pro 6000 Ada) — flash attn ✅
      sm90  → Hopper  (H100, H800)               — flash attn ✅
      sm120 → Blackwell (RTX 5090, RTX Pro 6000 Blackwell) — flash attn ❌

    Returns: (sm, gpu_name, flash_attn_ok)
    """
    if not torch.cuda.is_available():
        return 0, "no GPU", False
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    name = torch.cuda.get_device_name()
    flash_ok = sm >= 80   # Ampere and above; let flash-attn itself report if unsupported
    return sm, name, flash_ok


def count_trainable_params(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def safe_save(logger, model, optimizer, epoch, dirpath, name):
    """Wrap save_ckpt so a failure can never silently swallow our checkpoint."""
    try:
        # DDP wraps the model; unwrap to get the original state_dict
        raw_model = model.module if isinstance(model, DDP) else model
        save_ckpt(raw_model, optimizer, epoch, dirpath, name)
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # ------ DDP initialisation ------
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ.get("RANK", "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    
    # Set device BEFORE init_process_group to avoid NCCL warnings
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    is_main = (global_rank == 0)
    # --------------------------------

    args = parse_arguments()
    set_seed(args.seed + global_rank)   # unique seed per rank for data diversity

    date_str = dt.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str + "_llm_ddp"

    # Logger: pass enable=is_main so non-main ranks produce no output/files
    logger = Logger(date_str=date_str, enable=is_main, log_dir=log_dir,
                    enable_flags={'writer': args.logger_writer})

    # Detect GPU before init_process_group so we can auto-switch backend if needed
    sm, gpu_name, flash_ok = detect_gpu_config()

    # Blackwell (sm120+) cannot use NCCL — force gloo regardless of user flag
    if sm >= 100 and args.dist_backend == "nccl":
        print(f"[Rank {global_rank}] Blackwell sm{sm} ({gpu_name}) detected — "
              f"NCCL causes illegal memory access on sm120; forcing dist_backend: nccl → gloo")
        args.dist_backend = "gloo"

    if sys.platform != "win32":
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"   # Linux only — not valid on Windows
    os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
    os.environ["GLOO_SOCKET_NTHREADS"] = "8"
    dist.init_process_group(backend=args.dist_backend, timeout=datetime.timedelta(minutes=30),
                            device_id=device)
    logger.print(f"DDP backend={args.dist_backend} | world_size={world_size} | rank={global_rank}")

    if is_main:
        logger.log_basics(args=args, datetime=date_str)
        logger.print(f"DDP world_size={world_size} | "
                     f"per-GPU batch={args.train_batch_size} | "
                     f"total effective batch={args.train_batch_size * world_size}")

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
    train_set = AV2PromptDataset(train_dir, tokenizer_name=args.model_name,
                                 max_len_per_agent=args.max_seq_len)
    val_set   = AV2PromptDataset(val_dir,   tokenizer_name=args.model_name,
                                 max_len_per_agent=args.max_seq_len)
    full_train_set  = train_set   # keep reference for per-epoch resampling
    full_train_size = len(train_set)
    use_subset      = args.max_train_samples > 0 and full_train_size > args.max_train_samples

    if use_subset:
        logger.print(f"Train samples: {args.max_train_samples} / {full_train_size} "
                     f"(re-sampled each epoch) | Val samples: {len(val_set)} "
                     f"| max_seq_len={args.max_seq_len}")
    else:
        logger.print(f"Train samples: {full_train_size} | Val samples: {len(val_set)} "
                     f"| max_seq_len={args.max_seq_len}")

    if is_main and len(val_set) == 0:
        raise RuntimeError(
            f"Validation set is empty (path={val_dir}). "
            f"Best-checkpoint logic cannot work — aborting."
        )

    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=global_rank, shuffle=False)
    persistent  = args.num_workers > 0
    dl_val = DataLoader(val_set, batch_size=args.val_batch_size, sampler=val_sampler,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=persistent, collate_fn=dynamic_pad_collate)

    # dl_train is built per-epoch when use_subset=True; build once here for the no-subset case
    if not use_subset:
        train_sampler = DistributedSampler(full_train_set, num_replicas=world_size,
                                           rank=global_rank, shuffle=True)
        dl_train = DataLoader(full_train_set, batch_size=args.train_batch_size,
                              sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=persistent,
                              collate_fn=dynamic_pad_collate)
    else:
        train_sampler = None   # will be created per epoch
        dl_train      = None

    val_iters = len(dl_val)
    iters_per_epoch = (args.max_train_samples // (args.train_batch_size * world_size)
                       if use_subset else len(dl_train))
    logger.print(f"Iterations per epoch (per GPU): train≈{iters_per_epoch}, val={val_iters}")

    logger.print(f"GPU: {gpu_name}  |  SM={sm}  |  flash_attn_compatible={flash_ok}")

    use_flash_attn = args.flash_attn
    if use_flash_attn and not flash_ok:
        logger.print(
            f"WARNING: GPU sm{sm} ({gpu_name}) may not support flash_attn — "
            f"proceeding anyway, will crash at runtime if unsupported."
        )
    if use_flash_attn:
        logger.print(f"Flash Attention 2: Enabled (sm{sm})")
    else:
        logger.print(f"Flash Attention 2: Disabled (using SDPA)")
    
    # DDP workaround: use file-based synchronization to avoid concurrent model loading.
    import time
    lock_file = "/tmp/simpl_ddp_model_load.lock"
    
    if global_rank == 0:
        # Rank 0: create lock, load model, then remove lock
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        
        model = SmolLMMotionPredictor(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_targets.split(","),
            lora_dropout=args.lora_dropout,
            num_modes=args.num_modes,
            device=None,
            use_flash_attn=use_flash_attn,
            dtype=torch.bfloat16,
        )
        model = model.to(device).to(torch.bfloat16)
        if os.path.exists(lock_file):
            os.remove(lock_file)
    else:
        max_wait = 300
        waited = 0
        while os.path.exists(lock_file) and waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
        if waited >= max_wait:
            raise RuntimeError(f"Rank {global_rank} timed out waiting for rank 0 to load model")
        time.sleep(1)
        model = SmolLMMotionPredictor(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_targets.split(","),
            lora_dropout=args.lora_dropout,
            num_modes=args.num_modes,
            device=None,
            use_flash_attn=use_flash_attn,
            dtype=torch.bfloat16,
        )
        model = model.to(device).to(torch.bfloat16)
    
    dist.barrier()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, dynamic=False)
        logger.print(f"torch.compile enabled (mode={args.compile_mode}, dynamic=True)")

    loss_fn = LLMMotionLoss(device=device, soft_wta_alpha=args.soft_wta_alpha,
                            endpoint_weight=args.endpoint_weight)
    if is_main:
        logger.print(f"Loss: disp-space SmoothL1 + endpoint consistency (w={args.endpoint_weight}) | "
                     f"Soft WTA alpha={args.soft_wta_alpha} (winner=1.0, others={args.soft_wta_alpha})")

        n_trainable, n_total = count_trainable_params(model)
        logger.print(f"Model params: trainable={n_trainable:,} / total={n_total:,} "
                     f"({100.0 * n_trainable / max(n_total, 1):.2f}% trainable)")

    # 3. Optimizer (Differential Learning Rates)
    # DDP prefixes names with "module." — use substring match instead of startswith
    llm_params = [p for n, p in model.named_parameters() if p.requires_grad and "llm." in n]
    gru_params = [p for n, p in model.named_parameters() if p.requires_grad and "llm." not in n]

    if is_main:
        logger.print(f"Trainable groups: llm={sum(p.numel() for p in llm_params):,} params "
                     f"| gru_decoder={sum(p.numel() for p in gru_params):,} params")

    optimizer = AdamW([
        {'params': llm_params, 'lr': args.llm_lr},
        {'params': gru_params, 'lr': args.gru_lr},
    ], weight_decay=1e-4)

    # --- LR Scheduler ---
    if args.scheduler == "cosine_warmup_restart":
        # Cosine decay with linear warmup at the start of each cycle and a non-zero LR floor.
        # Within each T_0-epoch cycle:
        #   [0, warmup_epochs)      : linear ramp from eta_min_ratio → 1.0
        #   [warmup_epochs, T_0)    : cosine decay from 1.0 → eta_min_ratio
        # This avoids the near-zero LR trough before each restart.
        _T0   = args.T_0
        _wu   = args.warmup_epochs
        _r    = args.eta_min_ratio
        def _lr_lambda(epoch):
            cycle_pos = epoch % _T0
            if cycle_pos < _wu:
                return _r + (1.0 - _r) * cycle_pos / max(1, _wu)
            progress = (cycle_pos - _wu) / max(1, _T0 - _wu)
            return _r + (1.0 - _r) * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        if is_main:
            logger.print(f"Scheduler: CosineWarmupRestart | T_0={_T0}, "
                         f"warmup={_wu} epochs, eta_min_ratio={_r} "
                         f"(floor={args.llm_lr * _r:.2e} for llm, "
                         f"{args.gru_lr * _r:.2e} for gru)")
    elif args.scheduler == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=args.T_mult
        )
        if is_main:
            logger.print(f"Scheduler: CosineAnnealingWarmRestarts | T_0={args.T_0}, T_mult={args.T_mult}")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoches)
        if is_main:
            logger.print(f"Scheduler: CosineAnnealingLR | T_max={args.train_epoches}")

    # 4. Resume from checkpoint (optional)
    start_epoch = 0
    if args.resume_from:
        map_loc = {"cuda:0": f"cuda:{local_rank}"}
        ckpt = torch.load(args.resume_from, map_location=map_loc)
        raw_model = model.module if isinstance(model, DDP) else model
        missing, unexpected = raw_model.load_state_dict(ckpt["state_dict"], strict=False)
        if is_main and (missing or unexpected):
            logger.print(f"  [Resume] strict=False: {len(missing)} missing keys, "
                         f"{len(unexpected)} unexpected keys (expected when changing num_modes/architecture)")
        if args.reset_optimizer:
            if is_main:
                logger.print("--reset_optimizer: skipping optimizer state load (fresh Adam states)")
        else:
            optimizer.load_state_dict(ckpt["opt_state"])
            # Move optimizer states to the correct device (they are saved on CPU)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        start_epoch = args.resume_epoch          # epochs already completed (1-based count)
        # Fast-forward the scheduler to match the resumed epoch
        for _ in range(start_epoch):
            scheduler.step()
        if is_main:
            logger.print(f"Resumed from: {args.resume_from}")
            logger.print(f"Continuing from epoch {start_epoch + 1} | "
                         f"LR after fast-forward: {format_lr(optimizer)}")

    # 5. Training Loop
    niter = start_epoch * args.train_batch_size * world_size * iters_per_epoch
    best_val_loss = float('inf')
    best_minADE   = float('inf')
    patience_counter = 0
    # Shared flag for early-stop broadcast: rank 0 sets it, all ranks read it
    should_stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
    train_start = time.time()
    last_ckpt_name = f'{date_str}_llm_simpl_last.tar'
    best_ckpt_name = f'{date_str}_llm_simpl_best.tar'
    saved_any = False

    if is_main and args.early_stop_patience > 0:
        logger.print(f"Early stopping: patience={args.early_stop_patience} epochs, "
                     f"min_delta={args.early_stop_min_delta} m (monitored metric: minADE)")

    for epoch in range(start_epoch, args.train_epoches):
        # FIX: sync all ranks at the start of each epoch before any work begins
        dist.barrier()

        # Re-sample a fresh subset each epoch so the model sees different samples every time
        if use_subset:
            ep_indices = torch.randperm(
                full_train_size,
                generator=torch.Generator().manual_seed(args.seed + epoch)
            )[:args.max_train_samples].tolist()
            ep_train_set  = torch.utils.data.Subset(full_train_set, ep_indices)
            train_sampler = DistributedSampler(ep_train_set, num_replicas=world_size,
                                               rank=global_rank, shuffle=True)
            dl_train = DataLoader(ep_train_set, batch_size=args.train_batch_size,
                                  sampler=train_sampler, num_workers=args.num_workers,
                                  pin_memory=True, persistent_workers=False,
                                  collate_fn=dynamic_pad_collate)
            iters_per_epoch = len(dl_train)

        train_sampler.set_epoch(epoch)  # DDP: ensures different shuffle each epoch

        if is_main:
            logger.print('\n' + '=' * 80)
            logger.print(f'Epoch {epoch + 1}/{args.train_epoches}')
            logger.print(f'  - LR : {format_lr(optimizer)}')
            logger.print(f'  - Batch size (per-GPU/total): {args.train_batch_size}/{args.train_batch_size * world_size}')
            logger.print(f'  - GPUs                      : {world_size}')
            logger.print(f'  - LoRA rank / alpha         : {args.lora_r} / {args.lora_alpha}')
            logger.print(f'  - LoRA targets              : {args.lora_targets}')
            logger.print(f'  - Prediction modes (K)      : {args.num_modes}')
            logger.print(f'  - AMP (bf16)                : True (weights + autocast)')
            logger.print(f'  - Grad clip max norm        : {args.grad_clip}')
            logger.print(f'  - Max seq len               : {args.max_seq_len}')
            logger.print(f'  - torch.compile             : {args.compile} (mode={args.compile_mode})')
            logger.print('=' * 80)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # --- TRAIN ---
        epoch_start = time.time()
        train_loss_meter = AverageMeterForDict()
        loss_window = deque(maxlen=args.running_window)

        model.train()

        pbar = tqdm(dl_train, disable=(not is_main or args.no_pbar), ncols=110,
                    desc=f"[Train ep {epoch + 1}]")

        for i, data in enumerate(pbar):
            input_ids       = data["input_ids"].to(device)        # [B, L]
            attention_mask  = data["attention_mask"].to(device)    # [B, L]
            agent_positions = data["agent_positions"].to(device)   # [B, N]
            agent_valid     = data["agent_valid"].to(device)       # [B, N]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predicted_trajs = model(input_ids, attention_mask, agent_positions, agent_valid)
                loss_out = loss_fn(predicted_trajs, data)

            loss = loss_out["loss"]

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            scalar_losses = {k: v.item() for k, v in loss_out.items()}
            train_loss_meter.update(scalar_losses)
            loss_window.append(scalar_losses["loss"])

            niter += args.train_batch_size * world_size
            if is_main:
                logger.add_dict(scalar_losses, niter, prefix='train/')

            running_loss = sum(loss_window) / len(loss_window)
            gnorm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            postfix = {
                "loss": f"{scalar_losses['loss']:.4f}",
                f"loss_avg{len(loss_window)}": f"{running_loss:.4f}",
                "gnorm": f"{gnorm_val:.2f}",
            }
            pbar.set_postfix(postfix)

            if is_main and ((i + 1) % args.print_interval == 0 or (i + 1) == iters_per_epoch):
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
        if is_main:
            logger.print(
                f'[Training] Epoch {epoch + 1} done | avg loss: {loss_avg:.6f} | '
                f'time: {(time.time() - epoch_start) / 60.0:.3f} mins | '
                f'max mem: {max_memory} MB | next-epoch {format_lr(optimizer)}'
            )

        # ------ Always save a 'last' checkpoint, regardless of val ------
        if is_main and args.save_last_every_epoch:
            ok = safe_save(logger, model, optimizer, epoch, args.ckpt_dir, last_ckpt_name)
            saved_any = saved_any or ok

        # FIX: sync before validation so all ranks enter together
        dist.barrier()

        # --- VALIDATION ---
        if (epoch + 1) % args.val_interval == 0:
            val_start = time.time()
            val_loss_meter = AverageMeterForDict()
            ade_meter = AverageMeterForDict()
            model.eval()

            # FIX: set_epoch on val_sampler too, consistent with original
            val_sampler.set_epoch(epoch)

            val_pbar = tqdm(dl_val, disable=(not is_main or args.no_pbar), ncols=110,
                            desc=f"[Val   ep {epoch + 1}]")

            with torch.no_grad():
                for i, data in enumerate(val_pbar):
                    input_ids       = data["input_ids"].to(device)        # [B, L]
                    attention_mask  = data["attention_mask"].to(device)    # [B, L]
                    agent_positions = data["agent_positions"].to(device)   # [B, N]
                    agent_valid     = data["agent_valid"].to(device)       # [B, N]

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        predicted_trajs = model(input_ids, attention_mask, agent_positions, agent_valid)
                        loss_out = loss_fn(predicted_trajs, data)

                    val_loss_meter.update({k: v.item() for k, v in loss_out.items()})

                    # minADE/minFDE: best mode over K for focal agent (index 0)
                    # predicted_trajs: [B, N, K, T, 2]
                    pred_disp_focal = predicted_trajs[:, 0].float()           # [B, K, T, 2]
                    K_val           = pred_disp_focal.shape[1]
                    anchor          = data["gt_anchor"][:, 0].to(device).float()             # [B, 2]
                    # [B, K, T, 2]: cumsum per mode
                    pred_abs_focal  = anchor.unsqueeze(1).unsqueeze(1) + \
                                      pred_disp_focal.cumsum(dim=2)           # [B, K, T, 2]
                    gt_focal        = data["gt_abs_trajectories"][:, 0].to(device).float()  # [B, T, 2]
                    mask_focal      = data["gt_masks"][:, 0].to(device)                     # [B, T]
                    B_val           = pred_abs_focal.shape[0]

                    # L2 per mode per step: [B, K, T]
                    l2_dist = (pred_abs_focal - gt_focal.unsqueeze(1)).norm(dim=-1)

                    # minADE: min over K of mean-over-valid-steps L2
                    valid_per_sample  = mask_focal.float().sum(dim=1).clamp(min=1)          # [B]
                    per_mode_ade      = (l2_dist * mask_focal.unsqueeze(1).float()).sum(dim=2) \
                                        / valid_per_sample.unsqueeze(1)                     # [B, K]
                    min_ade           = per_mode_ade.min(dim=1).values.mean()

                    # minFDE: min over K of L2 at last valid timestep
                    last_valid_idx    = (mask_focal.long().cumsum(dim=1) *
                                         mask_focal.long()).argmax(dim=1)                   # [B]
                    idx_fde           = last_valid_idx.unsqueeze(1).unsqueeze(2) \
                                        .expand(-1, K_val, 1)                               # [B, K, 1]
                    per_mode_fde      = l2_dist.gather(2, idx_fde).squeeze(2)              # [B, K]
                    min_fde           = per_mode_fde.min(dim=1).values.mean()

                    ade_meter.update({"minADE": min_ade.item(), "minFDE": min_fde.item()})

                    if is_main:
                        val_pbar.set_postfix({
                            "loss": f"{val_loss_meter.metrics['loss'].avg:.4f}",
                            "minADE": f"{ade_meter.metrics['minADE'].avg:.4f}",
                            "minFDE": f"{ade_meter.metrics['minFDE'].avg:.4f}",
                        })

            val_pbar.close()

            # FIX: use distributed_mean (all_gather + mean) to aggregate metrics across ranks,
            # matching the original's approach instead of the broken all_reduce+manual-divide.
            metric_keys = ["loss", "minADE", "minFDE"]
            local_vals = []
            for key in metric_keys:
                meter = val_loss_meter if key == "loss" else ade_meter
                local_vals.append(meter.metrics[key].avg if key in meter.metrics else 0.0)

            local_tensor = torch.tensor(local_vals, dtype=torch.float64, device=device)
            global_mean  = distributed_mean(local_tensor)  # all_gather + mean across ranks

            val_loss_avg = global_mean[0].item()
            min_ade_avg  = global_mean[1].item()
            min_fde_avg  = global_mean[2].item()

            if is_main:
                logger.print(
                    f'[Validation] ep {epoch + 1} | loss: {val_loss_avg:.4f} | '
                    f'minADE: {min_ade_avg:.4f} m | minFDE: {min_fde_avg:.4f} m | '
                    f'time: {(time.time() - val_start) / 60.0:.3f} mins'
                )
                logger.add_scalar('val/loss',   val_loss_avg, it=epoch)
                logger.add_scalar('val/minADE', min_ade_avg,  it=epoch)
                logger.add_scalar('val/minFDE', min_fde_avg,  it=epoch)

                # Best checkpoint criterion: minADE (lower is better)
                ade_improved = min_ade_avg < best_minADE - args.early_stop_min_delta
                if ade_improved:
                    improvement = best_minADE - min_ade_avg
                    best_minADE   = min_ade_avg
                    best_val_loss = val_loss_avg   # keep for final summary
                    patience_counter = 0
                    logger.print(
                        f'  >> minADE improved by {improvement:.4f} m → {best_minADE:.4f} m; '
                        f'saving best checkpoint...'
                    )
                    ok = safe_save(logger, model, optimizer, epoch, args.ckpt_dir, best_ckpt_name)
                    saved_any = saved_any or ok
                else:
                    patience_counter += 1
                    remaining = (args.early_stop_patience - patience_counter
                                 if args.early_stop_patience > 0 else -1)
                    patience_str = (f' | patience {patience_counter}/{args.early_stop_patience}'
                                    if args.early_stop_patience > 0 else '')
                    logger.print(
                        f'  -- No minADE improvement (best={best_minADE:.4f} m, '
                        f'current={min_ade_avg:.4f} m){patience_str}'
                    )
                    if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                        logger.print(
                            f'  !! Early stopping triggered: no improvement for '
                            f'{patience_counter} consecutive epochs.'
                        )
                        should_stop_flag[0] = 1

            # Broadcast early-stop signal from rank 0 to all ranks
            dist.broadcast(should_stop_flag, src=0)

            # FIX: sync after validation before next epoch starts
            dist.barrier()

            if should_stop_flag.item() == 1:
                if is_main:
                    logger.print(f"All ranks received early-stop signal — exiting training loop.")
                break

    # ------ Final summary ------
    if is_main:
        elapsed_min = (time.time() - train_start) / 60.0
        logger.print(
            f"\nTraining completed in {elapsed_min:.2f} mins | "
            f"best minADE = {best_minADE:.4f} m | best val loss = {best_val_loss:.4f}"
        )

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

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
