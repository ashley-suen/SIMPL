"""
Multi-GPU DDP training script for the Hybrid LLM Motion Predictor.

Architecture: Hybrid Token (semantic text + numerical embedding) +
              Level-k Interaction Decoding (GameFormer-style).

Launch:
    torchrun --nproc_per_node=4 train_hybrid.py --features_dir data_av2/features [options]
"""
import os
os.environ.setdefault("USE_LIBUV", "0")
import sys
import glob
import time
import argparse
import contextlib
import datetime
import math
from datetime import datetime as dt
from collections import deque
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data_av2.av2_hybrid_dataset import AV2HybridDataset
from simpl.hybrid_llm_model import HybridLLMPredictor
from simpl.hybrid_loss import HybridMotionLoss
from utils.logger import Logger
from utils.utils import AverageMeterForDict, set_seed, save_ckpt, distributed_mean


# ── Collate ───────────────────────────────────────────────────────────────────

def hybrid_collate_fn(batch):
    """
    Pads text_input_ids / text_attention_mask to the longest text in the batch.
    All other fields are fixed-size and handled by default_collate.
    """
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

    # Paths
    parser.add_argument("--features_dir", required=True, type=str)
    parser.add_argument("--ckpt_dir",     type=str, default="saved_models/")
    parser.add_argument("--exp_name",     type=str, default="hybrid_ddp",
                        help="Experiment name; logs go to log/{exp_name}/")

    # Dataset
    parser.add_argument("--max_text_len", type=int, default=500,
                        help="Max tokens for the semantic text portion (no coord steps)")
    parser.add_argument("--max_agents",   type=int, default=6)
    parser.add_argument("--max_lanes",    type=int, default=20)
    parser.add_argument("--max_text_agents", type=int, default=6,
                        help="# surrounding agents enumerated IN TEXT (top-influence). "
                             "Decoupled from max_agents (numerical channel) to keep the "
                             "LLM sequence short; the rest are handled numerically.")

    # Training
    parser.add_argument("--train_batch_size",   type=int, default=4)
    parser.add_argument("--val_batch_size",     type=int, default=8)
    parser.add_argument("--train_epoches",      type=int, default=64)
    parser.add_argument("--val_interval",       type=int, default=1)
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--num_workers",        type=int, default=4)
    parser.add_argument("--max_train_samples",  type=int, default=0)
    parser.add_argument("--logger_writer",      action="store_true")
    parser.add_argument("--no_pbar",            action="store_true")

    # Model
    parser.add_argument("--model_name",  type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--num_modes",     type=int, default=6)
    parser.add_argument("--rgsf_layers", type=int, default=0,
                        help="Relative-Geometric Scene Fusion layers (exp15). "
                             "0 = disabled (pre-exp15 behaviour).")
    parser.add_argument("--rgsf_dim",    type=int, default=256,
                        help="RGSF internal (down-projected) fusion dimension.")
    parser.add_argument("--n_bezier_ctrl", type=int, default=6,
                        help="Number of Bézier control points per mode (degree = n-1). "
                             "6 control points (degree 5) covers straight/lane-change/turn.")
    parser.add_argument("--n_levels",      type=int, default=2,
                        help="Number of Level-k interaction refinement rounds")

    # LoRA
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str, default="all-linear",
                        help="LoRA target modules. 'all-linear' (default) targets every "
                             "nn.Linear in the LLM automatically. Pass a comma-separated "
                             "list (e.g. 'q_proj,v_proj') for selective targeting.")

    # Loss
    parser.add_argument("--soft_wta_alpha",    type=float, default=0.0,
                        help="Weight for non-winner modes in soft WTA. 0.0 (default) "
                             "= pure scene-level WTA (exp12): alpha>0 pulls all modes "
                             "toward the conditional mean, fighting mode diversity")
    parser.add_argument("--cls_weight",        type=float, default=0.5,
                        help="Weight for scene-level mode-classification loss "
                             "(supervises per-mode confidence; needed for brierMinFDE)")

    # Ablation
    parser.add_argument("--llm_correction_scale", type=float, default=1.0,
                        help="Scale applied to the LLM correction before adding to "
                             "agent_emb (Route A ablation). 1.0 = normal. 0.0 = "
                             "encoder-only (LLM correction zeroed out; LLM LoRA grad "
                             "is also killed since d_loss/d_h_correction=0). Use 0.0 "
                             "to test whether LLM dominance is capping minADE.")
    parser.add_argument("--scene_guidance_scale", type=float, default=1.0,
                        help="Scale applied to the LLM [SCN] scenario-query delta in "
                             "the decoder (exp12). 1.0 = normal. 0.0 = exp12-A "
                             "ablation: scenario queries are pure learned embeddings, "
                             "isolating the decoder-restructure gain from the LLM "
                             "scene-guidance gain.")

    # Optimiser
    parser.add_argument("--llm_lr",    type=float, default=5e-5)
    parser.add_argument("--gru_lr",    type=float, default=1e-4)
    parser.add_argument("--grad_clip",     type=float, default=5.0,
                        help="Gradient clip for encoder+decoder params (GameFormer uses 5.0)")
    parser.add_argument("--llm_grad_clip", type=float, default=2.5,
                        help="Gradient clip for LLM LoRA params (separate from encoder). "
                             "exp11 showed pre-clip norms of ~10 at 1.0 — the LLM branch "
                             "was being truncated to 10% of its desired step")
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    # Scheduler
    parser.add_argument("--scheduler",      type=str, default="cosine_warmup_restart",
                        choices=["cosine", "cosine_restart", "cosine_warmup_restart"])
    parser.add_argument("--T_0",            type=int,   default=20)
    parser.add_argument("--T_mult",         type=int,   default=1)
    parser.add_argument("--warmup_epochs",  type=int,   default=2)
    parser.add_argument("--eta_min_ratio",  type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--print_interval",       type=int, default=100)
    parser.add_argument("--running_window",        type=int, default=50)
    parser.add_argument("--save_last_every_epoch", action="store_true", default=True)

    # Resume
    parser.add_argument("--resume_from",    type=str, default="")
    parser.add_argument("--resume_epoch",   type=int, default=0)
    parser.add_argument("--reset_optimizer",action="store_true")

    # Hardware
    parser.add_argument("--flash_attn",    action="store_true")
    parser.add_argument("--compile",       action="store_true")
    parser.add_argument("--compile_mode",  type=str, default="default",
                        choices=["default", "reduce-overhead",
                                 "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--dist_backend",  type=str, default="gloo",
                        choices=["gloo", "nccl"])

    # Early stopping
    parser.add_argument("--early_stop_patience",  type=int,   default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001)


    return parser.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────

def format_lr(optimizer):
    parts = []
    names = ["llm", "enc+dec"]
    for i, pg in enumerate(optimizer.param_groups):
        name = names[i] if i < len(names) else f"g{i}"
        parts.append(f"{name}_lr={pg['lr']:.2e}")
    return ", ".join(parts)


def detect_gpu_config():
    if not torch.cuda.is_available():
        return 0, "no GPU", False
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return sm, torch.cuda.get_device_name(), sm >= 80


def safe_save(logger, model, optimizer, epoch, dirpath, name):
    try:
        raw = model.module if isinstance(model, DDP) else model
        save_ckpt(raw, optimizer, epoch, dirpath, name)
        full = os.path.join(dirpath, name)
        if os.path.exists(full):
            logger.print(f"  >> Saved {full} ({os.path.getsize(full)/1024**2:.1f} MB)")
        return True
    except Exception as e:
        logger.print(f"  !! save_ckpt FAILED for {name}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ.get("RANK", "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    device  = torch.device(f"cuda:{local_rank}")
    is_main = (global_rank == 0)

    args = parse_arguments()
    set_seed(args.seed + global_rank)

    date_str = dt.now().strftime("%Y%m%d-%H%M%S")
    logger   = Logger(date_str=date_str, enable=is_main,
                      log_dir=f"log/{args.exp_name}",
                      enable_flags={"writer": args.logger_writer})

    sm, gpu_name, flash_ok = detect_gpu_config()
    if sm >= 100 and args.dist_backend == "nccl":
        # Blackwell (sm>=100) is supported by NCCL>=2.21 / PyTorch>=2.5.
        # Allow NCCL unless the user explicitly requests gloo.
        print(f"[Rank {global_rank}] Blackwell sm{sm} — using NCCL backend")

    if args.dist_backend == "gloo":
        if sys.platform != "win32":
            os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["GLOO_SOCKET_NTHREADS"]  = "8"
    dist.init_process_group(backend=args.dist_backend,
                            timeout=datetime.timedelta(minutes=30),
                            device_id=device)

    logger.print(f"DDP backend={args.dist_backend} | world_size={world_size} | rank={global_rank}")

    if is_main:
        logger.log_basics(args=args, datetime=date_str)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        probe = os.path.join(args.ckpt_dir, f".write_probe_{date_str}")
        with open(probe, "w") as f: f.write("ok")
        os.remove(probe)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    train_dir = os.path.join(args.features_dir, "train")
    val_dir   = os.path.join(args.features_dir, "val")

    train_set = AV2HybridDataset(
        train_dir, tokenizer_name=args.model_name,
        max_agents=args.max_agents, max_lanes=args.max_lanes,
        max_text_len=args.max_text_len, max_text_agents=args.max_text_agents)
    val_set   = AV2HybridDataset(
        val_dir, tokenizer_name=args.model_name,
        max_agents=args.max_agents, max_lanes=args.max_lanes,
        max_text_len=args.max_text_len, max_text_agents=args.max_text_agents)

    full_train_set  = train_set
    full_train_size = len(train_set)
    use_subset = args.max_train_samples > 0 and full_train_size > args.max_train_samples
    logger.print(f"Train: {full_train_size} | Val: {len(val_set)} | "
                 f"max_text_len={args.max_text_len} | max_agents={args.max_agents} "
                 f"| max_lanes={args.max_lanes}")

    persistent = args.num_workers > 0
    val_sampler = DistributedSampler(val_set, num_replicas=world_size,
                                     rank=global_rank, shuffle=False)
    dl_val = DataLoader(val_set, batch_size=args.val_batch_size, sampler=val_sampler,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=persistent, collate_fn=hybrid_collate_fn)

    if not use_subset:
        train_sampler = DistributedSampler(full_train_set, num_replicas=world_size,
                                           rank=global_rank, shuffle=True)
        dl_train = DataLoader(full_train_set, batch_size=args.train_batch_size,
                              sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=persistent,
                              collate_fn=hybrid_collate_fn, drop_last=True)
    else:
        train_sampler = dl_train = None

    iters_per_epoch = (args.max_train_samples // (args.train_batch_size * world_size)
                       if use_subset else len(dl_train))
    logger.print(f"GPU: {gpu_name} SM={sm} | flash_attn_compat={flash_ok}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    use_flash_attn = args.flash_attn
    if use_flash_attn and not flash_ok:
        logger.print(f"WARNING: sm{sm} may not support flash_attn — proceeding anyway")

    import time as _time
    lock_file = "/tmp/simpl_hybrid_model_load.lock"

    if global_rank == 0:
        with open(lock_file, "w") as f: f.write(str(os.getpid()))
        model = HybridLLMPredictor(
            model_name=args.model_name,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_targets if args.lora_targets == "all-linear"
                                else args.lora_targets.split(","),
            lora_dropout=args.lora_dropout,
            n_levels=args.n_levels,
            max_agents=args.max_agents, max_lanes=args.max_lanes,
            num_modes=args.num_modes, n_bezier_ctrl=args.n_bezier_ctrl,
            rgsf_layers=args.rgsf_layers, rgsf_dim=args.rgsf_dim,
            use_flash_attn=use_flash_attn, dtype=torch.bfloat16)
        model = model.to(device).to(torch.bfloat16)
        if os.path.exists(lock_file): os.remove(lock_file)
    else:
        waited = 0
        while os.path.exists(lock_file) and waited < 300:
            _time.sleep(0.5); waited += 0.5
        _time.sleep(1)
        model = HybridLLMPredictor(
            model_name=args.model_name,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_targets if args.lora_targets == "all-linear"
                                else args.lora_targets.split(","),
            lora_dropout=args.lora_dropout,
            n_levels=args.n_levels,
            max_agents=args.max_agents, max_lanes=args.max_lanes,
            num_modes=args.num_modes, n_bezier_ctrl=args.n_bezier_ctrl,
            rgsf_layers=args.rgsf_layers, rgsf_dim=args.rgsf_dim,
            use_flash_attn=use_flash_attn, dtype=torch.bfloat16)
        model = model.to(device).to(torch.bfloat16)

    dist.barrier()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)

    # Ablation knobs on the underlying module so they survive DDP wrapping.
    # llm_correction_scale=0.0 → encoder-only (LLM correction zeroed).
    # scene_guidance_scale=0.0 → exp12-A (scenario queries without LLM guidance).
    _raw = model.module if isinstance(model, DDP) else model
    _raw.llm_correction_scale = args.llm_correction_scale
    _raw.traj_decoder.scene_guidance_scale = args.scene_guidance_scale

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, dynamic=False)
        logger.print(f"torch.compile enabled (mode={args.compile_mode})")

    # ── 3. Loss ───────────────────────────────────────────────────────────────
    loss_fn = HybridMotionLoss(n_levels=args.n_levels, device=device,
                               soft_wta_alpha=args.soft_wta_alpha,
                               cls_weight=args.cls_weight)
    if is_main:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.print(f"Params: trainable={n_train:,} / total={n_total:,} "
                     f"({100.*n_train/max(n_total,1):.2f}%)")
        logger.print(f"Loss: Level-k HybridMotionLoss | n_levels={args.n_levels} | "
                     f"weights={loss_fn.level_weights}")
        logger.print(f"[Ablation] llm_correction_scale={args.llm_correction_scale} "
                     f"({'encoder-only, LLM correction zeroed' if args.llm_correction_scale == 0.0 else 'normal LLM path'})")
        logger.print(f"[Ablation] scene_guidance_scale={args.scene_guidance_scale} "
                     f"({'exp12-A: no LLM scenario guidance' if args.scene_guidance_scale == 0.0 else 'LLM scenario guidance ON'})")

    # ── 4. Optimiser (Differential LR) ────────────────────────────────────────
    # llm_correction_proj / scene_proj bridge LLM→decoder space; they must be in
    # the slow group (llm_lr) alongside LoRA. If placed in enc_params (gru_lr=2×
    # faster), they outpace the LoRA weights that produce h_llm/h_scn, amplifying
    # random LLM noise into the decoder before the LLM has converged
    # (diag_corr_ratio >> 1 at ep1 — observed in exp10 before the fix).
    _slow_keywords = ("llm.",  "llm_correction_proj", "scene_proj")
    llm_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and any(k in n for k in _slow_keywords)]
    enc_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and not any(k in n for k in _slow_keywords)]
    optimizer = AdamW([
        {"params": llm_params, "lr": args.llm_lr},
        {"params": enc_params, "lr": args.gru_lr},
    ], weight_decay=1e-4)
    logger.print(f"Optimiser: llm_params={sum(p.numel() for p in llm_params):,} "
                 f"| enc+dec_params={sum(p.numel() for p in enc_params):,}")

    # ── 5. Scheduler ──────────────────────────────────────────────────────────
    if args.scheduler == "cosine_warmup_restart":
        _T0, _wu, _r = args.T_0, args.warmup_epochs, args.eta_min_ratio
        def _lr_lambda(epoch):
            cp = epoch % _T0
            if cp < _wu:
                return _r + (1.0 - _r) * cp / max(1, _wu)
            prog = (cp - _wu) / max(1, _T0 - _wu)
            return _r + (1.0 - _r) * 0.5 * (1.0 + math.cos(math.pi * prog))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        logger.print(f"Scheduler: CosineWarmupRestart T_0={_T0} warmup={_wu} eta_min={_r}")
    elif args.scheduler == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=args.T_mult)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.train_epoches)

    # ── 6. Resume ─────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location={f"cuda:0": f"cuda:{local_rank}"})
        raw  = model.module if isinstance(model, DDP) else model
        raw.load_state_dict(ckpt["state_dict"], strict=False)
        if not args.reset_optimizer:
            optimizer.load_state_dict(ckpt["opt_state"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        start_epoch = args.resume_epoch
        for _ in range(start_epoch): scheduler.step()
        logger.print(f"Resumed from {args.resume_from} | epoch {start_epoch+1}")

    # ── 7. Training Loop ──────────────────────────────────────────────────────
    niter         = start_epoch * args.train_batch_size * world_size * iters_per_epoch
    best_val_loss = float("inf")
    best_minADE   = float("inf")
    patience_ctr  = 0
    should_stop   = torch.zeros(1, dtype=torch.int32, device=device)
    # Prefix checkpoints with exp_name so weights and logs (log/{exp_name}/) share
    # the same experiment counter — easy to pair up and archive afterwards.
    last_ckpt     = f"{args.exp_name}_{date_str}_hybrid_last.tar"
    best_ckpt     = f"{args.exp_name}_{date_str}_hybrid_best.tar"

    for epoch in range(start_epoch, args.train_epoches):
        dist.barrier()

        if use_subset:
            ep_idx = torch.randperm(
                full_train_size,
                generator=torch.Generator().manual_seed(args.seed + epoch)
            )[:args.max_train_samples].tolist()
            ep_set  = torch.utils.data.Subset(full_train_set, ep_idx)
            train_sampler = DistributedSampler(ep_set, num_replicas=world_size,
                                               rank=global_rank, shuffle=True)
            dl_train = DataLoader(ep_set, batch_size=args.train_batch_size,
                                  sampler=train_sampler, num_workers=args.num_workers,
                                  pin_memory=True, persistent_workers=False,
                                  collate_fn=hybrid_collate_fn, drop_last=True)
            iters_per_epoch = len(dl_train)

        train_sampler.set_epoch(epoch)

        if is_main:
            logger.print("\n" + "=" * 80)
            logger.print(f"Epoch {epoch+1}/{args.train_epoches}")
            logger.print(f"  - LR              : {format_lr(optimizer)}")
            logger.print(f"  - Batch (GPU/eff) : {args.train_batch_size} / "
                         f"{args.train_batch_size * world_size * args.grad_accum_steps}")
            logger.print(f"  - LoRA r/α        : {args.lora_r}/{args.lora_alpha}")
            logger.print(f"  - Level-k          : {args.n_levels}")
            logger.print(f"  - Agents/Lanes     : {args.max_agents}/{args.max_lanes}")
            logger.print("=" * 80)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        meter       = AverageMeterForDict()
        loss_window = deque(maxlen=args.running_window)
        epoch_start = time.time()
        accum       = args.grad_accum_steps
        grad_norm_llm = torch.tensor(0.0)
        grad_norm_enc = torch.tensor(0.0)
        optimizer.zero_grad()

        pbar = tqdm(dl_train, disable=(not is_main or args.no_pbar),
                    ncols=120, desc=f"[Train ep {epoch+1}]")

        for i, data in enumerate(pbar):
            text_ids  = data["text_input_ids"].to(device)
            text_mask = data["text_attention_mask"].to(device)
            ag_feat   = data["agent_features"].to(device)
            ag_valid  = data["agent_valid"].to(device)
            lane_feat = data["lane_features"].to(device)
            lane_valid= data["lane_valid"].to(device)
            gt_anchor = data["gt_anchor"].to(device)

            is_last_accum = ((i + 1) % accum == 0) or ((i + 1) == iters_per_epoch)
            sync_ctx = contextlib.nullcontext() if is_last_accum else model.no_sync()

            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    all_preds, all_scores, codebook = model(text_ids, text_mask, ag_feat, ag_valid,
                                                            lane_feat, lane_valid, gt_anchor)
                    loss_out  = loss_fn(all_preds, data, all_scores, codebook)

                (loss_out["loss"] / accum).backward()

            if is_last_accum:
                # Separate clipping: LLM LoRA vs encoder+decoder
                grad_norm_llm = torch.nn.utils.clip_grad_norm_(
                    llm_params, args.llm_grad_clip)
                grad_norm_enc = torch.nn.utils.clip_grad_norm_(
                    enc_params, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            scalars = {k: v.item() for k, v in loss_out.items()}
            meter.update(scalars)
            loss_window.append(scalars["loss"])
            niter += args.train_batch_size * world_size

            if is_main:
                logger.add_dict(scalars, niter, prefix="train/")

            gnorm_llm_val = grad_norm_llm.item() if torch.is_tensor(grad_norm_llm) else float(grad_norm_llm)
            gnorm_enc_val = grad_norm_enc.item() if torch.is_tensor(grad_norm_enc) else float(grad_norm_enc)
            running_loss  = sum(loss_window) / len(loss_window)
            pbar.set_postfix({
                "loss":   f"{scalars['loss']:.4f}",
                f"avg{len(loss_window)}": f"{running_loss:.4f}",
                "gn_enc": f"{gnorm_enc_val:.2f}",
                "gn_llm": f"{gnorm_llm_val:.2f}",
            })

            if is_main and ((i+1) % args.print_interval == 0 or (i+1) == iters_per_epoch):
                elapsed = time.time() - epoch_start
                ips     = (i+1) / max(elapsed, 1e-6)
                eta     = (iters_per_epoch - (i+1)) / max(ips, 1e-6)
                mem     = torch.cuda.memory_allocated(device) // 2**20
                extra   = " | ".join(f"{k}={v:.4f}" for k, v in scalars.items()
                                     if k != "loss")
                logger.print(
                    f"  [ep {epoch+1} | {i+1}/{iters_per_epoch}] "
                    f"loss={scalars['loss']:.4f} (avg={running_loss:.4f}) | "
                    f"{extra} | gn_enc={gnorm_enc_val:.2f} gn_llm={gnorm_llm_val:.2f} | "
                    f"{ips:.2f} it/s | eta={eta/60:.1f}min | mem={mem}MB"
                )

        pbar.close()
        scheduler.step()
        peak = torch.cuda.max_memory_allocated(device) // 2**20
        logger.print(f"[Train] ep {epoch+1} avg_loss={meter.metrics['loss'].avg:.4f} | "
                     f"time={(time.time()-epoch_start)/60:.2f}min | peak_mem={peak}MB")

        if args.save_last_every_epoch and is_main:
            safe_save(logger, model, optimizer, epoch, args.ckpt_dir, last_ckpt)

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % args.val_interval != 0:
            continue

        dist.barrier()
        model.eval()
        val_meter = AverageMeterForDict()
        ade_meter = AverageMeterForDict()
        val_start = time.time()
        # Winner-mode usage histogram (mode-collapse monitor, exp12): counts how
        # often each mode is the per-scene oracle FDE winner across the val set.
        mode_hist = torch.zeros(args.num_modes, device=device)

        val_sampler.set_epoch(epoch)

        pbar_v = tqdm(dl_val, disable=(not is_main or args.no_pbar),
                      ncols=120, desc=f"[Val   ep {epoch+1}]")
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

                # ── JOINT scene-level avgMinADE / avgMinFDE over ALL scored agents ──
                # The K modes are K joint "worlds"; one shared mode per scene is
                # chosen to minimise the joint error, then ADE/FDE is averaged over
                # scored agents. Matches the AV2 multi-world avgMinADE(K=6) protocol.
                pred_pos   = all_preds[-1].float()                              # [B, N, K, T, 2]
                anchor     = gt_anchor.float()                                  # [B, N, 2]
                pred_abs   = anchor.unsqueeze(2).unsqueeze(2) + pred_pos        # [B, N, K, T, 2]
                gt_abs_f   = data["gt_abs_trajectories"].to(device).float()     # [B, N, T, 2]
                gt_masks_f = data["gt_masks"].to(device)                        # [B, N, T]
                train_m    = data["train_mask"].to(device)                      # [B, N]

                # valid future steps for scored agents only
                vmask      = (gt_masks_f & train_m.unsqueeze(-1)).float()       # [B, N, T]
                l2_dist    = (pred_abs - gt_abs_f.unsqueeze(2)).norm(dim=-1)    # [B, N, K, T]

                # per-agent, per-mode ADE (mean over that agent's valid steps)
                step_cnt     = vmask.sum(dim=-1).clamp(min=1)                   # [B, N]
                per_mode_ade = (l2_dist * vmask.unsqueeze(2)).sum(dim=-1) \
                               / step_cnt.unsqueeze(-1)                         # [B, N, K]

                # per-agent, per-mode FDE (at that agent's last valid step)
                last_t       = (gt_masks_f.long().cumsum(-1) *
                                gt_masks_f.long()).argmax(-1)                   # [B, N]
                idx_fde      = last_t.unsqueeze(-1).unsqueeze(-1).expand(
                                   -1, -1, l2_dist.shape[2], 1)                 # [B, N, K, 1]
                per_mode_fde = l2_dist.gather(3, idx_fde).squeeze(3)            # [B, N, K]

                # scene-level mode aggregation: average over scored agents
                scored       = train_m.float()                                 # [B, N]
                agent_cnt    = scored.sum(dim=1).clamp(min=1)                   # [B]
                scene_ade    = (per_mode_ade * scored.unsqueeze(-1)).sum(dim=1) \
                               / agent_cnt.unsqueeze(-1)                        # [B, K]
                scene_fde    = (per_mode_fde * scored.unsqueeze(-1)).sum(dim=1) \
                               / agent_cnt.unsqueeze(-1)                        # [B, K]

                MISS = 2.0   # AV2 miss threshold: FDE > 2.0 m counts as a miss

                # ── MULTI-AGENT (multi-world) metrics: ONE shared best world per ──
                # scene, selected by avg FDE over scored actors (AV2 multi-agent
                # protocol); all multi-agent metrics are reported in that world.
                scene_logit = (all_scores[-1].float() * scored.unsqueeze(-1)).sum(dim=1) \
                              / agent_cnt.unsqueeze(-1)                        # [B, K]
                scene_prob  = torch.softmax(scene_logit, dim=-1)              # [B, K]
                best        = scene_fde.argmin(dim=1)                         # [B] shared world
                mode_hist  += torch.bincount(best, minlength=args.num_modes).float()

                avg_min_fde = scene_fde.gather(1, best.unsqueeze(1)).squeeze(1)        # [B]
                avg_min_ade = scene_ade.gather(1, best.unsqueeze(1)).squeeze(1).mean()
                p_best      = scene_prob.gather(1, best.unsqueeze(1)).squeeze(1)        # [B]
                avg_brier_fde = (avg_min_fde + (1.0 - p_best) ** 2).mean()
                avg_min_fde = avg_min_fde.mean()
                # actorMR: fraction of scored actors with FDE > 2 m in the chosen world
                fde_win  = per_mode_fde.gather(
                    2, best[:, None, None].expand(-1, per_mode_fde.shape[1], 1)).squeeze(2)  # [B,N]
                actor_mr = ((fde_win > MISS).float() * scored).sum() / scored.sum().clamp(min=1)
                # actorCR: collision rate among scored actors in the chosen world.
                # Centroid-distance approximation (no bbox dims available): two actors
                # collide if their predicted centroids are < COLLISION m apart at any
                # commonly-valid future step. Value is indicative, not the bbox-exact AV2 CR.
                COLLISION = 2.0
                Tf = pred_abs.shape[3]
                pred_win = pred_abs.gather(
                    2, best[:, None, None, None, None].expand(-1, pred_abs.shape[1], 1, Tf, 2)
                ).squeeze(2)                                                   # [B, N, T, 2]
                pdist = (pred_win[:, :, None] - pred_win[:, None, :]).norm(dim=-1)   # [B,N,N,T]
                vt    = (gt_masks_f & train_m.unsqueeze(-1))                   # [B, N, T] bool
                pair_v = vt[:, :, None] & vt[:, None, :]                       # [B,N,N,T]
                eye    = torch.eye(pred_win.shape[1], device=device, dtype=torch.bool)
                pair_v = pair_v & ~eye[None, :, :, None]                       # exclude self
                actor_collide = ((pdist < COLLISION) & pair_v).any(dim=(2, 3))  # [B, N]
                actor_cr = (actor_collide.float() * scored).sum() / scored.sum().clamp(min=1)

                # ── FOCAL-agent metrics (agent 0; AV2 single-agent leaderboard, K=6) ──
                f_ade_pm, f_fde_pm = per_mode_ade[:, 0, :], per_mode_fde[:, 0, :]  # [B, K]
                focal_ade = f_ade_pm.min(dim=1).values.mean()
                focal_fde = f_fde_pm.min(dim=1).values.mean()
                focal_mr  = (f_fde_pm.min(dim=1).values > MISS).float().mean()
                f_logit = all_scores[-1][:, 0, :].float()
                f_prob  = torch.softmax(f_logit, dim=-1)
                f_best  = f_fde_pm.argmin(dim=1)
                f_pbest = f_prob.gather(1, f_best.unsqueeze(1)).squeeze(1)
                focal_bfde = (f_fde_pm.min(dim=1).values + (1.0 - f_pbest) ** 2).mean()

                ade_meter.update({
                    "avgMinADE": avg_min_ade.item(), "avgMinFDE": avg_min_fde.item(),
                    "avgBrierMinFDE": avg_brier_fde.item(), "actorMR": actor_mr.item(),
                    "actorCR": actor_cr.item(),
                    "focalMinADE": focal_ade.item(), "focalMinFDE": focal_fde.item(),
                    "focalBrierFDE": focal_bfde.item(), "focalMR": focal_mr.item()})
                if is_main:
                    pbar_v.set_postfix({
                        "loss":      f"{val_meter.metrics['loss'].avg:.4f}",
                        "avgMinADE": f"{ade_meter.metrics['avgMinADE'].avg:.4f}",
                        "avgMinFDE": f"{ade_meter.metrics['avgMinFDE'].avg:.4f}",
                        "actorMR":   f"{ade_meter.metrics['actorMR'].avg:.4f}",
                    })

        pbar_v.close()

        # Aggregate ALL metrics across ranks in a SINGLE collective call.
        # CRITICAL: distributed_mean() calls dist.all_gather() which is a collective
        # op — every rank MUST call it the same number of times, or DDP deadlocks.
        # All per-level loss keys are bundled here (NOT inside `if is_main`).
        # Ordered metric keys (extend here to add metrics — aggregation is generic).
        ade_keys = ["avgMinADE", "avgMinFDE", "avgBrierMinFDE", "actorMR", "actorCR",
                    "focalMinADE", "focalMinFDE", "focalBrierFDE", "focalMR"]
        extra_keys  = [k for k in val_meter.metrics if k != "loss"]
        local_vals  = ([val_meter.metrics["loss"].avg]
                       + [ade_meter.metrics[k].avg for k in ade_keys]
                       + [val_meter.metrics[k].avg for k in extra_keys])
        local_tensor  = torch.tensor(local_vals, dtype=torch.float64, device=device)
        global_tensor = distributed_mean(local_tensor)          # collective on every rank
        global_vals   = global_tensor.tolist()
        val_loss_avg  = global_vals[0]
        ade_avgs      = dict(zip(ade_keys, global_vals[1:1 + len(ade_keys)]))
        extra_avgs    = dict(zip(extra_keys, global_vals[1 + len(ade_keys):]))
        min_ade_avg   = ade_avgs["avgMinADE"]   # drives best-ckpt / early-stop (multi-agent)
        dist.all_reduce(mode_hist)                              # collective on every rank

        if is_main:
            logger.print(
                f"[Validation-MultiAgent] ep {epoch+1} | loss: {val_loss_avg:.4f} | "
                f"avgMinADE6: {ade_avgs['avgMinADE']:.4f} m | avgMinFDE6: {ade_avgs['avgMinFDE']:.4f} m | "
                f"avgBrierMinFDE6: {ade_avgs['avgBrierMinFDE']:.4f} | actorMR6: {ade_avgs['actorMR']:.4f} | "
                f"actorCR6: {ade_avgs['actorCR']:.4f} | time: {(time.time()-val_start)/60:.3f} mins"
            )
            logger.print(
                f"[Validation-Focal] ep {epoch+1} | "
                f"minADE6: {ade_avgs['focalMinADE']:.4f} m | minFDE6: {ade_avgs['focalMinFDE']:.4f} m | "
                f"b-minFDE6: {ade_avgs['focalBrierFDE']:.4f} | MR6: {ade_avgs['focalMR']:.4f}  "
                f"(agent-0, single-agent leaderboard-comparable)"
            )
            logger.add_scalar("val/loss", val_loss_avg, it=epoch)
            for k, v in ade_avgs.items():
                logger.add_scalar(f"val/{k}", v, it=epoch)
            for k, v in extra_avgs.items():
                logger.add_scalar(f"val/{k}", v, it=epoch)

            # ── Anchor codebook diagnostics ──────────────────────────────────
            # Buffers are kept identical across ranks (all_reduce in ema_update),
            # so rank-0 is representative. Print each anchor's (x, y) position,
            # its EMA assignment count, and the spread (min pairwise distance) so
            # mode collapse is visible at a glance.
            raw_cb = model.module if isinstance(model, DDP) else model

            # ── LLM / lane branch contribution (is the LLM actually working?) ──
            # getattr-guarded so a stale model file (without the diag attrs) prints
            # nan instead of crashing the whole run.
            _corr = getattr(raw_cb, "diag_corr_ratio", float("nan"))
            _lane = getattr(raw_cb, "diag_lane_ratio", float("nan"))
            _scn  = getattr(raw_cb, "diag_scn_ratio",  float("nan"))
            _rgsf = getattr(raw_cb, "diag_rgsf_ratio", float("nan"))
            logger.print(f"[Branch] ep {epoch+1} | LLM corr/enc: {_corr:.4f} "
                         f"| lane_cross/input: {_lane:.4f} "
                         f"| scn_q/base_q: {_scn:.4f} "
                         f"| rgsf/enc: {_rgsf:.4f}")
            logger.add_scalar("diag/llm_corr_ratio", _corr, it=epoch)
            logger.add_scalar("diag/lane_ratio",     _lane, it=epoch)
            logger.add_scalar("diag/scn_ratio",      _scn,  it=epoch)
            logger.add_scalar("diag/rgsf_ratio",     _rgsf, it=epoch)

            # ── Winner-mode usage histogram (mode-collapse monitor) ───────────
            # With anchor-cls removed, scene-WTA is the only diversity mechanism;
            # a healthy run keeps several modes in play. One mode taking >80% of
            # scenes signals collapse (mitigation: scene-level entropy regulariser).
            _hist = mode_hist / mode_hist.sum().clamp(min=1)
            _hstr = " ".join(f"m{j}:{100*v:.1f}%" for j, v in enumerate(_hist.tolist()))
            logger.print(f"[Modes] ep {epoch+1} | oracle-winner usage: {_hstr}")
            for j in range(args.num_modes):
                logger.add_scalar(f"modes/winner_share_{j}", _hist[j].item(), it=epoch)

            cb     = raw_cb.traj_decoder.codebook
            anc    = cb.anchors.float().cpu()                       # [K, 2]
            cnt    = cb.ema_count.float().cpu()                     # [K]
            K_anc  = anc.shape[0]
            pdist  = torch.cdist(anc, anc)                          # [K, K]
            pdist.fill_diagonal_(float("inf"))
            min_sep = pdist.min().item()
            logger.print(f"[Anchors] ep {epoch+1} | min pairwise sep: {min_sep:.3f} m")
            for j in range(K_anc):
                logger.print(f"    anchor[{j}] = ({anc[j,0]:+7.2f}, {anc[j,1]:+7.2f}) "
                             f"| ema_count = {cnt[j]:8.2f}")
                logger.add_scalar(f"anchor/cnt_{j}", cnt[j].item(), it=epoch)
                logger.add_scalar(f"anchor/x_{j}",   anc[j, 0].item(), it=epoch)
                logger.add_scalar(f"anchor/y_{j}",   anc[j, 1].item(), it=epoch)
            logger.add_scalar("anchor/min_sep", min_sep, it=epoch)

        # Save best checkpoint
        if min_ade_avg < best_minADE - args.early_stop_min_delta:
            improvement  = best_minADE - min_ade_avg
            best_minADE  = min_ade_avg
            best_val_loss = val_loss_avg
            patience_ctr  = 0
            if is_main:
                logger.print(
                    f"  >> avgMinADE improved by {improvement:.4f} m → {best_minADE:.4f} m; "
                    f"saving best checkpoint...")
                safe_save(logger, model, optimizer, epoch, args.ckpt_dir, best_ckpt)
        else:
            patience_ctr += 1
            patience_str = (f" | patience {patience_ctr}/{args.early_stop_patience}"
                            if args.early_stop_patience > 0 else "")
            if is_main:
                logger.print(
                    f"  -- No avgMinADE improvement (best={best_minADE:.4f} m, "
                    f"current={min_ade_avg:.4f} m){patience_str}")

        if args.early_stop_patience > 0 and patience_ctr >= args.early_stop_patience:
            if is_main:
                should_stop.fill_(1)
            dist.broadcast(should_stop, src=0)
            if should_stop.item():
                logger.print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    logger.print(f"\nDone. Best minADE={best_minADE:.4f}m | best_val_loss={best_val_loss:.4f}")
    dist.destroy_process_group()



if __name__ == "__main__":
    main()
