#!/usr/bin/env python3
"""
train_qwen_copyne.py: DDP Training for Frozen Qwen3-ASR-0.6B + CopyNE + U2++ Decoder.
Optimized for Kaggle 2x NVIDIA T4 (2x16GB VRAM, Turing sm_75).

Features:
- DDP multi-GPU (torchrun --nproc_per_node=2) and single-GPU fallback.
- Distributed dev evaluation with all_reduce to prevent rank idle timeout.
- Differential learning rates: 1e-3 for CopyNE/Dual-LSTM/U2++, 1e-5 for base LM head.
- Checkpoint saving of trainable parameters only (~150MB instead of ~1.5GB).
- Session time limit safeguard (--max_hours 11.0) for Kaggle 12h runtime limit.
- Resumption support via --pre_model for Stage 2 (U2++ rescoring).
"""

import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from typing import Dict, Any

import orjson
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml

from model.qwen_copyne_u2pp import QwenCopyNEU2PlusPlus
from utils.qwen_copyne_data import QwenCopyNEDataset, QwenCopyNECollator

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_qwen_copyne")


def setup_ddp():
    """Initialize Distributed Data Parallel if running under torchrun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=60),
        )
        return rank, local_rank, world_size
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Frozen Qwen3-ASR + CopyNE + U2++ on Kaggle 2x T4",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=str, default="conf/qwen3_copyne_u2pp.yaml", help="Path to YAML config")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train JSONL with entities")
    parser.add_argument("--dev_data", type=str, required=True, help="Path to dev JSONL with entities")
    parser.add_argument("--output_dir", type=str, default="exp/qwen3_copyne", help="Directory to save checkpoints")
    parser.add_argument("--pre_model", type=str, default=None, help="Path to initial checkpoint to resume or warm-start")
    parser.add_argument("--alpha_r2l", type=float, default=None, help="R2L loss weight override (0.0 for L2R only, 0.3 for U2++)")
    parser.add_argument("--alpha_copy", type=float, default=None, help="Copy loss weight override")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device batch size override")
    parser.add_argument("--accum_grad", type=int, default=None, help="Gradient accumulation steps override")
    parser.add_argument("--max_epochs", type=int, default=None, help="Max training epochs override")
    parser.add_argument("--lr_modules", type=float, default=None, help="Learning rate for new modules")
    parser.add_argument("--lr_base_lm", type=float, default=None, help="Learning rate for base LM head")
    parser.add_argument("--max_hours", type=float, default=None, help="Max runtime in hours before graceful exit (default from config: 11.0)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes per GPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=20, help="Logging step frequency")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Validation step frequency")
    return parser.parse_args()


def evaluate_dev(model: nn.Module, dev_loader: DataLoader, device: torch.device, world_size: int) -> float:
    """Run validation across all DDP ranks and compute global average loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dev_loader:
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss_dict = model(batch)
                loss = loss_dict["loss"]

            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if world_size > 1 and dist.is_initialized():
        stats = torch.tensor([total_loss, float(total_samples)], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        avg_loss = stats[0].item() / max(1.0, stats[1].item())
    else:
        avg_loss = total_loss / max(1, total_samples)

    return avg_loss


def check_time_limit_ddp(
    start_wall_time: float,
    max_hours: Any,
    device: torch.device,
    world_size: int,
) -> tuple[bool, float]:
    """Check wall-clock limit on Rank 0 and broadcast decision to all DDP ranks."""
    if max_hours is None or max_hours <= 0:
        return False, 0.0
    elapsed_hours = (time.monotonic() - start_wall_time) / 3600.0
    stop_flag = 1.0 if elapsed_hours >= float(max_hours) else 0.0
    if world_size > 1 and dist.is_initialized():
        sync_tensor = torch.tensor([stop_flag, elapsed_hours], device=device, dtype=torch.float64)
        dist.broadcast(sync_tensor, src=0)
        stop_flag = sync_tensor[0].item()
        elapsed_hours = sync_tensor[1].item()
    return (stop_flag >= 0.5), elapsed_hours


def save_checkpoint(
    path: str,
    raw_model: QwenCopyNEU2PlusPlus,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    best_dev_loss: float,
    args: Any,
    epoch_completed: bool = True,
):
    """Save only trainable parameters and optimizer state to conserve disk and bandwidth."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": raw_model.get_trainable_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "epoch_completed": epoch_completed,
        "global_step": global_step,
        "best_dev_loss": best_dev_loss,
        "args": vars(args),
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved to {path} (size: {os.path.getsize(path) / 1024 / 1024:.1f} MB)")


def main():
    start_wall_time = time.monotonic()
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main_process = (rank == 0)

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    config = load_config(args.config)

    # CLI Overrides
    batch_size = args.batch_size or config["training"].get("batch_size", 8)
    accum_grad = args.accum_grad or config["training"].get("accum_grad", 2)
    max_epochs = args.max_epochs or config["training"].get("max_epochs", 15)
    lr_modules = args.lr_modules or float(config["training"].get("lr_modules", 1e-3))
    lr_base_lm = args.lr_base_lm or float(config["training"].get("lr_base_lm", 1e-5))
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    alpha_r2l = args.alpha_r2l if args.alpha_r2l is not None else float(config["training"].get("alpha_r2l", 0.3))
    alpha_copy = args.alpha_copy if args.alpha_copy is not None else float(config["training"].get("alpha_copy", 1.0))
    warmup_steps = int(config["training"].get("warmup_steps", 1000))
    clip_grad_norm = float(config["training"].get("clip_grad_norm", 5.0))
    raw_max_hours = args.max_hours if args.max_hours is not None else config["training"].get("max_hours", 11.0)
    max_hours = float(raw_max_hours) if (raw_max_hours is not None and float(raw_max_hours) > 0) else None

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Starting training on {world_size} GPU(s). Local rank: {local_rank}")
        logger.info(f"Effective Batch Size = {batch_size} * {world_size} * {accum_grad} = {batch_size * world_size * accum_grad}")
        logger.info(f"alpha_r2l: {alpha_r2l}, alpha_copy: {alpha_copy}, lr_modules: {lr_modules}, lr_base_lm: {lr_base_lm}, max_hours: {max_hours}")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1. Load Pretrained Qwen3-ASR-0.6B Backbone
    qwen_model_id = config["model"].get("qwen_model_id", "Qwen/Qwen3-ASR-0.6B-hf")
    if is_main_process:
        logger.info(f"Loading frozen Qwen3 backbone: {qwen_model_id}")

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        processor = AutoProcessor.from_pretrained(qwen_model_id, fix_markdown=False)
        qwen_backbone = AutoModelForCausalLM.from_pretrained(
            qwen_model_id,
            torch_dtype=torch.float16,
            device_map=None,
        )
    except Exception as e:
        logger.error(f"Failed to load Qwen3 backbone from {qwen_model_id}: {e}")
        raise

    # 2. Build QwenCopyNEU2PlusPlus Model
    model = QwenCopyNEU2PlusPlus(
        qwen_model=qwen_backbone,
        d_e=config["model"].get("d_e", 512),
        d_a=config["model"].get("d_a", 512),
        emb_proj_dim=config["model"].get("emb_proj_dim", 256),
        lstm_layers=config["model"].get("lstm_layers", 3),
        r2l_mode=config["model"].get("r2l_mode", "cross_attn"),
        r2l_dim=config["model"].get("d_e", 512),
        r2l_layers=config["model"].get("r2l_layers", 3),
        alpha_r2l=alpha_r2l,
        freeze_base_lm_head=config["model"].get("freeze_base_lm", True),
        dropout=config["model"].get("lstm_dropout", 0.1),
    )
    model.to(device)

    # 3. Resume / Warm-Start from Checkpoint if requested
    start_epoch = 0
    global_step = 0
    best_dev_loss = float("inf")

    if args.pre_model and os.path.exists(args.pre_model):
        if is_main_process:
            logger.info(f"Loading checkpoint from: {args.pre_model}")
        try:
            ckpt = torch.load(args.pre_model, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_trainable_state_dict(state_dict, strict=False)
            if "epoch" in ckpt:
                epoch_done = ckpt.get("epoch_completed", True)
                start_epoch = ckpt["epoch"] + (1 if epoch_done else 0)
            if "global_step" in ckpt:
                global_step = ckpt["global_step"]
            if "best_dev_loss" in ckpt:
                best_dev_loss = ckpt["best_dev_loss"]
            if is_main_process:
                logger.info(f"Loaded successfully. Resuming at epoch {start_epoch}, step {global_step}")
        except Exception as e:
            logger.error(f"Error loading checkpoint from {args.pre_model}: {e}")
            raise

    # 4. Prepare DDP Wrapper
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        raw_model = model.module
    else:
        raw_model = model

    # 5. Datasets and DataLoaders
    max_audio_secs = config["data"].get("max_audio_length_secs", 30.0)
    train_dataset = QwenCopyNEDataset(args.train_data, processor=processor, max_audio_secs=max_audio_secs)
    dev_dataset = QwenCopyNEDataset(args.dev_data, processor=processor, max_audio_secs=max_audio_secs)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dev_sampler = DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    collator = QwenCopyNECollator(processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        sampler=dev_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )

    # 6. Optimizer and Scaler
    param_groups = raw_model.get_optimizer_param_groups(
        lr_modules=lr_modules,
        lr_base_lm=lr_base_lm,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    total_training_steps = max(1, (len(train_loader) // accum_grad) * max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_training_steps, eta_min=1e-6)

    # 7. Training Loop with Kaggle Session Timeout Safeguard
    session_timeout_triggered = False

    if is_main_process:
        logger.info(f"Beginning training: {max_epochs} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(start_epoch, max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        raw_model.qwen.eval()  # Frozen Qwen3 backbone stays in eval mode
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        accum_loss = 0.0
        step_start_time = time.monotonic()

        for step, batch in enumerate(train_loader):
            # Move batch tensors to local GPU
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss_dict = model(batch)
                loss = loss_dict["loss"]
                loss_scaled = loss / accum_grad

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item()

            if (step + 1) % accum_grad == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                epoch_loss += accum_loss / accum_grad
                accum_loss = 0.0

                if global_step % args.log_interval == 0 and is_main_process:
                    steps_per_sec = args.log_interval / max(1e-5, (time.monotonic() - step_start_time))
                    lr_curr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch {epoch:02d} | Step {global_step:06d} | "
                        f"Loss: {loss.item():.4f} (L2R: {loss_dict['loss_l2r'].item():.4f}, "
                        f"R2L: {loss_dict['loss_r2l'].item():.4f}) | "
                        f"LR: {lr_curr:.2e} | Speed: {steps_per_sec:.2f} step/s"
                    )
                    step_start_time = time.monotonic()

                # Step-based Evaluation
                if global_step % args.eval_interval == 0:
                    dev_loss = evaluate_dev(model, dev_loader, device, world_size)
                    if is_main_process:
                        logger.info(f"--> Dev Evaluation at Step {global_step}: Dev Loss = {dev_loss:.4f} (Prev Best: {best_dev_loss:.4f})")
                        if dev_loss < best_dev_loss:
                            best_dev_loss = dev_loss
                            best_path = os.path.join(args.output_dir, "checkpoint_best.pt")
                            save_checkpoint(best_path, raw_model, optimizer, scheduler, epoch, global_step, best_dev_loss, args, epoch_completed=False)
                    model.train()
                    raw_model.qwen.eval()

                # DDP-Synchronized Kaggle Session Limit Check (inside optimizer step block)
                reached, elapsed_hours = check_time_limit_ddp(start_wall_time, max_hours, device, world_size)
                if reached:
                    if is_main_process:
                        logger.warning(f"Elapsed time {elapsed_hours:.2f}h reached --max_hours {max_hours:.2f}h at step {global_step}. Triggering immediate session checkpoint.")
                    session_timeout_triggered = True
                    break

        if session_timeout_triggered:
            if world_size > 1 and dist.is_initialized():
                dist.barrier()
            if is_main_process:
                timeout_path = os.path.join(args.output_dir, "checkpoint_session_timeout.pt")
                save_checkpoint(timeout_path, raw_model, optimizer, scheduler, epoch, global_step, best_dev_loss, args, epoch_completed=False)
                last_path = os.path.join(args.output_dir, "checkpoint_last.pt")
                save_checkpoint(last_path, raw_model, optimizer, scheduler, epoch, global_step, best_dev_loss, args, epoch_completed=False)
                logger.info(f"Gracefully exiting session. Resume using: --pre_model {timeout_path}")
            if world_size > 1 and dist.is_initialized():
                dist.barrier()
            break

        # End of Epoch Evaluation & Checkpointing
        dev_loss = evaluate_dev(model, dev_loader, device, world_size)
        if is_main_process:
            logger.info(f"Epoch {epoch:02d} Complete. Dev Loss: {dev_loss:.4f} (Best: {best_dev_loss:.4f})")
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_path = os.path.join(args.output_dir, "checkpoint_best.pt")
                save_checkpoint(best_path, raw_model, optimizer, scheduler, epoch, global_step, best_dev_loss, args, epoch_completed=True)

            last_path = os.path.join(args.output_dir, "checkpoint_last.pt")
            save_checkpoint(last_path, raw_model, optimizer, scheduler, epoch, global_step, best_dev_loss, args, epoch_completed=True)

        reached, elapsed_hours = check_time_limit_ddp(start_wall_time, max_hours, device, world_size)
        if reached:
            if is_main_process:
                timeout_path = os.path.join(args.output_dir, "checkpoint_session_timeout.pt")
                save_checkpoint(timeout_path, raw_model, optimizer, scheduler, epoch, global_step, best_dev_loss, args, epoch_completed=True)
                logger.info(f"Elapsed time {elapsed_hours:.2f}h reached --max_hours {max_hours:.2f}h after Epoch {epoch:02d}. Resume using: --pre_model {timeout_path}")
            break

    cleanup_ddp()
    if is_main_process:
        logger.info("Training script completed successfully.")


if __name__ == "__main__":
    main()

