"""
train_ddp.py - Minimal but production-ready DDP training template with:
 - checkpoint save & resume
 - LR scheduler (step / cosine / reduce_on_plateau)
 - logging only on main process
 - AMP (autocast + GradScaler)
 - DistributedSampler.set_epoch
Usage:
  torchrun --nproc_per_node=4 train_ddp.py --data fake --epochs 10 --batch-size 32
"""

import os
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--nworkers", type=int, default=4)
    p.add_argument("--backend", type=str, default="nccl", help="dist backend")
    p.add_argument("--data", type=str, default="fake", choices=["fake"], help="placeholder for dataset")
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume")
    p.add_argument("--save-every", type=int, default=1, help="save checkpoint every N epochs")
    p.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine", "reduce_on_plateau"])
    p.add_argument("--step-size", type=int, default=10, help="StepLR step_size")
    p.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma")
    p.add_argument("--accum-steps", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--sync-bn", action="store_true", help="convert BatchNorm to SyncBatchNorm")
    p.add_argument("--eval-every", type=int, default=1, help="run validation every N epochs")
    return p.parse_args()


# -------------------------
# Distributed setup helpers
# -------------------------
def setup_distributed(backend: str = "nccl"):
    """Initialize process group and set CUDA device from LOCAL_RANK set by torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Fallback to single-process (non-distributed) mode
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    return local_rank, rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def print_main(*args, **kwargs):
    """Print only on rank 0"""
    if is_main_process():
        print(*args, **kwargs)


# -------------------------
# Checkpoint utilities
# -------------------------
def save_checkpoint(state: dict, save_dir: str, filename: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    full = os.path.join(save_dir, filename)
    if is_main_process():
        torch.save(state, full)
        print_main(f"Saved checkpoint: {full}")


def load_checkpoint(path: str, device: torch.device):
    if path is None or not os.path.isfile(path):
        return None
    map_location = {"cuda:%d" % 0: f"cuda:{device.index}"} if device.type == "cuda" else "cpu"
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


def strip_module_prefix(state_dict):
    """When loading into non-DDP model while checkpoint saved from DDP (module.*), strip keys."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state


# -------------------------
# Data & model builders
# -------------------------
def build_dataloader(batch_size, nworkers, world_size):
    transform = transforms.Compose([transforms.ToTensor()])
    # placeholder: use FakeData; replace with real dataset
    dataset = datasets.FakeData(transform=transform, size=10000, image_size=(3, 224, 224), num_classes=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=nworkers, pin_memory=True)
    return loader, sampler


def build_model(device: torch.device, sync_bn: bool = False):
    model = models.resnet18(pretrained=False)
    model.to(device)
    if sync_bn and dist.is_initialized():
        # convert to sync batchnorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


# -------------------------
# Training & validation
# -------------------------
def validate(model, device, loader):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    # aggregate across processes
    if dist.is_initialized():
        tensor = torch.tensor([loss_sum, correct, total], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        loss_sum, correct, total = tensor.tolist()
    avg_loss = loss_sum / total
    acc = correct / total
    return avg_loss, acc


def train_loop(args):
    local_rank, rank, world_size = setup_distributed(args.backend)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print_main(f"Starting training on device {device}, world_size={world_size}")

    # Build dataloaders
    train_loader, train_sampler = build_dataloader(args.batch_size, args.nworkers, world_size)
    val_loader, val_sampler = build_dataloader(args.batch_size, args.nworkers, world_size)  # placeholder, use proper val set

    # Build model, wrap with DDP
    model = build_model(device, sync_bn=args.sync_bn)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None)

    # Optimizer & scheduler
    # If using DDP, model.parameters() are fine
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "reduce_on_plateau":
        # Note: ReduceLROnPlateau expects the metric to decide step
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.gamma, patience=2)
    else:
        scheduler = None

    scaler = GradScaler()

    start_epoch = 0
    best_acc = 0.0

    # Resume checkpoint if provided
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        if ckpt is not None:
            # load model state (handle module. prefix)
            state_dict = ckpt.get("model_state", ckpt.get("state_dict", None))
            if state_dict is not None:
                state_dict = strip_module_prefix(state_dict)
                # If model is DDP-wrapped, state goes into model.module
                target = model.module if isinstance(model, DDP) else model
                target.load_state_dict(state_dict)
                print_main(f"Loaded model weights from {args.resume}")
            if "optim_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optim_state"])
                    print_main("Loaded optimizer state.")
                except Exception as e:
                    print_main("Warning: failed to load optimizer state:", e)
            if "scaler_state" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler_state"])
                    print_main("Loaded AMP scaler state.")
                except Exception as e:
                    print_main("Warning: failed to load scaler state:", e)
            if "scheduler_state" in ckpt and scheduler is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                    print_main("Loaded scheduler state.")
                except Exception as e:
                    print_main("Warning: failed to load scheduler state:", e)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_acc = ckpt.get("best_acc", best_acc)
            print_main(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    criterion = torch.nn.CrossEntropyLoss().to(device)

    total_steps_per_epoch = len(train_loader)
    accumulation = max(1, args.accum_steps)
    print_main(f"Accumulation steps: {accumulation}, steps_per_epoch={total_steps_per_epoch}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        optimizer.zero_grad()

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast():
                out = model(x)
                loss = criterion(out, y) / accumulation

            scaler.scale(loss).backward()

            if (step + 1) % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Logging per-iteration only on main process (every 100 iters)
            if is_main_process() and (step % 100 == 0):
                # loss.item() corresponds to the scaled per-batch loss (re-multiply accumulation)
                print_main(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{total_steps_per_epoch}] Loss {loss.item() * accumulation:.4f}")

        # Epoch finished
        epoch_time = time.time() - epoch_start

        # Run validation (optionally every N epochs)
        val_loss, val_acc = validate(model, device, val_loader) if (epoch % args.eval_every == 0) else (None, None)

        # Scheduler step
        if args.scheduler == "reduce_on_plateau" and scheduler is not None:
            # scheduler expects metric: use val_acc (higher is better)
            if val_acc is not None:
                scheduler.step(val_acc)
        elif scheduler is not None:
            scheduler.step()

        # Save checkpoint
        # Prepare state dict from the wrapped model
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        ckpt = {
            "epoch": epoch,
            "model_state": state_dict,
            "optim_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_acc": best_acc,
        }
        # Save periodic checkpoint
        if (epoch % args.save_every == 0) and is_main_process():
            save_checkpoint(ckpt, args.save_dir, f"checkpoint_epoch_{epoch}.pt")

        # Save best by validation accuracy
        if val_acc is not None and val_acc > best_acc:
            best_acc = float(val_acc)
            if is_main_process():
                save_checkpoint(ckpt, args.save_dir, "best_checkpoint.pt")
                print_main(f"New best acc: {best_acc:.4f}")

        print_main(f"Epoch {epoch} finished in {epoch_time:.1f}s. val_loss={val_loss} val_acc={val_acc} lr={optimizer.param_groups[0]['lr']}")
    # end epochs

    cleanup_distributed()
    print_main("Training finished.")


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
