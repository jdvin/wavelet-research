from collections.abc import Iterable
from dataclasses import asdict, dataclass
from collections import defaultdict
from functools import partial
import math
from typing import Any, Callable, Iterator
import os
import random
from datetime import timedelta
import atexit
import signal
from enum import Enum

import boto3

# from muon import MuonWithAuxAdam
from loguru import logger
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import wandb
import yaml
from typing import Callable, List, Optional

# from distributed_shampoo import AdamGraftingConfig, DistributedShampoo
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    SequentialLR,
    ConstantLR,
    _LRScheduler,
)


from src.montagenet import MontageNet, MontageNetConfig
from utils.metrics import MetricManager


from contextlib import suppress


class Optimizer(Enum):
    ADAMW = "adamw"
    MUON = "muon"


def list_grad_mismatches(model, atol=1e-6):
    """
    Waits for all DDP gradient all-reduces to finish, then checks each
    parameter.  Prints a line for every .grad buffer that still differs
    across ranks (RMS error > atol).
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # 1.  Host-side sync: make sure all earlier collectives are queued
    # ------------------------------------------------------------------
    dist.barrier()  # default stream

    # ------------------------------------------------------------------
    # 2.  Device-side sync for *every* NCCL stream in this pg.
    #     A dummy all-reduce forces stream-ordering across all NCCL streams.
    # ------------------------------------------------------------------
    dummy = torch.zeros(1, device=device)
    dist.all_reduce(dummy, op=dist.ReduceOp.SUM)  # cannot launch until
    # all grad all-reduces
    # have completed

    torch.cuda.synchronize()  # wait for GPU work

    # ------------------------------------------------------------------
    # 3.  Compare each parameter
    # ------------------------------------------------------------------
    mismatches = []

    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            # Parameter unused on this rank – ignore for mismatch report.
            continue

        # Clone local grad and compute the global average
        g_local = grad.detach()
        g_avg = g_local.clone()
        dist.all_reduce(g_avg, op=dist.ReduceOp.AVG)

        # Root-mean-square difference
        sq_err = (g_local - g_avg).pow(2).sum()
        dist.all_reduce(sq_err, op=dist.ReduceOp.AVG)
        rms_err = (sq_err / g_local.numel()).sqrt()

        if rms_err > atol:
            mismatches.append(
                (
                    name,
                    tuple(param.shape),
                    grad.dtype,
                    g_local.norm().item(),
                    rms_err.item(),
                )
            )

    # ------------------------------------------------------------------
    # 4.  Print results (only if there are mismatches)
    # ------------------------------------------------------------------
    if mismatches:
        header = f"\n[rank {rank}] Gradients that differ across ranks:"
        print(header)
        for n, shape, dtype, norm2, rms in mismatches:
            print(
                f"   ✗ {n:<35}  shape={shape}  "
                f"dtype={dtype}  ||grad||₂={norm2:8.3e}  "
                f"rms diff={rms:8.3e}"
            )
        print()

    # Optional hard assertion: trip only once (rank 0) to avoid log spam
    if mismatches:
        raise RuntimeError("Some gradients still differ across ranks.")


@dataclass
class TrainingConfig:
    """Configuration for training a model.

    Attributes:
        model_key: The key for the target model class in the `MODEL_REGISTRY`.
        dataset_path: The path to the dataset.
        num_epochs: The number of epochs to train the model for.
        batch_size: The total batch size. Must be divisible by `micro_batch_size`. Gradients are accumulated over `batch_size` / `micro_batch_size` steps.
        micro_batch_size: The number of examples inference is performed over per micro step.
        validation_interval: The interval at which validation is performed. Measured in fractions of an epoch. Rounded up to the nearest multiple of the number of steps in an epoch.
        log_interval: The interval at which metrics are logged. Measured in steps.
        max_lr: The maximum learning rate for the optimizer.
        weight_decay: The weight decay for the optimizer.
    """

    world_size: int
    run_name: str
    run_group: str
    run_project: str
    eval_first: bool
    device: str
    dtype: str
    training_config_path: str
    model_config_path: str
    checkpoints: bool
    ds_split_configs: list[dict[str, Any]]
    num_epochs: int
    batch_size: int
    train_micro_batch_size: int
    val_micro_batch_size: int
    validation_interval: float
    log_interval: float  # Measured in steps.
    max_lr: float
    lr_schedules: list[dict[str, Any]]
    weight_decay: float
    grad_clip: float

    def __post_init__(self):
        assert self.batch_size % self.train_micro_batch_size == 0
        assert self.train_micro_batch_size * self.world_size <= self.batch_size


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_microbatch(
    dataloader_iterator: Iterator,
    device: str | int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    micro_batch = next(dataloader_iterator)
    return {
        k: (
            v.pin_memory().to(
                device=device,
                dtype=dtype if torch.is_floating_point(v) else torch.long,
                non_blocking=True,
            )
            if isinstance(device, int)
            else v.to(
                device=device, dtype=dtype if torch.is_floating_point(v) else torch.long
            )
        )
        for k, v in micro_batch.items()
        if isinstance(v, torch.Tensor)
    }


def upload_to_s3(local_file_path: str, s3_bucket: str, s3_key: str) -> None:
    """
    Upload a file to S3.

    Args:
        local_file_path: Path to the local file to upload
        s3_bucket: S3 bucket name
        s3_key: S3 object key (path within the bucket)
    """
    try:
        # Create S3 client
        s3_client = boto3.client("s3")

        # Upload file to S3
        logger.info(f"Starting upload to s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(local_file_path, s3_bucket, s3_key)
        logger.info(f"Successfully uploaded checkpoint to s3://{s3_bucket}/{s3_key}")

        # Clean up local file after successful upload
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
            logger.info(f"Cleaned up local file: {local_file_path}")

    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        # Don't raise the exception to avoid interrupting training


def setup(
    rank: int,
    world_size: int,
    logger: Any,
    run_project: str,
    run_group: str,
    run_name: str,
    checkpoints: bool,
    training_config: TrainingConfig,
    model_config: MontageNetConfig,
):
    """Setup the environment for training."""
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    if rank != 0:
        # Suppress output from all ranks except rank 0.
        logger.remove()
    else:
        # Initialize checkpoints directory and wandb logging for the first rank.
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(
            project=run_project,
            group=run_group,
            name=run_name,
            config=dict(
                training_config=asdict(training_config),
                model_config=asdict(model_config),
            ),
        )
    if world_size > 1:
        torch.cuda.set_device(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        torch.distributed.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30),
        )


_cleanup_called = False


def cleanup(world_size: int):
    global _cleanup_called
    if _cleanup_called:
        return
    _cleanup_called = True

    if world_size > 1:
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass  # Ignore errors during cleanup


def register_cleanup_handlers(world_size: int):
    """Register cleanup handlers for graceful shutdown."""

    def cleanup_handler(*args):
        cleanup(world_size)
        os._exit(0)  # Force exit without running other atexit handlers

    # Only register signal handlers (remove atexit to avoid conflicts)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        try:
            signal.signal(sig, cleanup_handler)
        except (OSError, ValueError):
            pass


def format_number(number: int) -> str:
    """Format a number as a string with K, M, or B suffixes."""
    if number < 1_000:
        return str(number)
    if number < 1_000_000:
        return f"{number / 1_000:.2f}K"
    if number < 1_000_000_000:
        return f"{number / 1_000_000:.2f}M"
    return f"{number / 1_000_000_000:.2f}B"


def count_params(model: torch.nn.Module) -> dict[str, str]:
    """Count the number of trainable and untrainable parameters in a model."""
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrained = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "Trained": format_number(trained),
        "Untrained": format_number(untrained),
        "Total": format_number(trained + untrained),
    }


def log_model_details(model: torch.nn.Module) -> None:
    """Log the architecture and parameter counts of a model."""
    logger.info(f"Architecture:\n{model}")
    param_counts = count_params(model)
    logger.info(
        "| "
        + " | ".join(
            [f"{key} Parameters: {value}" for key, value in param_counts.items()]
        )
    )


def get_validation_step_indexes(
    validation_interval: float, steps_per_epoch: int
) -> set[int]:
    """Get the indexes of the steps at which validation should be performed.

    Validation interval is measured in fractions of an epoch."""
    assert 1 >= validation_interval > 0
    steps_per_validation = math.ceil(validation_interval * steps_per_epoch)
    validation_step_indexes = set(
        range(steps_per_validation, steps_per_epoch + 1, steps_per_validation)
    )
    if steps_per_validation % steps_per_epoch != 0:
        validation_step_indexes.add(steps_per_epoch)
    return validation_step_indexes


def get_adam_param_groups(
    named_parameters, weight_decay: float = 1e-1
) -> list[dict[str, str]]:
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in named_parameters.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in named_parameters.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return optim_groups


def get_muon_param_groups(
    named_parameters: dict[str, torch.Tensor],
    lrs: dict[str, float],
    weight_decay: float = 1e-1,
) -> list[dict]:
    muon_params = [
        p for n, p in named_parameters.items() if p.ndim >= 2 and "encoder.blocks" in n
    ]
    non_muon_params = [
        p
        for n, p in named_parameters.items()
        if p.ndim >= 2 and "encoder.blocks" not in n
    ]
    biases = [p for n, p in named_parameters.items() if p.ndim < 2]
    return [
        {
            "params": muon_params,
            "weight_decay": weight_decay,
            "lr": lrs["muon"],
            "use_muon": True,
        },
        {
            "params": non_muon_params,
            "weight_decay": weight_decay,
            "lr": lrs["adamw"],
            "use_muon": False,
        },
        {
            "params": biases,
            "weight_decay": 0.0,
            "lr": lrs["adamw"],
            "use_muon": False,
        },
    ]


class CarryOverScheduler(_LRScheduler):
    """
    Chain LR schedulers so every new stage *inherits* the optimiser's CURRENT
    learning rate(s) – no jumps, even with ConstantLR or other schedulers
    that look at group['initial_lr'].
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        factories: List[Callable[[torch.optim.Optimizer], _LRScheduler]],
        milestones: List[int],
        last_epoch: int = -1,
    ):
        if len(factories) < 1:
            raise ValueError("Need at least one factory.")
        if len(milestones) != len(factories) - 1:
            raise ValueError("len(milestones) must be len(factories)-1.")

        self.optimizer = optimizer
        self.factories = factories
        self.milestones = list(milestones)
        self.stage_idx = 0
        self.stage_start = 0
        self.current = self._make_scheduler(0)  # stage-0
        super().__init__(optimizer, last_epoch)

    # ------------------------------------------------------------------ #
    # helper ------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _make_scheduler(self, idx: int) -> _LRScheduler:
        """Build scheduler `idx`, patch its base_lrs to CURRENT LRs."""
        # 1) snapshot current LRs
        cur_lrs = [g["lr"] for g in self.optimizer.param_groups]

        # 2) build the scheduler (this will internally call step(0) once)
        sched = self.factories[idx](self.optimizer)

        # 3) restore the *current* LR and make the new scheduler see it
        for g, lr in zip(self.optimizer.param_groups, cur_lrs):
            g["lr"] = lr
            g["initial_lr"] = lr  # so future schedulers inherit
        sched.base_lrs = cur_lrs

        return sched

    # ------------------------------------------------------------------ #
    # required API ------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def get_last_lr(self):
        return self.current.get_last_lr()

    def step(self, epoch: Optional[int] = None) -> None:
        """Advance one optimiser **step**."""
        next_global = self.last_epoch + 1 if epoch is None else epoch
        local_step = next_global - self.stage_start

        # 1. run the active scheduler for *this* step
        self.current.step(local_step)

        # 2. if we just finished a milestone, prepare the next scheduler
        if (
            self.stage_idx < len(self.milestones)
            and next_global >= self.milestones[self.stage_idx]
        ):
            self.stage_idx += 1
            self.stage_start = self.milestones[self.stage_idx - 1]
            self.current = self._make_scheduler(self.stage_idx)

        self.last_epoch = next_global

    # ------------------------------------------------------------------ #
    # state-dict helpers ------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def state_dict(self):
        base = super().state_dict()
        base.update(
            {
                "stage_idx": self.stage_idx,
                "stage_start": self.stage_start,
                "current_state": self.current.state_dict(),
            }
        )
        return base

    def load_state_dict(self, state_dict):
        self.stage_idx = state_dict.pop("stage_idx")
        self.stage_start = state_dict.pop("stage_start")
        current_state = state_dict.pop("current_state")
        super().load_state_dict(state_dict)
        self.current = self._make_scheduler(self.stage_idx)
        self.current.load_state_dict(current_state)


class LRSchedule(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    CONSTANT = "constant"
    EXPONENTIAL = "exponential"


def configure_optimizers(
    named_parameters,
    total_steps: int,
    max_lr: float,
    lr_schedules: list[dict[str, Any]],
    weight_decay: float,
    use_optimizer: Optimizer = Optimizer.ADAMW,
):
    named_parameters = {n: p for n, p in named_parameters() if p.requires_grad}

    if use_optimizer == Optimizer.MUON:
        param_groups = get_muon_param_groups(
            named_parameters, {"muon": max_lr * 66, "adamw": max_lr}, weight_decay
        )
        # optimizer = MuonWithAuxAdam(param_groups)
    elif use_optimizer == Optimizer.ADAMW:
        param_groups = get_adam_param_groups(named_parameters, weight_decay)
        optimizer = AdamW(
            param_groups,
            lr=max_lr,
        )
    else:
        raise NotImplementedError(f"Unknown optimizer: {use_optimizer}")

    schedulers = []
    milestones = []
    for i, lr_schedule in enumerate(lr_schedules):
        current_milestone = total_steps * lr_schedule["start"]
        next_milestone = (
            total_steps * lr_schedules[i + 1]["start"]
            if i < len(lr_schedules) - 1
            else total_steps
        )
        milestone_iters = next_milestone - current_milestone
        if i != 0:
            milestones.append(current_milestone)
        schedule_type = LRSchedule(lr_schedule["type"])
        if schedule_type == LRSchedule.LINEAR:
            schedulers.append(
                partial(
                    LinearLR,
                    **lr_schedule["kargs"],
                    total_iters=milestone_iters,
                )
            )
        elif schedule_type == LRSchedule.COSINE:
            schedulers.append(
                partial(
                    CosineAnnealingLR,
                    **lr_schedule["kargs"],
                    T_max=milestone_iters,
                )
            )
        elif schedule_type == LRSchedule.CONSTANT:
            schedulers.append(
                partial(
                    ConstantLR,
                    **lr_schedule["kargs"],
                    total_iters=milestone_iters,
                )
            )
        elif schedule_type == LRSchedule.EXPONENTIAL:
            schedulers.append(
                partial(
                    ExponentialLR,
                    **lr_schedule["kargs"],
                )
            )
        else:
            raise NotImplementedError(f"Unknown lr schedule: {lr_schedule['type']}")
    scheduler = CarryOverScheduler(
        optimizer,
        schedulers,
        milestones=milestones,
    )

    return optimizer, scheduler


def worker_init_fn(worker_id: int, rank: int = 0) -> None:
    """Initialize DataLoader worker with deterministic seeds."""
    worker_seed = 42 + rank * 1000 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(
    dataset: dict[str, Dataset],
    train_microbatch_size: int,
    val_microbatch_size: int,
    rank: int,
    world_size: int,
    train_collate_fn: Callable[[list], dict],
    val_collate_fn: Callable[[list], dict],
) -> tuple[
    DataLoader,
    Sampler | None,
    dict[str, DataLoader],
    dict[str, Sampler] | dict[str, DistributedSampler] | dict[str, None],
]:
    if world_size > 1:
        train_sampler = DistributedSampler(
            dataset["train"], num_replicas=world_size, rank=rank, shuffle=True
        )
        val_samplers = {
            key: DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=False
            )
            for key, ds in dataset.items()
            if key.endswith("_val")
        }
    else:
        train_sampler, val_samplers = None, {
            key: None for key in dataset.keys() if key.endswith("_val")
        }
    from functools import partial

    train_dataloader = DataLoader(
        dataset["train"],  # type: ignore
        batch_size=train_microbatch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=train_collate_fn,
        num_workers=2,
        worker_init_fn=partial(worker_init_fn, rank=rank),
    )
    val_dataloaders = {
        key: DataLoader(
            ds,  # type: ignore
            batch_size=val_microbatch_size,
            shuffle=val_samplers[key] is None,
            sampler=val_samplers[key],
            collate_fn=val_collate_fn,
            num_workers=2,
            worker_init_fn=partial(worker_init_fn, rank=rank),
        )
        for key, ds in dataset.items()
        if key.endswith("_val")
    }
    return train_dataloader, train_sampler, val_dataloaders, val_samplers


def get_dataloader_iterator(
    dataloader: DataLoader, sampler: Sampler | None, epoch: int
) -> Iterator:
    """Get an iterator for a dataloader, with the sampler set to the correct epoch."""
    if isinstance(sampler, DistributedSampler):
        # Required to ensure that the order is different each epoch.
        sampler.set_epoch(epoch)
    return iter(dataloader)


@torch.no_grad()
def run_eval(
    model: MontageNet,
    val_dataloaders: dict[str, DataLoader],
    val_samplers: dict[str, DistributedSampler] | dict[str, Sampler] | dict[str, None],
    metrics: MetricManager,
    device: str | int,
    dtype: torch.dtype,
):
    """Run evaluation on the validation sets."""
    model.eval()
    for (dl_key, val_dataloader), (sampler_key, val_sampler) in zip(
        val_dataloaders.items(), val_samplers.items()
    ):
        assert dl_key == sampler_key
        val_pbar = tqdm(
            total=len(val_dataloader),
            desc=f"Running validation: {dl_key}.",
            leave=False,
            disable=device not in {0, "cuda:0", "cuda"},
        )
        val_dataloader_iterator = get_dataloader_iterator(
            val_dataloader, val_sampler, metrics.epoch.value
        )
        accum_loss = torch.tensor([0.0], device=device)
        for _ in range(len(val_dataloader)):
            micro_batch = get_microbatch(val_dataloader_iterator, device, dtype)
            loss, logits, labels = model(micro_batch)
            accum_loss += loss.item() / len(val_dataloader)

            val_pbar.update()
            out = {"logits": logits, "labels": labels}
            random_out = {
                "logits": torch.randn(logits.shape, device=device),
                "labels": labels,
            }
            metrics.val[dl_key]["accuracy"].update(out)
            # Update confusion matrix for per-class metrics
            metrics.val[dl_key]["confusion_matrix"].update(out)
            metrics.val[dl_key]["random_confusion_matrix"].update(random_out)

        # Log standard metrics
        metrics.val[dl_key]["accuracy"].log()
        # Log per-class metrics using the new method
        metrics.log_per_class_metrics(
            dl_key, metrics.val[dl_key]["confusion_matrix"].value
        )
        metrics.log_per_class_metrics(
            dl_key + "_random", metrics.val[dl_key]["random_confusion_matrix"].value
        )

        metrics.val[dl_key]["loss"].update(accum_loss)
        metrics.val[dl_key]["loss"].log()
        metrics.val[dl_key]["confusion_matrix"].log()
        metrics.val[dl_key]["random_confusion_matrix"].log()
        del accum_loss
    torch.cuda.empty_cache()
    model.train()
