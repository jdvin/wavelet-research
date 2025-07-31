import argparse
import os
from contextlib import nullcontext
from functools import partial
import math
import tempfile

from loguru import logger
import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as torch_mp
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from utils.data_utils import (
    EEGMMISplit,
    eeg_mmi_collate_fn,
    get_eeg_mmi_dataset,
    get_libri_brain_speech_dataset,
    get_nth_mask,
    libri_speech_brain_collate_fn,
)
from utils.electrode_utils import (
    Region,
    get_region_mask,
)

from pytorch_memlab import MemReporter

from utils.train_utils import (
    run_eval,
    setup,
    cleanup,
    register_cleanup_handlers,
    configure_optimizers,
    load_yaml,
    log_model_details,
    get_microbatch,
    get_dataloaders,
    get_validation_step_indexes,
    get_dataloader_iterator,
    TrainingConfig,
    upload_to_s3,
)

from utils.metrics import (
    MetricManager,
)

from src.montagenet import MontageNetConfig, MontageNet, TaskConfig


def main(
    rank: int,
    world_size: int,
    training_config_path: str,
    model_config_path: str,
    run_project: str,
    run_group: str,
    run_name: str,
    eval_first: bool,
    test_run: bool,
    device: str,
    checkpoints: bool,
    reset_data_cache: bool,
):
    cfg = TrainingConfig(
        **load_yaml(training_config_path),
        training_config_path=training_config_path,
        model_config_path=model_config_path,
        world_size=world_size,
        run_project=run_project,
        run_name=run_name,
        run_group=run_group,
        eval_first=eval_first,
        device=device,
        checkpoints=checkpoints,
    )
    grad_accum_steps = cfg.batch_size // (cfg.train_micro_batch_size * cfg.world_size)
    model_config = MontageNetConfig(
        **load_yaml(model_config_path),
        tasks=[
            TaskConfig(key="speech", n_classes=2),
            TaskConfig(key="speech_smooth", n_classes=2),
        ],
    )

    setup(
        rank=rank,
        world_size=world_size,
        logger=logger,
        run_project=run_project,
        run_group=run_group,
        run_name=run_name,
        checkpoints=checkpoints,
        training_config=cfg,
        model_config=model_config,
    )

    # Register cleanup handlers for graceful shutdown
    register_cleanup_handlers(world_size)
    is_main_process = rank == 0
    logger.info("Creating model instance.")
    # Create model.
    model: MontageNet = MontageNet(model_config, rank, world_size)

    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[cfg.dtype]

    model.to(rank, dtype=torch_dtype)
    # model = torch.compile(model)  # type: ignore
    assert len({param.device for param in model.parameters()}) == 1
    log_model_details(model)
    # reporter = MemReporter(model)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)  # type: ignore
    logger.info(f"Loading dataset from {cfg.dataset_path}.")
    # The first rank goes ahead to create the dataset if it does not already exist, before the other ranks then load it.
    # This is probably quite a strange pattern, but it is the simplest way to implement this behaviour.
    # TODO: Distributed dataset creation.
    # splits = {
    #     "train": EEGMMISplit(
    #         name="train",
    #         subjects=list(range(1, 100)),
    #         sessions=[1, 2],
    #     ),
    #     "eyes_val": EEGMMISplit(
    #         name="val",
    #         subjects=list(range(100, 110)),
    #         sessions=[1, 2],
    #     ),
    # }
    train_ds_getter = partial(
        get_libri_brain_speech_dataset,
        output_path="data/libri_brain_speech",
        partition="train",
        stride=100,
        oversample_silence_jitter=35,
    )
    val_ds_getter = partial(
        get_libri_brain_speech_dataset,
        output_path="data/libri_brain_speech",
        partition="validation",
        oversample_silence_jitter=70,
    )
    if is_main_process:
        # ds = get_eeg_mmi_dataset(
        #     source_base_path=cfg.dataset_path,
        #     output_path="data/eeg_mmi",
        #     splits=splits,
        #     labels_map={"base_e_open": 0, "base_e_clos": 1},
        #     tasks_map={"m_eyes_open": 0, "m_eyes_clos": 0},
        #     reset_cache=reset_data_cache,
        # )
        ds = {
            "train": train_ds_getter(),
            "speech_val": val_ds_getter(),
        }

        if world_size > 1:
            dist.barrier()
    else:
        dist.barrier()
        # ds = get_eeg_mmi_dataset(
        #     source_base_path=cfg.dataset_path,
        #     output_path="data/eeg_mmi",
        #     splits=splits,
        #     labels_map={"e_open": 0, "e_clos": 1},
        #     tasks_map={"m_eyes_open": 0, "m_eyes_clos": 0},
        #     reset_cache=reset_data_cache,
        # )
        ds = {
            "train": train_ds_getter(),
            "speech_val": val_ds_getter(),
        }

    logger.info("Creating data loaders.")
    # fronto_occipital_electrodes = (
    #     torch.tensor(
    #         get_region_mask(
    #             model.module.data_config.channel_positions.numpy(),
    #             [Region.FRONTAL, Region.OCCIPITAL],
    #         )
    #     )
    #     .unsqueeze(0)
    #     .unsqueeze(-1)
    # )
    # temporo_parietal_electrodes = (
    #     torch.tensor(
    #         get_region_mask(
    #             model.module.data_config.channel_positions.numpy(),
    #             [Region.PARIETAL, Region.TEMPORAL],
    #         )
    #     )
    #     .unsqueeze(0)
    #     .unsqueeze(-1)
    # )
    # train_mask = get_nth_mask(model.module.data_config.max_channels, 2, 1)
    # val_mask = get_nth_mask(model.module.data_config.max_channels, 2, 2)

    # train_collate_fn = get_eeg_mmi_collate_fn(mask=temporo_parietal_electrodes)
    # val_collate_fn = get_eeg_mmi_collate_fn(mask=fronto_occipital_electrodes)
    train_collate_fn = libri_speech_brain_collate_fn
    val_collate_fn = libri_speech_brain_collate_fn

    # Create data loaders.
    (
        train_dataloader,
        train_sampler,
        val_dataloaders,
        val_samplers,
    ) = get_dataloaders(
        ds,  # type: ignore
        cfg.train_micro_batch_size,
        cfg.val_micro_batch_size,
        rank,
        world_size,
        train_collate_fn,
        val_collate_fn,
    )
    # Steps per epoch is the number of batches in the training set.
    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    validation_step_indexes = get_validation_step_indexes(
        cfg.validation_interval, steps_per_epoch
    )

    device_type = "cuda" if "cuda" in device else "cpu"
    scaler_context = (
        nullcontext()
        if cfg.dtype == "fp32"
        else autocast(device_type=device_type, dtype=torch_dtype)
    )
    scaler = GradScaler(device=device_type, enabled=cfg.dtype == "fp16")

    logger.info("Creating optimizer.")

    optim, lr_scheduler = configure_optimizers(
        model.module.named_parameters,
        total_steps=steps_per_epoch * cfg.num_epochs,
        max_lr=cfg.max_lr,
        lr_schedules=cfg.lr_schedules,
        weight_decay=cfg.weight_decay,
    )
    metrics = MetricManager(
        device=rank,
        world_size=world_size,
        is_main_process=is_main_process,
        log_interval=cfg.log_interval,
        batch_size=cfg.batch_size,
        validation_dataset_keys=list(val_dataloaders.keys()),
    )

    metrics.lr.update(lr_scheduler.get_last_lr()[0])
    logger.info("Spinning Dataloader.")
    train_dataloader_iterator = get_dataloader_iterator(
        train_dataloader, train_sampler, metrics.epoch.value  # type: ignore
    )
    micro_batch = get_microbatch(train_dataloader_iterator, rank, torch_dtype)
    logger.info("Beginning Training.")
    train_pbar = tqdm(
        total=steps_per_epoch,
        desc=f"Epoch {metrics.epoch.value}/{cfg.num_epochs}.",
        leave=False,
        disable=rank not in {0, "cuda:0", "cuda"},
    )
    # logger.debug("====Post Init====")
    # reporter.report(device=rank)
    if eval_first:
        run_eval(
            model=model.module,
            val_dataloaders=val_dataloaders,
            val_samplers=val_samplers,
            metrics=metrics,
            device=rank,
            dtype=torch_dtype,
        )
    metrics.start_step_timer()
    while True:
        is_accumulating = (
            metrics.microstep.value % grad_accum_steps != 0
            and metrics.epoch_microstep.value != len(train_dataloader)
        )
        # Forward and backward pass.
        # Do not sync gradients whilst accumulating.
        ddp_context = (
            model.no_sync() if world_size > 1 and is_accumulating else nullcontext()
        )
        # torch.cuda.memory._record_memory_history()
        with ddp_context:
            with scaler_context:
                loss, logits, labels = model(micro_batch)
                # logger.debug("====Forward Pass====")
                # reporter.report()
                loss = loss / grad_accum_steps
            metrics.train_loss.update(loss.item())
            # Get the next batch straight away without blocking whilst we compute the backward pass,
            # unless we are at the end of the epoch.
            if metrics.epoch_microstep.value < len(train_dataloader) - 1:
                micro_batch = get_microbatch(
                    train_dataloader_iterator, rank, torch_dtype
                )
            scaler.scale(loss).backward()  # type: ignore

        # torch.cuda.memory._dump_snapshot("new_snapshot.pickle")
        # break
        # If we are still accumulating gradients then skip gradient application and logging.
        if is_accumulating:
            metrics.microstep.update(1)
            metrics.epoch_microstep.update(
                (metrics.microstep.value, len(train_dataloader))
            )
            continue
        # list_grad_mismatches(model)
        if cfg.grad_clip > 0:
            scaler.unscale_(optim)
            metrics.train_gradnorm.update(
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            )
        # Gradient application and logging.
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        del loss, logits, labels
        lr_scheduler.step()
        metrics.end_step_timer()
        train_pbar.update()
        if metrics.epoch_step.value in validation_step_indexes or test_run:
            torch.cuda.empty_cache()
            run_eval(
                model=model.module,
                val_dataloaders=val_dataloaders,
                val_samplers=val_samplers,
                metrics=metrics,
                device=rank,
                dtype=torch_dtype,
            )
        metrics.log()
        if metrics.epoch_microstep.value == len(train_dataloader) or test_run:
            if is_main_process and checkpoints:
                # Create temporary file for checkpoint
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pt"
                ) as tmp_file:
                    tmp_file_path = tmp_file.name

                # Save model state to temporary file
                torch.save(model.module.state_dict(), tmp_file_path)

                # Upload to S3
                s3_bucket = "neural-decode-models"
                s3_key = f"{run_name}/{cfg.run_project}_{cfg.run_group}_{cfg.run_name}_ep{metrics.epoch.value}.pt"
                upload_to_s3(tmp_file_path, s3_bucket, s3_key)

            metrics.epoch.update(1)
            train_pbar = tqdm(
                total=steps_per_epoch,
                desc=f"Epoch: {metrics.epoch.value}/{cfg.num_epochs}.",
                leave=False,
                disable=not is_main_process,
            )
            train_dataloader_iterator = get_dataloader_iterator(
                train_dataloader, train_sampler, metrics.epoch.value
            )
            # Get the first microbatch of the new epoch.
            micro_batch = get_microbatch(train_dataloader_iterator, rank, torch_dtype)

        if metrics.epoch.value == cfg.num_epochs + 1:
            logger.info("Training complete.")
            break
        metrics.step_iterators(
            cfg.batch_size, steps_per_epoch, len(train_dataloader), lr_scheduler
        )
        metrics.start_step_timer()
    cleanup(world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--run-project", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--run-group", type=str, required=True)
    parser.add_argument("--eval-first", action="store_true")
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--device", type=str)
    parser.add_argument("--training-config-path", type=str, default=None)
    parser.add_argument("--model-config-path", type=str, default=None)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--checkpoints", action="store_true", default=False)
    parser.add_argument("--reset-data-cache", action="store_true", default=False)

    args = parser.parse_args()

    if args.world_size == 1:
        main(rank=0, **vars(args))
    else:
        try:
            torch_mp.spawn(  # type: ignore
                main,
                args=(
                    args.world_size,
                    args.training_config_path,
                    args.model_config_path,
                    args.run_project,
                    args.run_group,
                    args.run_name,
                    args.eval_first,
                    args.test_run,
                    args.device,
                    args.checkpoints,
                    args.reset_data_cache,
                ),
                nprocs=args.world_size,
                join=True,
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            raise
