import argparse
import json

from loguru import logger
import torch
from torch.package import package_exporter

from src.montagenet import MontageNet, MontageNetConfig
from utils.train_utils import TrainingConfig, load_yaml, recursive_to_serializable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config-path", type=str, required=True)
    parser.add_argument("--model-config-path", type=str, required=True)
    parser.add_argument("--state-dict-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    logger.info("Loading Config.")
    training_config = TrainingConfig(
        **load_yaml(args.train_config_path),
        training_config_path=args.train_config_path,
        model_config_path="",
        world_size=1,
        run_project="",
        run_name="",
        run_group="",
        eval_first=False,
        device="",
        checkpoints=False,
    )
    model_config = MontageNetConfig(**load_yaml(args.model_config_path))
    logger.info("Loading Model.")
    # Ensure the state dict is compatible with the model config.
    model = MontageNet(model_config, rank=0, world_size=1)
    model.load_state_dict(torch.load(args.state_dict_path, map_location="cpu"))
    logger.info("Exporting Model.")
    with package_exporter.PackageExporter(f"{args.output_path}.pkg") as e:
        # Put YOUR code inside, treat 3rd-party libs as extern
        e.intern("src.*")
        e.intern("src.components.*")
        e.extern("torch.**")
        e.extern("einops.**")
        # Save target module.
        e.save_module("src.montagenet")
        # Save the state.
        e.save_pickle("assets", "state.pkl", model.state_dict())
        e._write(
            "config" "task_config.json",
            json.dumps({
                "ds_labels_map": training_config.ds_labels_map,
                "ds_tasks_map": training_config.ds_tasks_map,
            }),
        )
        e.save_pickle("config", "model_config.pkl", model_config)
    logger.info("Done.")


if __name__ == "__main__":
    main()
