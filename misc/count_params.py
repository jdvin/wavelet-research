import argparse
import yaml

from src.montagenet import MontageNet, MontageNetConfig
from utils.train_utils import count_params


def load_config(path: str) -> MontageNetConfig:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return MontageNetConfig(**config)


def main(config_path: str) -> None:
    config = load_config(config_path)
    model = MontageNet(config, rank=0, world_size=1)
    param_counts = count_params(model)
    for key, value in param_counts.items():
        print(f"{key} Parameters: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count MontageNet parameters for a model config."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the MontageNet YAML config (e.g., config/eyes_montagenet_cfg.yaml).",
    )
    args = parser.parse_args()
    main(args.config_path)
