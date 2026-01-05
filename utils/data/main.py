import os
from typing import Any

from loguru import logger

from src.montagenet import DataConfig, MontageNetConfig
from utils.electrode_utils import (
    EPOC14_CHANNEL_POSITIONS,
    INSIGHT5_CHANNEL_POSITIONS,
    LEMON_CHANNEL_POSITIONS,
    NEUROTECHS_CHANNEL_POSITIONS,
    PHYSIONET_64_CHANNEL_POSITIONS,
    RESTING_METHODS_CHANNEL_POSITIONS,
    create_mask,
)
from utils.torch_datasets import MappedLabelDataset, MultiMappedLabelDataset
from utils.train_utils import TrainingConfig, load_yaml

from .common import DataSource, DataSplit, Headset
from .eeg_mmi import EEGMMIDataSplit, extract_eeg_mmi_split
from .emotiv_alpha import (
    EmotivAlphaDataSplit,
    EmotivAlphaEpocDataSplit,
    EmotivAlphaInsightDataSplit,
    extract_emotiv_alpha_suppression_split,
)
from .lemon_rest import LemonRestingStateDataSplit, extract_lemon_resting_state
from .neurotechs import NeurotechsEyesDataSplit, extract_neurotechs_eyes_split
from .resting_methods import RestingEEGMethodsDataSplit, extract_resting_methods_split


def ds_split_factory(splits: list[dict[str, Any]]):
    out = []
    for split in splits:
        data_source = DataSource(split["data_source"])
        headset = Headset(split["headset"])
        if data_source == DataSource.EEG_MMI:
            out.append(EEGMMIDataSplit(**split))
        elif data_source == DataSource.EMOTIVE_ALPHA:
            if headset == Headset.EMOTIV_EPOC_14:
                out.append(EmotivAlphaEpocDataSplit(**split))
            elif headset == Headset.EMOTIV_INSIGHT_5:
                out.append(EmotivAlphaInsightDataSplit(**split))
            else:
                raise NotImplementedError(f"Unknown headset: {headset}")
        elif data_source == DataSource.LEMON_REST:
            out.append(LemonRestingStateDataSplit(**split))
        elif data_source == DataSource.NEUROTECHS:
            out.append(NeurotechsEyesDataSplit(**split))
        elif data_source == DataSource.RESTING_METHODS:
            out.append(RestingEEGMethodsDataSplit(**split))
        else:
            raise NotImplementedError(f"Unknown data source: {data_source}")
    return out


def get_multi_mapped_label_datasets(
    splits: list[DataSplit],
    tasks_map: dict[str, int],
    labels_map: dict[str, int],
    data_config: DataConfig,
    reset_cache: bool = False,
):
    ret = {}
    for split in splits:
        logger.info(f"Creating dataset for split: {split.split_name}")
        os.makedirs(split.output_path, exist_ok=True)

        electrode_positions = None
        if split.headset == Headset.PHYSIONET_64:
            electrode_positions = PHYSIONET_64_CHANNEL_POSITIONS
        elif split.headset == Headset.EMOTIV_INSIGHT_5:
            electrode_positions = INSIGHT5_CHANNEL_POSITIONS
        elif split.headset == Headset.EMOTIV_EPOC_14:
            electrode_positions = EPOC14_CHANNEL_POSITIONS
        elif split.headset == Headset.LEMON_61:
            electrode_positions = LEMON_CHANNEL_POSITIONS
        elif split.headset == Headset.UNICORN_HYBRID_BLACK_8:
            electrode_positions = NEUROTECHS_CHANNEL_POSITIONS
        elif split.headset == Headset.BRAIN_ACTICHAMP_31:
            electrode_positions = RESTING_METHODS_CHANNEL_POSITIONS
        else:
            raise NotImplementedError(f"Unknown headset: {split.headset}")
        assert electrode_positions is not None
        logger.info(f"Got electrode positions for headset: {split.headset}.")

        channel_mask = None
        if split.channel_mask_config is not None:
            logger.info(f"Creating channel mask: {split.channel_mask_config}")
            channel_mask = create_mask(
                electrode_positions.numpy(), split.channel_mask_config
            )

        if split.data_source == DataSource.EEG_MMI:
            logger.info("Extracting EEG MMI data.")
            assert isinstance(split, EEGMMIDataSplit)
            split_eeg, split_labels, split_metadata = extract_eeg_mmi_split(
                split.source_base_path,
                split,
                split.output_path,
                epoch_length_sec=split.epoch_length_sec,
                reset_cache=reset_cache,
            )
        elif split.data_source == DataSource.EMOTIVE_ALPHA:
            logger.info("Extracting Emotiv Alpha data.")
            assert isinstance(split, EmotivAlphaDataSplit)
            split_eeg, split_labels, split_metadata = (
                extract_emotiv_alpha_suppression_split(
                    split,
                    reset_cache=reset_cache,
                )
            )
        elif split.data_source == DataSource.LEMON_REST:
            logger.info("Extracting LEMON resting-state data.")
            assert isinstance(split, LemonRestingStateDataSplit)
            split_eeg, split_labels, split_metadata = extract_lemon_resting_state(
                split=split,
                ignore_cache=reset_cache,
            )
        elif split.data_source == DataSource.NEUROTECHS:
            logger.info("Extracting Neurotechs eyes-open/closed baseline data.")
            assert isinstance(split, NeurotechsEyesDataSplit)
            split_eeg, split_labels, split_metadata = extract_neurotechs_eyes_split(
                split=split,
                reset_cache=reset_cache,
            )
        elif split.data_source == DataSource.RESTING_METHODS:
            logger.info("Extracting resting EEG study methods data.")
            assert isinstance(split, RestingEEGMethodsDataSplit)
            split_eeg, split_labels, split_metadata = extract_resting_methods_split(
                split=split,
                reset_cache=reset_cache,
            )
        else:
            raise NotImplementedError(f"Unknown data source: {split.data_source}")

        dataset = MappedLabelDataset(
            split_eeg,
            split_labels,
            split_metadata,
            labels_map,
            tasks_map,
            electrode_positions,
            data_config,
            split.sampling_rate,
            channel_mask,
            participants=split.subjects,
        )

        ds = ret.get(split.split_name, None)
        if ds is None:
            ret[split.split_name] = MultiMappedLabelDataset([dataset])
        else:
            ds.append_dataset(dataset)
    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data processing standalone script.")
    parser.add_argument("--training-config-path", type=str, required=True)
    parser.add_argument("--model-config-path", type=str, required=True)

    args = parser.parse_args()
    cfg = TrainingConfig(
        **load_yaml(args.training_config_path),
        training_config_path=args.training_config_path,
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
    ds_splits = ds_split_factory(cfg.ds_split_configs)
    ds = get_multi_mapped_label_datasets(
        ds_splits,
        model_config.tasks_map,
        model_config.labels_map,
        model_config.data_config,
        reset_cache=True,
    )
