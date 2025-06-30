from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict
from functools import partial
from enum import Enum
from typing import Any, Callable, Mapping


import torch
from torch import Tensor, tensor
from torch.distributed import (
    ReduceOp,
    all_gather_object,
    all_reduce,
    all_gather_into_tensor,
)
import wandb
import time

THINGS_CONCEPTS_PATH = "data/things_concepts.csv"
# SYNONYM_MAP = {
#     object_word.lower().strip(): [
#         synonym.lower().replace("_", " ").strip()
#         for synonym in synonyms.split(",")
#         if synonym.lower().replace("_", " ").strip() != object_word.lower().strip()
#     ]
#     for object_word, synonyms in pd.read_csv(THINGS_CONCEPTS_PATH)[
#         ["Word", "WordNet Synonyms"]
#     ].values
# }


class MetricLogType(Enum):
    SCALAR = "scalar"
    PLOT = "plot"
    TABLE = "table"


class MetricResetRule(Enum):
    ON_LOG = "on_log"
    ON_EPOCH = "on_epoch"
    MANUAL = "manual"


class MetricType(Enum):
    STATE = "state"
    GENERATION = "generation"


Loggable = int | float
Plottable = list[tuple[float, float]]
MetricState = int | float | Tensor | list | dict | None
# TODO: Should this be dataclass with explicit possible values?
InferenceArtefacts = (
    Mapping[str, Tensor]
    | Mapping[str, float | int | list[str]]
    | tuple[Tensor | float | int, ...]
    | list[str]
    | Tensor
    | float
    | int
)


def return_first_value(x: InferenceArtefacts) -> MetricState:
    if isinstance(x, dict):
        list(x.values())[0]
    elif isinstance(x, tuple):
        return x[0]
    else:
        assert isinstance(x, (Tensor, float, int))
        return x


def divide(state: MetricState) -> Tensor:
    assert isinstance(state, Tensor)
    return state[0] / state[1]


add = lambda x, y: x + y
identity = lambda x: x
replace = lambda x, y: y


def all_reduce_mean(t: Tensor | list, ws: int) -> Tensor:
    assert isinstance(t, Tensor)
    all_reduce(t, op=ReduceOp.AVG)
    return t


def all_reduce_sum(t: Tensor | list, ws: int) -> Tensor:
    assert isinstance(t, Tensor)
    all_reduce(t, op=ReduceOp.SUM)
    return t


def all_gather_concat(t: Tensor | list, ws: int) -> Tensor:
    assert isinstance(t, Tensor)
    # Create a tensor to hold the concatenated values.
    out = torch.zeros(t.size(0) * ws, *t.size()[1:], dtype=t.dtype, device=t.device)
    all_gather_into_tensor(out, t.contiguous())
    return out


def all_gather_append(t: Tensor | list, ws: int) -> list:
    assert isinstance(t, list)
    out = [None] * ws
    all_gather_object(out, t)
    return out


def distributed_identity(t: Tensor | list, ws: int) -> Tensor | list:
    return t


def row_concat(t1: MetricState, t2: MetricState) -> Tensor:
    assert isinstance(t1, Tensor) and isinstance(t2, Tensor)
    return torch.concat([t1, t2], dim=0)


def position_in_cycle(inference_artefacts: InferenceArtefacts) -> int:
    """For a given intra-epoch cycle (e.g., step or microstep), given the absolute number of the current cycle (`inference_artefacts[0]`; indexed from 1)
    and the total number of cycles in an epoch (`inference_artefacts[1]`; index from 1), return the number of the current cycle relative to the current epoch.
    """
    assert isinstance(inference_artefacts, tuple)
    out = (inference_artefacts[0] - 1) % inference_artefacts[1] + 1
    assert isinstance(out, int)
    return out


def flatten_ranks(root: list[dict[str, list[Any]]]) -> dict[str, list[Any]]:
    out = defaultdict(list)
    for rank in root:
        for key, item in rank.items():
            out[key].extend(item)
    return out


def get_accuracy_generations(generations: list[dict[str, list[str]]]) -> float:
    """Slightly less naiive accuracy calculation for generations."""
    accuracy = 0

    if isinstance(generations, list):
        flattened_generations = flatten_ranks(generations)
    else:
        flattened_generations = generations
    for target_text, pred_text in zip(
        flattened_generations["targets"], flattened_generations["predictions"]
    ):
        # TODO: This should be done in the data processing stage - check!.
        target_text = target_text.lower().strip()
        pred_text = pred_text.lower().strip()
        # Using `in` allows to account for noise in the generation at the expense of speed.
        pred_text_is_synonym = any(
            [synonym in pred_text for synonym in SYNONYM_MAP.get("true_text") or []]
        )
        if target_text in pred_text or pred_text_is_synonym:
            accuracy += 1 / len(flattened_generations["predictions"])
    return accuracy


def get_accuracy_contrastive(out: Mapping[str, Tensor]) -> Tensor:
    logits, labels = out["logits"], out["labels"]
    return torch.tensor(
        [
            (labels[range(labels.shape[0]), logits.argmax(dim=1)] == 1).float().sum(),
            labels.shape[0],
        ],
        device=logits.device,
    )


def get_accuracy_simple(out: Mapping[str, Tensor]) -> Tensor:
    logits, labels = out["logits"], out["labels"]
    return torch.tensor(
        [
            (logits.argmax(dim=1) == labels).sum(),
            labels.shape[0],
        ],
        device=logits.device,
    )


def get_average_logits_for_label(out: dict[str, Tensor], label: int) -> Tensor:
    logits, labels = out["logits"], out["labels"]
    mask = labels == label
    target_logits = logits[mask]
    return torch.tensor(
        [target_logits.sum(), mask.sum()],
        device=logits.device,
    )


get_average_positive_logits = partial(get_average_logits_for_label, label=1)
get_average_negative_logits = partial(get_average_logits_for_label, label=-1)


def calculate_throughput(step_time_batch_size: tuple[float, int]) -> float:
    """Calculate samples per second from step time and batch size."""
    step_time, batch_size = step_time_batch_size
    if step_time > 0:
        return batch_size / step_time
    return 0.0


def construct_table(
    generations: dict[str, list[str]] | list[dict[str, list[str]]]
) -> dict:
    """Construct a wandb table for logging generations.

    Generations from each rank are nested in batches."""
    out = wandb.Table(columns=["Target", "Prediction"])
    if isinstance(generations, list):
        column_data = flatten_ranks(generations)
    else:
        column_data = generations
    for target, prediction in zip(column_data["targets"], column_data["predictions"]):
        out.add_data(target, prediction)
    return out


@dataclass
class Metric:
    """An object for the unified tracking and logging of information about a training run.

    Attributes:
        state: Holds the current relevant features of the metric prior to computation. Initial state must be passed in at instantiation.
        transform_fn: Maps InferenceArtefacts to metric state. By default, this function
        accum_fn: Combines two instances of the metrics state. By default, this function will add the two states together.
        reduce_fn: Defines the ReduceOp performed on the metric state if it is a tensor and we are in a distributed setting.
        compute_fn: Maps the metric state to a loggable or plottable value. By default, this function will return the state as is.
        log_every_step: Defines whether the metric should be logged every step or if logging will need to be called manually for the instance.
        log_type: Defines how the metric should be logged.
        reset_rule: Defines when the metric should be set to the intial state. If no reset is desired, set to MetricResetRule.MANUAL.
    """

    name: str
    state: Tensor | int | float | list | None = None
    metric_type: MetricType = MetricType.STATE
    log_every_step: bool = True
    log_type: MetricLogType = MetricLogType.SCALAR
    reset_rule: MetricResetRule = MetricResetRule.ON_LOG
    transform_fn: Callable[[InferenceArtefacts], MetricState] = return_first_value
    accum_fn: Callable[[MetricState, MetricState], MetricState] = add
    reduce_fn: Callable[[Tensor | list, int], Tensor | list] = all_reduce_mean
    compute_fn: Callable[[MetricState], Any] = identity
    device: str | int = "cpu"
    world_size: int = 1
    is_distributed: bool = False

    def __post_init__(self):
        self.is_distributed = self.world_size > 1
        if isinstance(self.state, Tensor):
            self.state = self.state.to(self.device)
        self._default_state = deepcopy(self.state)

    @property
    def value(self) -> Any:
        assert self.state is not None
        if isinstance(self.state, Tensor) and self.is_distributed:
            self.state = self.reduce_fn(self.state, self.world_size)
        return self.compute_fn(self.state)

    def update(self, inference_artefacts: InferenceArtefacts) -> None:
        if self.state is None:
            self.state = self.transform_fn(inference_artefacts)
        else:
            self.state = self.accum_fn(
                self.state,
                self.transform_fn(inference_artefacts),
            )

    def log(self) -> None:
        value = self.value
        if self.device in (0, "cuda", "cuda:0", "cpu", "mps"):
            if isinstance(value, Tensor):
                value = value.item()
            wandb.log(
                {self.name: (plot.line(**value) if isinstance(value, dict) else value)},
                commit=False,
            )
        if self.reset_rule == MetricResetRule.ON_LOG:
            self.reset()

    def reset(self) -> None:
        self.state = deepcopy(self._default_state)


@dataclass
class MetricManager:
    def __init__(self, device, world_size, batch_size, is_main_process, log_interval):
        self.is_main_process = is_main_process
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.micro_batch_size = batch_size // world_size  # For tracking examples per microbatch
        self.step_start_time = None
        self.train_loss = Metric(
            "train/loss", torch.tensor([0.0]), device=device, world_size=world_size
        )
        self.train_gradnorm = Metric(
            "train/gradnorm", tensor([0.0]), device=device, world_size=world_size
        )
        self.microstep = Metric(
            "train/microstep",
            1,
            reset_rule=MetricResetRule.MANUAL,
            device=device,
            world_size=world_size,
        )
        self.step = Metric(
            "train/step",
            1,
            reset_rule=MetricResetRule.MANUAL,
            device=device,
            world_size=world_size,
        )
        self.epoch_microstep = Metric(
            "train/epoch_microstep",
            1,
            transform_fn=position_in_cycle,
            accum_fn=replace,
            reset_rule=MetricResetRule.MANUAL,
            device=device,
            world_size=world_size,
        )
        self.epoch_step = Metric(
            "train/epoch_step",
            1,
            transform_fn=position_in_cycle,
            accum_fn=replace,
            reset_rule=MetricResetRule.MANUAL,
            device=device,
            world_size=world_size,
        )
        self.lr = Metric(
            "train/lr",
            0,
            accum_fn=replace,
            reset_rule=MetricResetRule.MANUAL,
            device=device,
            world_size=world_size,
        )
        self.epoch = Metric(
            "train/epoch",
            1,
            reset_rule=MetricResetRule.MANUAL,
            device=device,
            world_size=world_size,
        )
        self.val_loss = Metric(
            "val/loss",
            tensor([0.0]),
            log_every_step=False,
            device=device,
            world_size=world_size,
        )
        self.val_accuracy = Metric(
            "val/accuracy",
            tensor([0.0]),
            transform_fn=get_accuracy_simple,
            reduce_fn=distributed_identity,  # All ranks should have the same values anyway.
            compute_fn=divide,
            log_every_step=False,
            device=device,
            world_size=world_size,
        )
        self.throughput = Metric(
            "train/samples_per_sec",
            0.0,
            transform_fn=calculate_throughput,
            accum_fn=replace,
            reset_rule=MetricResetRule.MANUAL,
            device=device,
            world_size=world_size,
        )
        self.step_time = Metric(
            "train/step_time_ms",
            0.0,
            transform_fn=lambda x: x * 1000,  # Convert to milliseconds
            accum_fn=replace,
            reset_rule=MetricResetRule.MANUAL,
            device=device,
            world_size=world_size,
        )
        self.examples_seen = Metric(
            "train/examples_seen",
            0,
            reset_rule=MetricResetRule.MANUAL,
            device=device,
            world_size=world_size,
        )

    def log(self):
        if not self.step.value % self.log_interval == 0:
            return
        for key, metric in self.__dict__.items():
            if not isinstance(metric, Metric):
                continue
            if metric.log_every_step:
                metric.log()

        if self.is_main_process:
            wandb.log({}, commit=True)

    def start_step_timer(self):
        """Call this at the beginning of each training step."""
        self.step_start_time = time.time()
    
    def end_step_timer(self):
        """Call this at the end of each training step to log timing metrics."""
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.step_time.update(step_time)
            self.throughput.update((step_time, self.batch_size))
            self.step_start_time = None

    def step_iterators(self, steps_per_epoch: int, num_microbatches: int, lr_scheduler):
        self.step.update(1)
        self.microstep.update(1)
        self.epoch_step.update((self.step.value, steps_per_epoch))
        self.epoch_microstep.update((self.microstep.value, num_microbatches))
        self.lr.update(lr_scheduler.get_last_lr()[0])
