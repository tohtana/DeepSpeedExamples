"""Metrics collection, CSV writing, and reduction helpers for AutoEP example."""

import csv
import json
import os
from typing import Any

import torch
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator

METRICS_COLUMNS: list[str] = [
    "step",
    "loss_ce",
    "loss_total",
    "loss_objective_tag",
    "iter_time_sec",
    "tokens_per_sec",
    "global_tokens_per_sec",
    "cuda_memory_allocated_bytes",
    "cuda_peak_memory_allocated_bytes",
    "cuda_peak_memory_reserved_bytes",
    "local_expert_param_numel",
    "global_expert_param_numel_est",
    "expert_partition_ratio",
    "use_grouped_mm",
]


class MetricsLogger:
    """Accumulates per-step metrics and writes to CSV.

    Only rank 0 writes the file. Metric reduction (DP-mean for loss,
    max-over-ranks for memory/time) must be performed by the caller
    before calling log_step().
    """

    def __init__(self, csv_path: str, rank: int) -> None:
        self.csv_path = csv_path
        self.rank = rank
        self._file = None
        self._writer = None

    def log_step(self, metrics: dict[str, Any]) -> None:
        """Append one row of metrics. Keys must match METRICS_COLUMNS."""
        if self.rank != 0:
            return

        if self._file is None:
            parent = os.path.dirname(self.csv_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=METRICS_COLUMNS)
            self._writer.writeheader()

        self._writer.writerow({k: metrics.get(k, "") for k in METRICS_COLUMNS})
        self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


def reduce_loss(loss_tensor: torch.Tensor, dp_world_size: int) -> float:
    """All-reduce loss across DP ranks and return DP-mean as Python float.

    Uses torch.distributed.all_reduce with ReduceOp.SUM on the DEFAULT process group
    (all ranks), then divides by dp_world_size.
    """
    loss_clone = loss_tensor.clone().detach()
    dist.all_reduce(loss_clone, op=dist.ReduceOp.SUM)
    return (loss_clone / dp_world_size).item()


def reduce_max(value: float) -> float:
    """All-reduce a scalar across all ranks and return the max.

    Creates a 1-element tensor on current device, all_reduces with ReduceOp.MAX.
    """
    tensor = torch.tensor([value], device=get_accelerator().current_device_name())
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def write_run_metadata(metadata: dict[str, Any], path: str) -> None:
    """Write run metadata dict to JSON file (atomic write via tmp+rename)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metadata, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(os.path.dirname(path) or ".", os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
