"""AutoEP vs ZeRO-3 leaf training comparison script.

Runs a randomly-initialized Mixtral MoE model in either AutoEP+ZeRO-2 mode
or HF-native+ZeRO-3 leaf-module mode, collecting per-step metrics for comparison.

Launch via deepspeed launcher:
    deepspeed --num_gpus 8 train_compare.py --mode autoep --deepspeed_config configs/ds_autoep_zero2.json
    deepspeed --num_gpus 8 train_compare.py --mode zero3_leaf --deepspeed_config configs/ds_zero3_leaf.json
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import deepspeed
from deepspeed.accelerator import get_accelerator

from data_utils import SyntheticBatchGenerator, build_mixtral_config
from init_weights import load_init_weights_artifact, save_init_weights_artifact
from metrics import MetricsLogger, reduce_loss, reduce_max, write_run_metadata
from validation import (
    collect_run_metadata,
    validate_autoep_engine,
    validate_zero3_leaf_engine,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoEP vs ZeRO-3 leaf training comparison"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["autoep", "zero3_leaf"],
        help="Training mode",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        required=True,
        help="Path to DeepSpeed JSON config",
    )
    parser.add_argument("--steps", type=int, default=50, help="Total optimizer steps")
    parser.add_argument(
        "--warmup_steps", type=int, default=5, help="Warmup steps before measurement"
    )
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Log every N optimizer steps"
    )
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--micro_batch_size", type=int, default=2, help="Micro batch size per GPU"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--target_global_tokens_per_update",
        type=int,
        default=None,
        help="Target global tokens per optimizer update; derives grad_accum per mode",
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--autoep_size",
        type=int,
        default=None,
        help="Override autoep_size (AutoEP mode only)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=str,
        choices=["on", "off"],
        default="off",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--include_router_aux_loss",
        type=str,
        choices=["on", "off"],
        default="off",
        help="Include router auxiliary loss",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable torch.use_deterministic_algorithms",
    )
    parser.add_argument(
        "--allow_untested_versions",
        action="store_true",
        help="Bypass version compatibility gate",
    )
    parser.add_argument(
        "--metrics_out", type=str, default=None, help="CSV output path"
    )
    parser.add_argument(
        "--run_metadata_out", type=str, default=None, help="Metadata JSON output path"
    )
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default=None,
        help="Save checkpoint to this directory after training",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Load checkpoint from this directory before training",
    )
    parser.add_argument(
        "--save_init_weights",
        type=str,
        default=None,
        help="Save pre-DeepSpeed init weights artifact (.safetensors)",
    )
    parser.add_argument(
        "--load_init_weights",
        type=str,
        default=None,
        help="Load pre-DeepSpeed init weights artifact (.safetensors)",
    )
    parser.add_argument(
        "--init_weights_only",
        action="store_true",
        help="Save init weights artifact and exit before DeepSpeed initialization",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by deepspeed launcher",
    )
    args = parser.parse_args()
    validate_init_weight_args(args, parser)
    return args


def validate_init_weight_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Validate init-weights/checkpoint argument combinations."""
    if args.init_weights_only and args.save_init_weights is None:
        parser.error("--init_weights_only requires --save_init_weights.")

    if args.init_weights_only and args.load_checkpoint is not None:
        parser.error("--init_weights_only is incompatible with --load_checkpoint.")

    if args.init_weights_only and args.save_checkpoint is not None:
        parser.error("--init_weights_only is incompatible with --save_checkpoint.")

    if args.load_init_weights is not None and args.load_checkpoint is not None:
        parser.error("--load_init_weights is incompatible with --load_checkpoint.")

    if args.save_init_weights is not None and args.load_init_weights is not None:
        parser.error(
            "--save_init_weights and --load_init_weights cannot be used together."
        )

    if args.save_init_weights is not None and not args.save_init_weights.endswith(
        ".safetensors"
    ):
        parser.error("--save_init_weights path must end with '.safetensors'.")

    if args.load_init_weights is not None:
        if not args.load_init_weights.endswith(".safetensors"):
            parser.error("--load_init_weights path must end with '.safetensors'.")
        if not os.path.isfile(args.load_init_weights):
            parser.error(
                f"--load_init_weights file does not exist: {args.load_init_weights}"
            )


def load_ds_config(config_path: str) -> dict:
    """Load and return DeepSpeed config as a dict."""
    with open(config_path) as f:
        return json.load(f)


def main():
    args = parse_args()

    # Set defaults for output paths
    if args.metrics_out is None:
        args.metrics_out = f"metrics_{args.mode}.csv"
    if args.run_metadata_out is None:
        args.run_metadata_out = f"run_metadata_{args.mode}.json"

    if args.init_weights_only:
        rank = 0
        world_size = 1
    else:
        # deepspeed.initialize handles distributed setup, but we need rank for logging
        deepspeed.init_distributed()
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if get_accelerator().is_available():
            local_rank_env = int(os.environ.get("LOCAL_RANK", args.local_rank))
            if local_rank_env >= 0:
                get_accelerator().set_device(local_rank_env)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[rank {rank}] %(levelname)s: %(message)s",
    )

    # Set seeds BEFORE model construction
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if get_accelerator().is_available():
        get_accelerator().manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load DS config
    ds_config = load_ds_config(args.deepspeed_config)

    # Validate precision
    bf16_enabled = ds_config.get("bf16", {}).get("enabled", False)
    fp16_enabled = ds_config.get("fp16", {}).get("enabled", False)
    if not bf16_enabled and not fp16_enabled:
        logger.error(
            "Neither bf16 nor fp16 is enabled. FP32-only is not supported."
        )
        sys.exit(2)
    if fp16_enabled and not bf16_enabled:
        logger.warning(
            "fp16 is enabled but bf16 is preferred for Hopper grouped GEMM fast-path."
        )

    # Override batch config from CLI args
    ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    ds_config["gradient_accumulation_steps"] = args.grad_accum
    # Remove train_batch_size if present; let DeepSpeed derive it
    ds_config.pop("train_batch_size", None)

    # Override autoep_size if provided
    if args.mode == "autoep" and args.autoep_size is not None:
        if "expert_parallel" not in ds_config:
            ds_config["expert_parallel"] = {}
        ds_config["expert_parallel"]["autoep_size"] = args.autoep_size

    # Read autoep_size from config for validation
    autoep_size = 1
    if args.mode == "autoep":
        autoep_size = ds_config.get("expert_parallel", {}).get("autoep_size", 1)
        if autoep_size == 1:
            logger.warning(
                "autoep_size=1: EP communication is bypassed (degenerate case). "
                "Set autoep_size >= 2 to test expert parallelism."
            )

    # Derive grad_accum from target_global_tokens_per_update if provided
    if args.target_global_tokens_per_update is not None:
        if args.mode == "autoep":
            dp_ws = world_size // autoep_size
        else:
            dp_ws = world_size
        tokens_per_microstep = args.seq_len * args.micro_batch_size * dp_ws
        if tokens_per_microstep == 0:
            logger.error("tokens_per_microstep is 0; check seq_len, micro_batch_size.")
            sys.exit(2)
        derived_ga = args.target_global_tokens_per_update / tokens_per_microstep
        if derived_ga != int(derived_ga) or derived_ga < 1:
            logger.error(
                f"target_global_tokens_per_update={args.target_global_tokens_per_update} "
                f"is not evenly divisible by tokens_per_microstep={tokens_per_microstep}. "
                f"Derived grad_accum={derived_ga} is not a positive integer."
            )
            sys.exit(2)
        args.grad_accum = int(derived_ga)
        ds_config["gradient_accumulation_steps"] = args.grad_accum
        if rank == 0:
            logger.info(
                f"Derived grad_accum={args.grad_accum} from "
                f"target_global_tokens_per_update={args.target_global_tokens_per_update}"
            )

    # Read load_balance_coeff for validation
    load_balance_coeff = None
    if args.mode == "autoep":
        load_balance_coeff = ds_config.get("expert_parallel", {}).get(
            "load_balance_coeff", None
        )

    # Build model config
    output_router_logits = args.include_router_aux_loss == "on"
    model_config = build_mixtral_config(
        num_layers=args.num_layers,
        output_router_logits=output_router_logits,
    )
    num_experts = model_config.num_local_experts  # HF "num_local_experts" = total experts per layer

    if rank == 0:
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Model: Mixtral with {args.num_layers} layers, {num_experts} experts")
        logger.info(f"Seq len: {args.seq_len}, Micro batch: {args.micro_batch_size}")
        logger.info(f"Grad accum: {args.grad_accum}, Steps: {args.steps}")

    # Build model with random weights
    model = AutoModelForCausalLM.from_config(model_config)

    init_weights_context = {
        "init_weights_path": None,
        "init_weights_sha256": None,
        "init_weights_loaded": False,
        "init_weights_schema_version": None,
    }

    # Optional: save pre-DeepSpeed init weights artifact
    if args.save_init_weights is not None:
        if args.init_weights_only or rank == 0:
            init_weights_context = save_init_weights_artifact(
                args.save_init_weights,
                model,
                args=args,
                model_config=model_config,
                rank=rank,
            )
            if rank == 0:
                logger.info(
                    f"Saved init weights artifact to {init_weights_context['init_weights_path']}"
                )
        if not args.init_weights_only and torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Optional: load pre-DeepSpeed init weights artifact
    if args.load_init_weights is not None:
        if torch.distributed.is_initialized():
            # Rank-0 validates readability and broadcasts result to all ranks.
            read_ok = int(
                rank == 0
                and os.path.isfile(args.load_init_weights)
                and os.access(args.load_init_weights, os.R_OK)
            )
            check = torch.tensor([read_ok], device=get_accelerator().current_device_name())
            torch.distributed.broadcast(check, src=0)
            if check.item() != 1:
                logger.error(
                    f"Init weights path is not readable on rank 0: {args.load_init_weights}"
                )
                sys.exit(2)
            torch.distributed.barrier()

        try:
            init_weights_context = load_init_weights_artifact(
                args.load_init_weights,
                model,
                args=args,
                model_config=model_config,
            )
        except Exception as e:
            logger.error(f"Failed to load init weights artifact: {e}")
            sys.exit(2)
        if rank == 0:
            logger.info(
                "Loaded init weights artifact "
                f"{init_weights_context['init_weights_path']} "
                f"(sha256={init_weights_context['init_weights_sha256']})"
            )

    if args.init_weights_only:
        if rank == 0:
            logger.info("init_weights_only completed successfully.")
        return

    # Enable gradient checkpointing if requested (BEFORE deepspeed.initialize)
    if args.gradient_checkpointing == "on":
        model.gradient_checkpointing_enable()
        if rank == 0:
            logger.warning(
                "Gradient checkpointing enabled. tokens_per_expert will be inflated 2x "
                "by forward recomputation. Router logit hooks run 4x per layer."
            )

    # Initialize DeepSpeed engine
    try:
        if args.mode == "autoep":
            # AutoEP: do NOT pass optimizer; let DS build from config for MoE param groups
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                config=ds_config,
                model_parameters=model.parameters(),
            )
        else:
            # ZeRO-3 leaf: same pattern (DS builds optimizer from config)
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                config=ds_config,
                model_parameters=model.parameters(),
            )
    except Exception as e:
        logger.error(f"deepspeed.initialize() failed: {e}")
        sys.exit(2)

    if rank == 0:
        logger.info(f"DeepSpeed engine initialized. dp_world_size={engine.dp_world_size}")

    # Post-init validation
    gc_enabled = args.gradient_checkpointing == "on"
    if args.mode == "autoep":
        val_result = validate_autoep_engine(
            engine, autoep_size, num_experts, load_balance_coeff, gc_enabled
        )
    else:
        val_result = validate_zero3_leaf_engine(engine)

    if rank == 0:
        for w in val_result.get("warnings", []):
            logger.warning(f"Validation: {w}")
        for e in val_result.get("errors", []):
            logger.error(f"Validation: {e}")
        if not val_result["valid"]:
            logger.error("Post-init validation failed.")
            sys.exit(2)
        logger.info("Post-init validation passed.")
        if args.mode == "autoep":
            logger.info(
                f"Expert params: local={val_result['local_expert_param_numel']:,}, "
                f"global_est={val_result['global_expert_param_numel_est']:,}, "
                f"partition_ratio={val_result['expert_partition_ratio']:.1f}, "
                f"use_grouped_mm={val_result['use_grouped_mm']}"
            )

    # Checkpoint load (optional)
    start_step = 0
    if args.load_checkpoint is not None:
        load_path, client_state = engine.load_checkpoint(args.load_checkpoint)
        if load_path is None:
            logger.error(
                f"Failed to load checkpoint from {args.load_checkpoint}"
            )
            sys.exit(2)
        start_step = client_state.get("step", 0) if client_state else 0
        if rank == 0:
            logger.info(f"Loaded checkpoint from {load_path}, starting at step {start_step}")

    # Get DP rank for batch generator
    import deepspeed.comm as dist_comm

    dp_rank = dist_comm.get_rank(engine.data_parallel_group)
    dp_world_size = engine.dp_world_size

    # Create batch generator
    batch_gen = SyntheticBatchGenerator(
        vocab_size=model_config.vocab_size,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        total_steps=args.steps,
        grad_accum=args.grad_accum,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        seed=args.seed,
    )

    # Collect run metadata
    metadata = collect_run_metadata(
        args.mode,
        args,
        engine,
        val_result,
        init_weights_context=init_weights_context,
    )

    # Setup metrics logger
    metrics_logger = MetricsLogger(args.metrics_out, rank)

    # Determine loss objective tag
    loss_tag = "ce_plus_aux" if output_router_logits else "ce_only"

    # Static metric fields from validation
    static_metrics = {}
    if args.mode == "autoep":
        static_metrics.update(
            {
                "local_expert_param_numel": val_result["local_expert_param_numel"],
                "global_expert_param_numel_est": val_result[
                    "global_expert_param_numel_est"
                ],
                "expert_partition_ratio": val_result["expert_partition_ratio"],
                "use_grouped_mm": val_result["use_grouped_mm"],
            }
        )
    else:
        static_metrics.update(
            {
                "local_expert_param_numel": "",
                "global_expert_param_numel_est": "",
                "expert_partition_ratio": "",
                "use_grouped_mm": "",
            }
        )

    # Training loop
    if rank == 0:
        logger.info(f"Starting training for {args.steps} optimizer steps (warmup={args.warmup_steps})...")

    tokens_per_microstep = args.seq_len * args.micro_batch_size

    for step in range(start_step, args.steps):
        get_accelerator().synchronize()
        step_start = time.time()

        last_loss = None

        for accum_idx in range(args.grad_accum):
            batch = batch_gen.get_batch(step, accum_idx)
            batch_dict = {
                "input_ids": batch.input_ids.to(engine.device),
                "attention_mask": batch.attention_mask.to(engine.device),
                "labels": batch.labels.to(engine.device),
            }

            outputs = engine(**batch_dict)
            loss = outputs.loss
            last_loss = loss.detach().clone()

            engine.backward(loss)
            engine.step()

        get_accelerator().synchronize()
        step_end = time.time()
        iter_time = step_end - step_start

        # Reduce loss (DP-mean) - use last microstep loss
        reduced_loss = reduce_loss(last_loss, dp_world_size)

        # Non-finite loss check (every step including warmup)
        if not math.isfinite(reduced_loss):
            if rank == 0:
                logger.error(f"Non-finite loss at step {step}: {reduced_loss}")
            sys.exit(3)

        # Reset peak memory stats after warmup
        if step == args.warmup_steps - 1:
            get_accelerator().reset_peak_memory_stats()

        # Log metrics for steps >= warmup_steps
        if step >= args.warmup_steps and step % args.log_interval == 0:
            # Reduce timing (max across ranks)
            max_iter_time = reduce_max(iter_time)

            # Memory stats
            mem_allocated = get_accelerator().memory_allocated()
            mem_peak_allocated = get_accelerator().max_memory_allocated()
            mem_peak_reserved = get_accelerator().max_memory_reserved()

            # Throughput
            total_tokens_this_step = tokens_per_microstep * args.grad_accum
            tokens_per_sec = total_tokens_this_step / max_iter_time if max_iter_time > 0 else 0
            global_tokens_per_sec = (
                args.seq_len
                * args.micro_batch_size
                * args.grad_accum
                * dp_world_size
                / max_iter_time
                if max_iter_time > 0
                else 0
            )

            step_metrics = {
                "step": step,
                "loss_ce": reduced_loss,
                "loss_total": reduced_loss,
                "loss_objective_tag": loss_tag,
                "iter_time_sec": max_iter_time,
                "tokens_per_sec": tokens_per_sec,
                "global_tokens_per_sec": global_tokens_per_sec,
                "cuda_memory_allocated_bytes": mem_allocated,
                "cuda_peak_memory_allocated_bytes": mem_peak_allocated,
                "cuda_peak_memory_reserved_bytes": mem_peak_reserved,
                **static_metrics,
            }
            metrics_logger.log_step(step_metrics)

            if rank == 0:
                logger.info(
                    f"Step {step}: loss={reduced_loss:.6f}, "
                    f"time={max_iter_time:.3f}s, "
                    f"global_tps={global_tokens_per_sec:.0f}"
                )

    metrics_logger.close()

    # Checkpoint save (optional)
    if args.save_checkpoint is not None:
        engine.save_checkpoint(
            args.save_checkpoint, client_state={"step": args.steps}
        )
        if rank == 0:
            logger.info(f"Saved checkpoint to {args.save_checkpoint}")

    # Write run metadata (rank 0 only)
    if rank == 0:
        write_run_metadata(metadata, args.run_metadata_out)
        logger.info(f"Metrics written to {args.metrics_out}")
        logger.info(f"Metadata written to {args.run_metadata_out}")
        logger.info("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
