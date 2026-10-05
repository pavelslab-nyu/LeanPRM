"""Launch the paper's 300-update GRPO configuration with explicit local paths."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def command(args):
    root = Path(__file__).resolve().parent
    eight = args.model.rstrip("/").endswith("8B")
    options = {
        "algorithm.adv_estimator": "grpo",
        "data.train_files": str(args.data_dir / "train_weighted.parquet"),
        "data.val_files": str(args.data_dir / "validation.parquet"),
        "data.train_batch_size": 64,
        "data.val_batch_size": 200,
        "data.validation_shuffle": "False",
        "data.max_prompt_length": 8192,
        "data.max_response_length": 6656,
        "data.filter_overlong_prompts": "True",
        "data.truncation": "error",
        "+data.apply_chat_template_kwargs.enable_thinking": "True",
        "actor_rollout_ref.model.path": args.model,
        "actor_rollout_ref.model.use_remove_padding": "True",
        "actor_rollout_ref.model.enable_gradient_checkpointing": "True",
        "actor_rollout_ref.actor.optim.lr": "1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size": 64,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 2,
        "actor_rollout_ref.actor.use_kl_loss": "True",
        "actor_rollout_ref.actor.kl_loss_coef": 0.001,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.clip_ratio_low": 0.2,
        "actor_rollout_ref.actor.clip_ratio_high": 0.26,
        "actor_rollout_ref.actor.clip_ratio_c": 10.0,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.fsdp_config.param_offload": str(eight),
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": str(eight),
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 2,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 2 if eight else 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.7,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 32768,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.n": 8,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": "False",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 2,
        "actor_rollout_ref.ref.fsdp_config.param_offload": "True",
        "algorithm.use_kl_in_reward": "False",
        "reward_model.reward_manager": "batch",
        "custom_reward_function.path": str(
            root / ("solver_task.py" if args.task == "solver" else "verifier_task.py")
        ),
        "custom_reward_function.name": "compute_score_batch",
        "trainer.critic_warmup": 0,
        "trainer.default_hdfs_dir": "null",
        "trainer.default_local_dir": str(args.output / "checkpoints"),
        "trainer.resume_mode": "disable",
        "trainer.logger": "[console]",
        "trainer.project_name": "LeanPRM",
        "trainer.experiment_name": args.name,
        "trainer.n_gpus_per_node": args.gpus,
        "trainer.nnodes": 1,
        "trainer.val_before_train": "True",
        "trainer.save_freq": 50,
        "trainer.test_freq": 50,
        "trainer.max_actor_ckpt_to_keep": "null",
        "trainer.total_epochs": 100,
        "trainer.total_training_steps": args.updates,
        "trainer.rollout_data_dir": str(args.output / "rollouts/train"),
        "trainer.validation_data_dir": str(args.output / "rollouts/validation"),
    }
    return [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        *[f"{k}={v}" for k, v in options.items()],
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="leanprm-300")
    p.add_argument("--task", choices=["verifier", "solver"], default="verifier")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.gpus < 1 or args.updates < 1:
        p.error("GPU count and updates must be positive")
    args.data_dir = args.data_dir.resolve()
    args.output = args.output.resolve()
    for name in ["train_weighted.parquet", "validation.parquet"]:
        if not (args.data_dir / name).is_file():
            p.error(f"Missing {args.data_dir / name}")
    cmd = command(args)
    if args.dry_run:
        print(shlex.join(cmd))
        return
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "launch.json").write_text(
        json.dumps(
            {"argv": cmd, "model": args.model, "updates": args.updates}, indent=2
        )
        + "\n"
    )
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
