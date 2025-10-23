"""Compare heuristic baselines against a trained PPO policy on sum-rate."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

from huggingface_sb3 import EnvironmentName
import numpy as np
import yaml

from custom_envs.MISOenv import MISOEnv
from heuristics.max_norm_heuristic import HeuristicConfig as MaxNormConfig, run as run_max_norm
from heuristics.random_uniform import HeuristicConfig as RandomConfig, run as run_random
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.utils import get_model_path
import rl_zoo3.import_envs  # noqa: F401  # ensure custom envs are registered


@dataclass
class HeuristicResults:
    max_norm_rate: float
    max_norm_std: float
    random_mean_rate: float
    random_std_rate: float


@dataclass
class PolicyResults:
    mean_rate: float
    std_rate: float
    model_path: str
    label: str


def evaluate_heuristics(num_points: int, num_trials: int, seed: int) -> HeuristicResults:
    """Compute sum-rate metrics for the heuristic baselines."""
    env = MISOEnv(seed=seed)
    env  # noqa: F841  # Keep reference so RNG seed is applied before heuristics run.

    max_cfg = MaxNormConfig(num_points=num_points, seed=seed)
    _, max_norm_rate, max_norm_std = run_max_norm(max_cfg)

    rand_cfg = RandomConfig(num_points=num_points, num_trials=num_trials, seed=seed)
    random_mean_rate, random_std_rate = run_random(rand_cfg)

    return HeuristicResults(
        max_norm_rate=max_norm_rate,
        max_norm_std=max_norm_std,
        random_mean_rate=random_mean_rate,
        random_std_rate=random_std_rate,
    )


def _load_env_kwargs(log_path: str, env_name: EnvironmentName) -> dict | None:
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args.get("env_kwargs") is not None:
                return loaded_args["env_kwargs"]
    return None


def _create_eval_env(env_name: EnvironmentName, log_path: str, seed: int):
    stats_dir = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_dir, norm_reward=False, test_mode=True)
    env_kwargs = _load_env_kwargs(log_path, env_name)
    env = create_test_env(
        env_id=env_name.gym_id,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=seed,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    return env


def _evaluate_policy_final_sum_rate(model, env, n_eval_episodes: int, deterministic: bool) -> Tuple[float, float]:
    final_rates: list[float] = []
    for _ in range(n_eval_episodes):
        reset_output = env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        last_sum_rate = None
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, infos = env.step(action)
            info = infos[0]
            if "best_sum_rate" in info:
                last_sum_rate = float(info["best_sum_rate"])
            elif "sum_rate" in info:
                last_sum_rate = float(info["sum_rate"])
            if done[0]:
                break
        if last_sum_rate is None:
            raise RuntimeError("Environment did not return a sum_rate in info; cannot compute final-position metric.")
        final_rates.append(last_sum_rate)
    mean_rate = float(np.mean(final_rates)) if final_rates else 0.0
    std_rate = float(np.std(final_rates)) if final_rates else 0.0
    return mean_rate, std_rate


def evaluate_ppo(
    algo: str,
    env_id: str,
    logs_folder: str,
    load_best: bool,
    n_eval_episodes: int,
    deterministic: bool,
    device: str,
    seed: int,
) -> PolicyResults:
    """Load a trained PPO agent and evaluate it on the sum-rate reward."""
    env_name = EnvironmentName(env_id)
    _, model_path, log_path = get_model_path(
        exp_id=0,
        folder=logs_folder,
        algo=algo,
        env_name=env_name,
        load_best=load_best,
        load_checkpoint=None,
        load_last_checkpoint=False,
    )

    env = _create_eval_env(env_name, log_path, seed=seed)
    model_cls = ALGOS[algo]
    model = model_cls.load(model_path, device=device)
    mean_rate, std_rate = _evaluate_policy_final_sum_rate(model, env, n_eval_episodes, deterministic)
    env.close()

    return PolicyResults(mean_rate=mean_rate, std_rate=std_rate, model_path=model_path, label="PPO (best)")


def evaluate_grpo_model(
    model_path: str,
    algo: str,
    env_id: str,
    n_eval_episodes: int,
    deterministic: bool,
    device: str,
    seed: int,
    label: str,
) -> PolicyResults:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"GRPO model not found at path: {model_path}")

    env_name = EnvironmentName(env_id)
    log_path = os.path.dirname(model_path)
    env = _create_eval_env(env_name, log_path, seed=seed)
    model_cls = ALGOS[algo]
    model = model_cls.load(model_path, device=device)
    mean_rate, std_rate = _evaluate_policy_final_sum_rate(model, env, n_eval_episodes, deterministic)
    env.close()

    return PolicyResults(mean_rate=mean_rate, std_rate=std_rate, model_path=model_path, label=label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare heuristic baselines with PPO on sum-rate.")
    parser.add_argument("--algo", default="ppo", choices=list(ALGOS.keys()), help="RL algorithm to evaluate.")
    parser.add_argument("--env", default="MISOEnv-custom", help="Environment identifier.")
    parser.add_argument("--logs-folder", "-f", default="logs", help="Training logs directory.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes for PPO.")
    parser.add_argument("--device", default="auto", help="Device for loading the PPO policy.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions when evaluating PPO.")
    parser.add_argument("--heuristic-points", type=int, default=50_000, help="Number of discretised points for heuristics.")
    parser.add_argument("--heuristic-trials", type=int, default=1_000, help="Monte Carlo trials for the random heuristic.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed shared by env and heuristics.")
    parser.add_argument("--no-heuristics", action="store_true", help="Skip heuristic evaluation.")
    parser.add_argument("--no-ppo", action="store_true", help="Skip PPO evaluation.")
    parser.add_argument("--grpo-model", type=str, help="Path to a GRPO fine-tuned model ZIP for evaluation.")
    parser.add_argument("--grpo-label", type=str, default="GRPO", help="Label used when reporting the GRPO result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    heuristics_res: HeuristicResults | None = None
    if not args.no_heuristics:
        heuristics_res = evaluate_heuristics(args.heuristic_points, args.heuristic_trials, args.seed)

    policy_results: list[PolicyResults] = []
    if not args.no_ppo:
        policy_results.append(
            evaluate_ppo(
                algo=args.algo,
                env_id=args.env,
                logs_folder=args.logs_folder,
                load_best=True,
                n_eval_episodes=args.episodes,
                deterministic=args.deterministic,
                device=args.device,
                seed=args.seed,
            )
        )

    if args.grpo_model is not None:
        policy_results.append(
            evaluate_grpo_model(
                model_path=args.grpo_model,
                algo=args.algo,
                env_id=args.env,
                n_eval_episodes=args.episodes,
                deterministic=args.deterministic,
                device=args.device,
                seed=args.seed,
                label=args.grpo_label,
            )
        )

    print("\n=== Sum-Rate Comparison ===")
    if heuristics_res is not None:
        print(
            f"MaxNorm heuristic mean ± std: {heuristics_res.max_norm_rate:.6f} ± "
            f"{heuristics_res.max_norm_std:.6f} bits/s/Hz"
        )
        print(
            f"Random heuristic mean ± std: {heuristics_res.random_mean_rate:.6f} ± "
            f"{heuristics_res.random_std_rate:.6f} bits/s/Hz"
        )
    else:
        print("Heuristic evaluation skipped.")

    if policy_results:
        for result in policy_results:
            print(
                f"{result.label} mean sum-rate over {args.episodes} episodes: "
                f"{result.mean_rate:.6f} ± {result.std_rate:.6f} bits/s/Hz"
            )
            print(f"Model evaluated: {result.model_path}")
    else:
        print("Policy evaluation skipped.")


if __name__ == "__main__":
    main()
