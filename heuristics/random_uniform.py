"""Heuristic evaluation for MISOEnv using uniformly sampled discrete positions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple

import numpy as np

# Allow running the script without installing the project as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_envs.MISOenv import MISOEnv


@dataclass
class HeuristicConfig:
    num_points: int = 50_000
    num_trials: int = 1_000
    seed: int = 0


def build_point_cloud(env: MISOEnv, cfg: HeuristicConfig) -> np.ndarray:
    """Uniformly sample points within the feasible region."""
    rng = np.random.default_rng(cfg.seed)
    lower, upper = env.region.lower, env.region.upper
    return rng.uniform(lower, upper, size=(cfg.num_points, 3))


def sample_positions(point_cloud: np.ndarray, env: MISOEnv, rng: np.random.Generator) -> np.ndarray:
    """Select two positions per user from the discrete set while enforcing min distance."""
    positions = np.zeros((env.num_users, env.positions_per_user, 3), dtype=np.float64)
    indices = rng.choice(point_cloud.shape[0], size=(env.num_users, env.positions_per_user), replace=True)
    for user_idx in range(env.num_users):
        p0 = point_cloud[indices[user_idx, 0]]
        p1 = point_cloud[indices[user_idx, 1]]
        attempt = 0
        # Resample second point until distance constraint satisfied.
        while np.linalg.norm(p0 - p1) < env.min_distance:
            indices[user_idx, 1] = rng.integers(point_cloud.shape[0])
            p1 = point_cloud[indices[user_idx, 1]]
            attempt += 1
            # Fail-safe: if discrete set is too coarse, break after a few retries.
            if attempt > 1000:
                break
        positions[user_idx, 0] = p0
        positions[user_idx, 1] = p1
    return positions


def evaluate_trial(env: MISOEnv, point_cloud: np.ndarray, rng: np.random.Generator) -> float:
    """Return the sum-rate for one random assignment of positions."""
    positions = sample_positions(point_cloud, env, rng)
    eval_result = env.evaluate_positions(positions)
    return float(eval_result["sum_rate"])


def run(cfg: HeuristicConfig) -> Tuple[float, float]:
    env = MISOEnv(seed=cfg.seed)
    env.rng = np.random.default_rng()
    point_cloud = build_point_cloud(env, cfg)
    rng = np.random.default_rng(cfg.seed + 1)

    rates = np.empty(cfg.num_trials, dtype=np.float64)
    for idx in range(cfg.num_trials):
        rates[idx] = evaluate_trial(env, point_cloud, rng)

    return float(rates.mean()), float(rates.std(ddof=0))


def main():
    parser = argparse.ArgumentParser(description="Uniform random heuristic for MISOEnv.")
    parser.add_argument("--num-points", type=int, default=50_000, help="Number of discrete points in region.")
    parser.add_argument("--num-trials", type=int, default=1_000, help="Number of Monte Carlo trials.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    cfg = HeuristicConfig(num_points=args.num_points, num_trials=args.num_trials, seed=args.seed)
    mean_rate, std_rate = run(cfg)
    print(f"Average sum-rate over {cfg.num_trials} samples: {mean_rate:.6f} ± {std_rate:.6f} bits/s/Hz")


if __name__ == "__main__":
    main()
