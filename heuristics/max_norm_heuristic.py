"""Heuristic that selects RX positions maximizing channel 2-norm within a discretised region."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_envs.MISOenv import MISOEnv


@dataclass
class HeuristicConfig:
    num_points: int = 50_000
    seed: int = 0
    num_evals: int = 1_000


def build_point_cloud(env: MISOEnv, cfg: HeuristicConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    lower, upper = env.region.lower, env.region.upper
    return rng.uniform(lower, upper, size=(cfg.num_points, 3))


def select_best_positions(env: MISOEnv, point_cloud: np.ndarray) -> np.ndarray:
    """Pick two positions per user that maximise the channel 2-norm."""
    positions = np.zeros((env.num_users, env.positions_per_user, 3), dtype=np.float64)

    for user_idx in range(env.num_users):
        # Evaluate channel vector at each candidate point.
        channel_vectors = []
        norms = np.empty(point_cloud.shape[0], dtype=np.float64)
        for idx, point in enumerate(point_cloud):
            chan = (
                env.channel_model.compute_channel(
                    user_idx,
                    rx_position=torch.tensor(point, dtype=torch.float32),
                    tx_positions=env.tx_positions,
                )
                .numpy()
                .ravel()
            )
            channel_vectors.append(chan)
            norms[idx] = np.linalg.norm(chan)

        sorted_indices = np.argsort(norms)[::-1]
        best_idx = sorted_indices[0]
        positions[user_idx, 0] = point_cloud[best_idx]

        # Find the strongest remaining point that respects minimum distance.
        second_idx = None
        for candidate in sorted_indices[1:]:
            if np.linalg.norm(point_cloud[best_idx] - point_cloud[candidate]) >= env.min_distance:
                second_idx = candidate
                break

        # Fallback: if no point satisfies distance, reuse the best point.
        if second_idx is None:
            second_idx = best_idx

        positions[user_idx, 1] = point_cloud[second_idx]

    return positions


def run(cfg: HeuristicConfig) -> Tuple[np.ndarray, float, float]:
    env = MISOEnv(seed=cfg.seed)
    # Reinitialize RNG so each call samples fresh channel estimation errors.
    env.rng = np.random.default_rng()
    point_cloud = build_point_cloud(env, cfg)
    positions = select_best_positions(env, point_cloud)

    rates = np.empty(cfg.num_evals, dtype=np.float64)
    for idx in range(cfg.num_evals):
        eval_result = env.evaluate_positions(positions)
        rates[idx] = float(eval_result["sum_rate"])

    return positions, float(rates.mean()), float(rates.std(ddof=0))


def main():
    parser = argparse.ArgumentParser(description="Max-norm heuristic for MISOEnv.")
    parser.add_argument("--num-points", type=int, default=50_000, help="Number of discretised points.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--num-evals",
        type=int,
        default=1_000,
        help="Number of error resamplings used to average the robust sum-rate.",
    )
    args = parser.parse_args()

    cfg = HeuristicConfig(num_points=args.num_points, seed=args.seed, num_evals=args.num_evals)
    positions, mean_rate, std_rate = run(cfg)

    np.set_printoptions(precision=4, suppress=True)
    print("Selected positions (per user, two entries):")
    print(positions)
    print(f"Average sum-rate over {cfg.num_evals} samples: {mean_rate:.6f} ± {std_rate:.6f} bits/s/Hz")


if __name__ == "__main__":
    main()
