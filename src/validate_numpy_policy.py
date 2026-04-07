"""
Validate that NumPy PPO actor inference matches the original SB3 PPO actor.

Usage examples:
    python -m src.validate_numpy_policy --model models/best_v7/final_model.zip --weights real/models/ppo_policy_weights.npz
    python -m src.validate_numpy_policy --model models/best_v7/final_model.zip --weights real/models/ppo_policy_weights.npz --num-samples 12 --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

try:
    from real.ppo_numpy_policy import PpoNumpyPolicy
except Exception:
    # Fallback when namespace import is not available.
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "real" / "ppo_numpy_policy.py"
    spec = importlib.util.spec_from_file_location("ppo_numpy_policy", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ppo_numpy_policy"] = module
    spec.loader.exec_module(module)
    PpoNumpyPolicy = module.PpoNumpyPolicy


def build_test_observations(num_samples: int, obs_dim: int, seed: int) -> np.ndarray:
    """Create deterministic IMU-like test observations.

    Observation core (IMU-only):
      [tilt_angle, tilt_rate, last_torque_left, last_torque_right]

    If obs_dim > 4, remaining dimensions are set to zero for safe comparison.
    """
    rng = np.random.default_rng(seed)

    # A few deterministic edge/sanity cases first.
    base_cases = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.08, 0.6, 0.0, 0.0],
            [-0.12, -1.1, 0.15, -0.15],
            [0.18, 2.0, -0.35, -0.35],
            [-0.22, -2.5, 0.45, 0.40],
        ],
        dtype=np.float32,
    )

    target_n = max(1, int(num_samples))
    obs = np.zeros((target_n, obs_dim), dtype=np.float32)

    n_base = min(len(base_cases), target_n)
    if obs_dim > 0:
        obs[:n_base, : min(4, obs_dim)] = base_cases[:n_base, : min(4, obs_dim)]

    # Fill the rest with random but realistic IMU/control values.
    for i in range(n_base, target_n):
        tilt_angle = float(rng.uniform(-0.28, 0.28))      # rad
        tilt_rate = float(rng.uniform(-3.5, 3.5))         # rad/s
        last_u_left = float(rng.uniform(-0.5, 0.5))       # torque-equivalent
        last_u_right = float(rng.uniform(-0.5, 0.5))      # torque-equivalent

        if obs_dim >= 1:
            obs[i, 0] = tilt_angle
        if obs_dim >= 2:
            obs[i, 1] = tilt_rate
        if obs_dim >= 3:
            obs[i, 2] = last_u_left
        if obs_dim >= 4:
            obs[i, 3] = last_u_right

    return obs


def sb3_actor_forward(model: PPO, obs_batch: np.ndarray) -> np.ndarray:
    """Forward actor path exactly like the export path (deterministic actor output)."""
    policy = model.policy
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=policy.device)

    with torch.no_grad():
        features = policy.extract_features(obs_t)
        if policy.share_features_extractor:
            latent_pi = policy.mlp_extractor.forward_actor(features)
        else:
            pi_features = policy.pi_features_extractor(obs_t)
            latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        actions = policy.action_net(latent_pi)

    return actions.detach().cpu().numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate NumPy PPO actor vs SB3 actor outputs")
    parser.add_argument("--model", required=True, help="Path to SB3 PPO .zip model")
    parser.add_argument(
        "--weights",
        default="real/models/ppo_policy_weights.npz",
        help="Path to exported NumPy weights (.npz)",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of test observations")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for test observations")
    parser.add_argument("--tol-max", type=float, default=1e-5, help="Max absolute error tolerance")
    parser.add_argument("--tol-mean", type=float, default=1e-6, help="Mean absolute error tolerance")
    args = parser.parse_args()

    model_path = Path(args.model)
    weights_path = Path(args.weights)

    if not model_path.exists():
        print(f"[ERROR] SB3 model not found: {model_path}")
        return 1
    if not weights_path.exists():
        print(f"[ERROR] NumPy weights not found: {weights_path}")
        return 1

    print(f"[1/4] Loading SB3 PPO model: {model_path}")
    model = PPO.load(str(model_path))

    print(f"[2/4] Loading NumPy policy: {weights_path}")
    np_policy = PpoNumpyPolicy(weights_path)

    sb3_obs_dim = int(model.observation_space.shape[0])
    np_obs_dim = int(np_policy.obs_dim)
    if sb3_obs_dim != np_obs_dim:
        print(f"[ERROR] Observation dim mismatch: SB3={sb3_obs_dim}, NumPy={np_obs_dim}")
        return 1

    print(f"[3/4] Building {args.num_samples} IMU-like test observations (obs_dim={sb3_obs_dim})")
    obs_batch = build_test_observations(args.num_samples, sb3_obs_dim, args.seed)

    print("[4/4] Running actor inference comparison")
    sb3_actions = sb3_actor_forward(model, obs_batch)
    np_actions = np_policy.predict(obs_batch)

    abs_err = np.abs(sb3_actions - np_actions)
    max_abs_err = float(abs_err.max())
    mean_abs_err = float(abs_err.mean())

    print("\n=== Detailed per-sample comparison ===")
    for i in range(obs_batch.shape[0]):
        print(f"sample {i:02d} | obs={obs_batch[i].tolist()}")
        print(f"           | sb3={sb3_actions[i].tolist()}")
        print(f"           | numpy={np_actions[i].tolist()}")
        print(f"           | abs_err={abs_err[i].tolist()}")

    print("\n=== Global error metrics ===")
    print(f"max_abs_error  = {max_abs_err:.10e}")
    print(f"mean_abs_error = {mean_abs_err:.10e}")
    print(f"tol_max        = {args.tol_max:.10e}")
    print(f"tol_mean       = {args.tol_mean:.10e}")

    ok = (max_abs_err <= args.tol_max) and (mean_abs_err <= args.tol_mean)
    if ok:
        print("\n[PASS] NumPy policy reproduces SB3 actor outputs within tolerance.")
        return 0

    print("\n[FAIL] NumPy policy differs from SB3 actor beyond tolerance.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
