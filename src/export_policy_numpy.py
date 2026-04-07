"""
Export a Stable-Baselines3 PPO actor policy to a lightweight NumPy .npz file.

Goal:
- Keep only what is needed for real-time actor inference on Raspberry Pi.
- Avoid exporting the full SB3 object graph.

Usage:
    python -m src.export_policy_numpy models/best_v7/final_model.zip
    python -m src.export_policy_numpy models/best_v7/final_model.zip --output real/models/ppo_policy_weights.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch.nn as nn
    from stable_baselines3 import PPO
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install stable-baselines3 and torch before exporting. "
        f"Details: {exc}"
    )


def _activation_name(module: nn.Module) -> str:
    """Map torch activation modules to small stable string names."""
    if isinstance(module, nn.Tanh):
        return "tanh"
    if isinstance(module, nn.ReLU):
        return "relu"
    if isinstance(module, nn.ELU):
        return "elu"
    if isinstance(module, nn.LeakyReLU):
        return "leaky_relu"
    if isinstance(module, nn.Sigmoid):
        return "sigmoid"
    if isinstance(module, nn.Identity):
        return "identity"
    raise TypeError(f"Unsupported activation module for NumPy export: {module.__class__.__name__}")


def _num_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _extract_actor_ops(policy: nn.Module) -> Tuple[List[Dict[str, str]], List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Extract actor forward graph as a flat op sequence.

    Exported graph:
      features(obs) -> mlp_extractor.policy_net -> action_net

    For the IMU-only MLP setup used in this project, features extractor is expected
    to be flatten/identity-like (no trainable parameters).
    """
    feat_params = _num_params(policy.features_extractor)
    pi_feat_params = 0
    if hasattr(policy, "pi_features_extractor"):
        pi_feat_params = _num_params(policy.pi_features_extractor)

    if feat_params != 0 or pi_feat_params != 0:
        raise RuntimeError(
            "This exporter currently supports only MLP/flatten feature extractors "
            f"(features_extractor={feat_params}, pi_features_extractor={pi_feat_params} trainable params)."
        )

    if not hasattr(policy, "mlp_extractor") or not hasattr(policy.mlp_extractor, "policy_net"):
        raise RuntimeError("Could not find policy.mlp_extractor.policy_net in loaded SB3 policy")

    if not hasattr(policy, "action_net"):
        raise RuntimeError("Could not find policy.action_net in loaded SB3 policy")

    ops: List[Dict[str, str]] = []
    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    linear_names: List[str] = []

    modules = list(policy.mlp_extractor.policy_net.named_children())
    modules.append(("action_net", policy.action_net))

    for name, module in modules:
        if isinstance(module, nn.Linear):
            w = module.weight.detach().cpu().numpy().astype(np.float32)
            b = module.bias.detach().cpu().numpy().astype(np.float32)
            ops.append({"type": "linear", "name": name})
            weights.append(w)
            biases.append(b)
            linear_names.append(name)
            continue

        act = _activation_name(module)
        ops.append({"type": act, "name": name})

    return ops, weights, biases, linear_names


def export_policy_numpy(model_path: Path, output_path: Path) -> Path:
    """Load SB3 PPO and export actor policy weights in a compact NPZ format."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[1/3] Loading PPO model: {model_path}")
    model = PPO.load(str(model_path))
    policy = model.policy
    policy.eval()

    obs_dim = int(model.observation_space.shape[0])
    action_dim = int(model.action_space.shape[0])

    print("[2/3] Extracting actor graph and tensors")
    ops, weights, biases, linear_names = _extract_actor_ops(policy)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data: Dict[str, np.ndarray] = {
        "format_version": np.array([1], dtype=np.int32),
        "obs_dim": np.array([obs_dim], dtype=np.int32),
        "action_dim": np.array([action_dim], dtype=np.int32),
        "op_types": np.array([op["type"] for op in ops], dtype=np.str_),
        "op_names": np.array([op["name"] for op in ops], dtype=np.str_),
        "linear_names": np.array(linear_names, dtype=np.str_),
        "linear_count": np.array([len(weights)], dtype=np.int32),
    }

    for idx, (w, b) in enumerate(zip(weights, biases)):
        save_data[f"linear_{idx}_weight"] = w
        save_data[f"linear_{idx}_bias"] = b

    print(f"[3/3] Writing NPZ: {output_path}")
    np.savez(output_path, **save_data)

    size_kb = output_path.stat().st_size / 1024.0
    print("\nActor export summary")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  feature_extractor={policy.features_extractor.__class__.__name__}")
    print(f"  operations={len(ops)}, linear_layers={len(weights)}")
    linear_cursor = 0
    for i, op in enumerate(ops):
        if op["type"] == "linear":
            w = weights[linear_cursor]
            linear_cursor += 1
            print(f"    [{i:02d}] {op['name']:<16} linear  shape={tuple(w.shape)}")
        else:
            print(f"    [{i:02d}] {op['name']:<16} activation={op['type']}")
    print(f"\nSaved: {output_path} ({size_kb:.1f} KB)")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SB3 PPO actor policy to NumPy NPZ")
    parser.add_argument("model", help="Path to PPO SB3 .zip model")
    parser.add_argument(
        "--output",
        default="real/models/ppo_policy_weights.npz",
        help="Output NPZ path (default: real/models/ppo_policy_weights.npz)",
    )
    args = parser.parse_args()

    export_policy_numpy(Path(args.model), Path(args.output))


if __name__ == "__main__":
    main()
