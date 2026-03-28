"""
Conversion du modèle PPO vers ONNX pour Raspberry Pi.
Usage: python -m src.convert_to_onnx
"""

import argparse
import os
import numpy as np

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"[ERROR] Dépendances manquantes: {e}")


def convert_to_onnx(model_path: str, output_dir: str = "real/models"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/3] Chargement: {model_path}")
    model = PPO.load(model_path)
    policy = model.policy
    policy.eval()

    obs_dim = model.observation_space.shape[0]
    dummy_input = torch.randn(1, obs_dim)
    onnx_path = os.path.join(output_dir, "ppo_robot.onnx")

    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            features = self.policy.extract_features(obs)
            if self.policy.share_features_extractor:
                latent_pi = self.policy.mlp_extractor.forward_actor(features)
            else:
                pi_features = self.policy.pi_features_extractor(obs)
                latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            return self.policy.action_net(latent_pi)

    wrapped = PolicyWrapper(policy)
    wrapped.eval()

    print(f"[2/3] Export ONNX: {onnx_path}")
    torch.onnx.export(
        wrapped, dummy_input, onnx_path,
        opset_version=11,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )

    print(f"[3/3] Vérification")
    try:
        import onnx
        onnx.checker.check_model(onnx.load(onnx_path))
        print("[OK] Modèle ONNX valide!")
    except ImportError:
        print("[WARN] onnx non installé")

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n✅ Sauvegardé: {onnx_path} ({size_mb:.2f} MB)")
    print(f"   Copier vers Raspberry Pi: scp {onnx_path} pi@IP:~/models/")


def main():
    if not HAS_DEPS:
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_v6/final_model.zip")
    parser.add_argument("--output", default="real/models")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Modèle non trouvé: {args.model}")
        return

    convert_to_onnx(args.model, args.output)


if __name__ == "__main__":
    main()
