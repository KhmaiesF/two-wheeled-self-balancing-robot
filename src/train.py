"""
================================================================================
🤖 ENTRAÎNEMENT PPO - ROBOT AUTO-ÉQUILIBRANT
================================================================================

Script pour entraîner un modèle PPO avec curriculum learning.

Phases:
1. Stabilité pure (pas de push)
2. Push léger
3. Push moyen + domain randomization
4. Push fort

Usage:
    python -m src.train                    # Entraînement complet
    python -m src.train --steps 100000     # 100k steps
    python -m src.train --render           # Avec visualisation

================================================================================
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment import SelfBalancingRobotEnv, PhysicsConfig, RewardConfig


def make_env(mode: str = "pid+ppo", enable_push: bool = False, **kwargs):
    """Factory pour créer l'environnement."""
    def _init():
        return SelfBalancingRobotEnv(
            mode=mode,
            enable_random_push=enable_push,
            **kwargs
        )
    return _init


def main():
    parser = argparse.ArgumentParser(description="Entraînement PPO")
    parser.add_argument("--steps", type=int, default=300000,
                        help="Nombre total de steps")
    parser.add_argument("--render", action="store_true",
                        help="Visualiser pendant l'entraînement")
    parser.add_argument("--resume", type=str, default=None,
                        help="Reprendre depuis un modèle")
    parser.add_argument("--name", type=str, default=None,
                        help="Nom du run")
    args = parser.parse_args()
    
    # Configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"ppo_{timestamp}"
    save_dir = Path("models") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🤖 ENTRAÎNEMENT PPO - ROBOT AUTO-ÉQUILIBRANT")
    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"Steps: {args.steps:,}")
    print(f"Save dir: {save_dir}")
    print("=" * 60)
    
    # Créer environnement
    render_mode = "human" if args.render else None
    env = DummyVecEnv([make_env(
        mode="pid+ppo",
        enable_push=True,
        push_force_range=(0.3, 1.5),
        push_interval_range=(150, 400),
        render_mode=render_mode
    )])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Environnement d'évaluation
    eval_env = DummyVecEnv([make_env(mode="pid+ppo", enable_push=False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(save_dir / "logs"),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo"
    )
    
    # Créer ou charger modèle
    if args.resume:
        print(f"📦 Chargement modèle: {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(save_dir / "tensorboard")
        )
    
    print("\n🚀 Démarrage entraînement...\n")
    
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu")
    
    # Sauvegarder
    model.save(str(save_dir / "final_model"))
    env.save(str(save_dir / "vecnormalize.pkl"))
    
    print(f"\n✅ Modèle sauvegardé: {save_dir / 'final_model.zip'}")
    print(f"✅ Normalizer sauvegardé: {save_dir / 'vecnormalize.pkl'}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
