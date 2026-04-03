"""
================================================================================
🤖 ÉVALUATION - ROBOT AUTO-ÉQUILIBRANT
================================================================================

Évalue un modèle PPO sur différents scénarios.

Scénarios:
- baseline: Équilibrage simple sans perturbation
- push_light: Push léger (0.3-0.5N)
- push_heavy: Push fort (1-2N)
- recovery: Test de récupération après perturbation

Usage:
    python -m src.evaluate                           # Tous les scénarios
    python -m src.evaluate --scenario push_heavy     # Un scénario
    python -m src.evaluate --model models/best       # Modèle spécifique
    python -m src.evaluate --render                  # Avec visualisation

================================================================================
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3 import PPO

from src.environment import SelfBalancingRobotEnv


# Définition des scénarios
SCENARIOS = {
    "baseline": {
        "description": "Équilibrage simple",
        "enable_push": False,
        "duration": 600,  # 10 secondes
    },
    "push_light": {
        "description": "Push léger",
        "enable_push": True,
        "push_force_range": (0.3, 0.5),
        "push_interval_range": (120, 180),
        "duration": 1200,  # 20 secondes
    },
    "push_heavy": {
        "description": "Push fort",
        "enable_push": True,
        "push_force_range": (1.0, 2.0),
        "push_interval_range": (180, 300),
        "duration": 1200,
    },
    "endurance": {
        "description": "Test longue durée",
        "enable_push": True,
        "push_force_range": (0.5, 1.5),
        "push_interval_range": (200, 400),
        "duration": 3600,  # 60 secondes
    },
}


def evaluate_scenario(
    model,
    scenario_name: str,
    n_episodes: int = 5,
    render: bool = False,
    mode: str = "pid+ppo"
) -> Dict:
    """Évalue un scénario."""
    scenario = SCENARIOS[scenario_name]
    
    env_kwargs = {
        "mode": mode,
        "enable_random_push": scenario.get("enable_push", False),
        "max_episode_steps": scenario["duration"],
        "render_mode": "human" if render else None,
    }
    
    if "push_force_range" in scenario:
        env_kwargs["push_force_range"] = scenario["push_force_range"]
    if "push_interval_range" in scenario:
        env_kwargs["push_interval_range"] = scenario["push_interval_range"]
    
    env = SelfBalancingRobotEnv(**env_kwargs)
    
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        max_tilt = 0.0
        
        done = False
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(2)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            max_tilt = max(max_tilt, abs(obs[0]))
            
            done = terminated or truncated

        step_duration = env.physics.timestep * env.physics.sim_steps_per_action
        duration_s = episode_steps * step_duration
        
        results.append({
            "success": not terminated,  # terminated = chute
            "fell": bool(terminated),
            "steps": episode_steps,
            "duration_s": duration_s,
            "reward": episode_reward,
            "max_tilt_deg": np.degrees(max_tilt),
        })
    
    env.close()
    
    # Agrégation
    success_rate = sum(r["success"] for r in results) / n_episodes
    avg_reward = np.mean([r["reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_duration_s = np.mean([r["duration_s"] for r in results])
    avg_max_tilt = np.mean([r["max_tilt_deg"] for r in results])
    fall_rate = np.mean([r["fell"] for r in results])
    
    return {
        "scenario": scenario_name,
        "description": scenario["description"],
        "n_episodes": n_episodes,
        "success_rate": success_rate,
        "fall_rate": fall_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_duration_s": avg_duration_s,
        "avg_max_tilt_deg": avg_max_tilt,
    }


def main():
    parser = argparse.ArgumentParser(description="Évaluation robot")
    parser.add_argument("--model", type=str, default=None,
                        help="Chemin vers le modèle PPO")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Scénario spécifique à évaluer")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Nombre d'épisodes par scénario")
    parser.add_argument("--mode", type=str, default="pid+ppo",
                        choices=["pid", "ppo", "pid+ppo"],
                        help="Mode de contrôle")
    parser.add_argument("--render", action="store_true",
                        help="Visualiser")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 ÉVALUATION - ROBOT AUTO-ÉQUILIBRANT")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Épisodes: {args.episodes}")
    print("=" * 60)
    
    # Charger modèle si nécessaire
    model = None
    if args.mode in ["ppo", "pid+ppo"] and args.model:
        model_path = Path(args.model)
        if model_path.exists():
            print(f"📦 Chargement modèle: {model_path}")
            model = PPO.load(str(model_path))
        else:
            print(f"⚠️  Modèle non trouvé: {model_path}")
    
    # Scénarios à évaluer
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    
    results = []
    for scenario_name in scenarios:
        print(f"\n📊 Évaluation: {scenario_name}")
        print("-" * 40)
        
        result = evaluate_scenario(
            model,
            scenario_name,
            n_episodes=args.episodes,
            render=args.render,
            mode=args.mode
        )
        results.append(result)
        
        print(f"  Succès: {result['success_rate']*100:.0f}%")
        print(f"  Chute: {result['fall_rate']*100:.0f}%")
        print(f"  Reward moyen: {result['avg_reward']:.1f}")
        print(f"  Steps moyen: {result['avg_steps']:.0f}")
        print(f"  Durée moyenne: {result['avg_duration_s']:.2f}s")
        print(f"  Tilt max moyen: {result['avg_max_tilt_deg']:.1f}°")
    
    # Résumé
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ")
    print("=" * 60)
    print(f"{'Scénario':<15} {'Succès':>10} {'Chute':>10} {'Reward':>10} {'Tilt max':>12}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['scenario']:<15} {r['success_rate']*100:>9.0f}% {r['fall_rate']*100:>9.0f}% "
            f"{r['avg_reward']:>10.1f} {r['avg_max_tilt_deg']:>11.1f}°"
        )
    
    # Score global
    global_success = np.mean([r['success_rate'] for r in results])
    print("-" * 60)
    print(f"{'GLOBAL':<15} {global_success*100:>9.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
