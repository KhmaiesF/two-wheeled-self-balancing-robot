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

from src.environment import PhysicsConfig, SelfBalancingRobotEnv


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
    "push_extreme": {
        "description": "Push extrêmes fréquents",
        "enable_push": True,
        "push_force_range": (1.8, 3.0),
        "push_interval_range": (60, 120),
        "duration": 1800,
    },
    "endurance_hard": {
        "description": "Endurance difficile + bruit IMU renforcé",
        "enable_push": True,
        "push_force_range": (0.8, 2.4),
        "push_interval_range": (90, 150),
        "duration": 5400,
        "physics_overrides": {
            "tilt_noise_std": float(np.deg2rad(0.45)),
            "tilt_rate_noise_std": float(np.deg2rad(1.8)),
            "gyro_bias_std": float(np.deg2rad(0.7)),
        },
    },
    "randomized_physics": {
        "description": "Randomisation légère de physique au reset",
        "enable_push": True,
        "push_force_range": (0.7, 2.0),
        "push_interval_range": (100, 180),
        "duration": 1800,
        "enable_randomized_physics": True,
        "randomization_ranges": {
            "lateral_friction_scale": (0.9, 1.1),
            "rolling_friction_scale": (0.85, 1.15),
            "spinning_friction_scale": (0.85, 1.15),
            "effective_torque_scale": (0.92, 1.08),
            "base_mass_scale": (0.96, 1.04),
        },
    },
    "motor_asymmetry": {
        "description": "Asymétrie moteur gauche/droite",
        "enable_push": True,
        "push_force_range": (1.0, 2.0),
        "push_interval_range": (100, 170),
        "duration": 1800,
        "motor_left_scale": 0.88,
        "motor_right_scale": 1.00,
    },
    "control_delay": {
        "description": "Délai de commande moteur",
        "enable_push": True,
        "push_force_range": (1.0, 2.1),
        "push_interval_range": (85, 150),
        "duration": 1800,
        "control_delay_steps": 2,
        "physics_overrides": {
            "action_smoothing_alpha": 0.22,
        },
    },
    "sensor_bias_hard": {
        "description": "Biais capteurs IMU renforcé",
        "enable_push": True,
        "push_force_range": (1.0, 2.2),
        "push_interval_range": (90, 160),
        "duration": 1800,
        "extra_tilt_bias": float(np.deg2rad(1.2)),
        "extra_tilt_rate_bias": float(np.deg2rad(5.0)),
        "extra_noise_scale": 1.8,
    },
    "combined_hard": {
        "description": "Push + asymétrie + biais IMU + délai (progressif)",
        "enable_push": True,
        "push_force_range": (1.2, 2.4),
        "push_interval_range": (50, 90),
        "duration": 1800,
        "motor_left_scale": 0.97,
        "motor_right_scale": 1.00,
        "control_delay_steps": 1,
        "extra_tilt_bias": float(np.deg2rad(0.0)),
        "extra_tilt_rate_bias": float(np.deg2rad(0.0)),
        "extra_noise_scale": 1.00,
        "physics_overrides": {
            "action_smoothing_alpha": 0.34,
        },
    },
}


def _build_env_kwargs(scenario: Dict, mode: str, render: bool, ppo_scale: float) -> Dict:
    """Construit les kwargs d'environnement à partir d'un scénario."""
    env_kwargs = {
        "mode": mode,
        "enable_random_push": scenario.get("enable_push", False),
        "max_episode_steps": scenario["duration"],
        "render_mode": "human" if render else None,
        "ppo_scale": scenario.get("ppo_scale", ppo_scale),
        "enable_randomized_physics": scenario.get("enable_randomized_physics", False),
        "randomization_ranges": scenario.get("randomization_ranges"),
        "motor_left_scale": scenario.get("motor_left_scale", 1.0),
        "motor_right_scale": scenario.get("motor_right_scale", 1.0),
        "control_delay_steps": scenario.get("control_delay_steps", 0),
        "extra_tilt_bias": scenario.get("extra_tilt_bias", 0.0),
        "extra_tilt_rate_bias": scenario.get("extra_tilt_rate_bias", 0.0),
        "extra_noise_scale": scenario.get("extra_noise_scale", 1.0),
    }

    if "push_force_range" in scenario:
        env_kwargs["push_force_range"] = scenario["push_force_range"]
    if "push_interval_range" in scenario:
        env_kwargs["push_interval_range"] = scenario["push_interval_range"]

    physics_overrides = scenario.get("physics_overrides")
    if physics_overrides:
        physics = PhysicsConfig()
        for key, value in physics_overrides.items():
            setattr(physics, key, value)
        env_kwargs["physics_config"] = physics

    return env_kwargs


def _estimate_ppo_authority_pct(mode: str, ppo_scale: float, scenario_name: str | None) -> float:
    """Estime l'autorité PPO max (% vs PID max) avec la formule réelle de l'env."""
    if mode != "pid+ppo":
        return 0.0

    scenario_key = scenario_name if scenario_name in SCENARIOS else "baseline"
    scenario = SCENARIOS[scenario_key]
    env_kwargs = _build_env_kwargs(scenario, mode="pid+ppo", render=False, ppo_scale=ppo_scale)
    env = SelfBalancingRobotEnv(**env_kwargs)
    try:
        return float(env.get_estimated_ppo_authority_pct())
    finally:
        env.close()


def evaluate_scenario(
    model,
    scenario_name: str,
    n_episodes: int = 5,
    render: bool = False,
    mode: str = "pid+ppo",
    ppo_scale: float = 1.0,
) -> Dict:
    """Évalue un scénario."""
    scenario = SCENARIOS[scenario_name]
    env_kwargs = _build_env_kwargs(scenario, mode=mode, render=render, ppo_scale=ppo_scale)
    env = SelfBalancingRobotEnv(**env_kwargs)
    step_duration = env.physics.timestep * env.physics.sim_steps_per_action

    recovery_tilt_threshold = float(np.deg2rad(3.0))
    recovery_tilt_rate_threshold = float(np.deg2rad(10.0))
    recovery_hold_steps = 6

    oscillation_amplitude_threshold = float(np.deg2rad(1.2))
    
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        max_tilt = 0.0
        cum_motor_effort = 0.0
        effort_sum = 0.0
        effort_samples = 0
        push_count = 0
        oscillation_count = 0
        command_asymmetry_sum = 0.0

        prev_tilt = float(obs[0])

        waiting_recovery = False
        recovery_start_step = 0
        stable_count = 0
        recovery_times_s = []
        
        done = False
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(2)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            tilt = float(obs[0])
            tilt_rate = float(obs[1])

            max_tilt = max(max_tilt, abs(tilt))
            current_effort = float(info.get("motor_effort", abs(obs[2]) + abs(obs[3])))
            cum_motor_effort += current_effort * step_duration
            effort_sum += current_effort
            effort_samples += 1
            command_asymmetry_sum += float(info.get("command_asymmetry", abs(obs[2] - obs[3])))

            if np.sign(prev_tilt) != np.sign(tilt) and max(abs(prev_tilt), abs(tilt)) > oscillation_amplitude_threshold:
                oscillation_count += 1
            prev_tilt = tilt

            if info.get("push_applied", False):
                push_count += 1
                waiting_recovery = True
                recovery_start_step = episode_steps
                stable_count = 0

            if waiting_recovery:
                is_stable = (
                    abs(tilt) <= recovery_tilt_threshold
                    and abs(tilt_rate) <= recovery_tilt_rate_threshold
                )
                stable_count = stable_count + 1 if is_stable else 0

                if stable_count >= recovery_hold_steps:
                    recovery_steps = episode_steps - recovery_start_step
                    recovery_times_s.append(recovery_steps * step_duration)
                    waiting_recovery = False
                    stable_count = 0
            
            done = terminated or truncated
        duration_s = episode_steps * step_duration
        survival_time_s = duration_s
        time_to_fall_s = duration_s if terminated else np.nan
        avg_recovery_time_s = float(np.mean(recovery_times_s)) if recovery_times_s else np.nan
        avg_effort_motor = float(effort_sum / effort_samples) if effort_samples > 0 else 0.0
        avg_command_asymmetry = float(command_asymmetry_sum / effort_samples) if effort_samples > 0 else 0.0
        
        results.append({
            "success": not terminated,  # terminated = chute
            "fell": bool(terminated),
            "steps": episode_steps,
            "duration_s": duration_s,
            "survival_time_s": survival_time_s,
            "time_to_fall_s": time_to_fall_s,
            "reward": episode_reward,
            "max_tilt_deg": np.degrees(max_tilt),
            "avg_effort_motor": avg_effort_motor,
            "cum_motor_effort": cum_motor_effort,
            "avg_recovery_time_s": avg_recovery_time_s,
            "push_count": push_count,
            "oscillation_count": oscillation_count,
            "avg_command_asymmetry": avg_command_asymmetry,
        })
    
    env.close()
    
    # Agrégation
    success_rate = sum(r["success"] for r in results) / n_episodes
    avg_reward = np.mean([r["reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_duration_s = np.mean([r["duration_s"] for r in results])
    avg_survival_time_s = np.mean([r["survival_time_s"] for r in results])
    avg_max_tilt = np.mean([r["max_tilt_deg"] for r in results])
    fall_rate = np.mean([r["fell"] for r in results])
    avg_effort_motor = np.mean([r["avg_effort_motor"] for r in results])
    avg_cum_motor_effort = np.mean([r["cum_motor_effort"] for r in results])

    fall_times = [r["time_to_fall_s"] for r in results if not np.isnan(r["time_to_fall_s"])]
    avg_time_to_fall_s = float(np.mean(fall_times)) if fall_times else avg_duration_s

    recovery_values = [r["avg_recovery_time_s"] for r in results if not np.isnan(r["avg_recovery_time_s"])]
    avg_recovery_time_s = float(np.mean(recovery_values)) if recovery_values else np.nan

    avg_push_count = np.mean([r["push_count"] for r in results])
    avg_oscillation_count = np.mean([r["oscillation_count"] for r in results])
    avg_command_asymmetry = np.mean([r["avg_command_asymmetry"] for r in results])
    
    return {
        "scenario": scenario_name,
        "description": scenario["description"],
        "n_episodes": n_episodes,
        "success_rate": success_rate,
        "fall_rate": fall_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_duration_s": avg_duration_s,
        "avg_survival_time_s": avg_survival_time_s,
        "avg_max_tilt_deg": avg_max_tilt,
        "avg_effort_motor": avg_effort_motor,
        "avg_cum_motor_effort": avg_cum_motor_effort,
        "avg_time_to_fall_s": avg_time_to_fall_s,
        "avg_recovery_time_s": avg_recovery_time_s,
        "avg_push_count": avg_push_count,
        "avg_oscillation_count": avg_oscillation_count,
        "avg_command_asymmetry": avg_command_asymmetry,
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
    parser.add_argument("--ppo-scale", type=float, default=1.0,
                        help="Coefficient d'autorité PPO en mode pid+ppo (ex: 3, 5, 8, 10)")
    parser.add_argument("--render", action="store_true",
                        help="Visualiser")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 ÉVALUATION - ROBOT AUTO-ÉQUILIBRANT")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"PPO scale: {args.ppo_scale:.2f}")
    authority_pct = _estimate_ppo_authority_pct(args.mode, args.ppo_scale, args.scenario)
    print(f"Estimated PPO authority vs PID max: {authority_pct:.1f}%")
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
            mode=args.mode,
            ppo_scale=args.ppo_scale,
        )
        results.append(result)
        
        print(f"  Succès: {result['success_rate']*100:.0f}%")
        print(f"  Chute: {result['fall_rate']*100:.0f}%")
        print(f"  Reward moyen: {result['avg_reward']:.1f}")
        print(f"  Steps moyen: {result['avg_steps']:.0f}")
        print(f"  Durée moyenne: {result['avg_duration_s']:.2f}s")
        print(f"  Temps moyen de survie: {result['avg_survival_time_s']:.2f}s")
        print(f"  Temps moyen avant chute: {result['avg_time_to_fall_s']:.2f}s")
        print(f"  Tilt max moyen: {result['avg_max_tilt_deg']:.1f}°")
        print(f"  Effort moteur moyen: {result['avg_effort_motor']:.4f}")
        print(f"  Effort moteur cumulé: {result['avg_cum_motor_effort']:.2f}")

        if np.isnan(result["avg_recovery_time_s"]):
            print("  Récupération après push: n/a")
        else:
            print(f"  Temps moyen de récupération: {result['avg_recovery_time_s']:.2f}s")

        print(f"  Push moyens/épisode: {result['avg_push_count']:.1f}")
        print(f"  Oscillations moyennes/épisode: {result['avg_oscillation_count']:.1f}")
        print(f"  Dissymétrie commande moyenne: {result['avg_command_asymmetry']:.4f}")
    
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
