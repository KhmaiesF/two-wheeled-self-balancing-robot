"""
================================================================================
🤖 VISUALISATION ROBOT AUTO-ÉQUILIBRANT
================================================================================

Script pour visualiser le robot en temps réel avec PyBullet GUI.

Usage:
    python -m src.visualize --mode pid          # PID seul
    python -m src.visualize --mode pid+ppo      # PID + PPO
    python -m src.visualize --push              # Avec perturbations
    python -m src.visualize --duration 60       # 60 secondes

================================================================================
"""

import argparse
import time
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Visualisation robot auto-équilibrant")
    parser.add_argument("--mode", choices=["pid", "ppo", "pid+ppo"], default="pid",
                        help="Mode de contrôle")
    parser.add_argument("--duration", type=int, default=30,
                        help="Durée en secondes")
    parser.add_argument("--push", action="store_true",
                        help="Activer les perturbations aléatoires")
    parser.add_argument("--model", type=str, default=None,
                        help="Chemin vers le modèle PPO (optionnel)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 ROBOT AUTO-ÉQUILIBRANT - VISUALISATION")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Durée: {args.duration}s")
    print(f"Push: {'Oui' if args.push else 'Non'}")
    print("=" * 60)
    
    from src.environment import SelfBalancingRobotEnv
    
    # Créer environnement
    env = SelfBalancingRobotEnv(
        render_mode="human",
        mode=args.mode,
        enable_random_push=args.push,
        max_episode_steps=args.duration * 60,  # 60 Hz
    )
    
    # Charger modèle PPO si nécessaire
    model = None
    if args.mode in ["ppo", "pid+ppo"]:
        model_path = args.model
        if model_path is None:
            # Chercher le modèle par défaut
            default_path = Path("models/best_v5/final_model.zip")
            if default_path.exists():
                model_path = str(default_path)
        
        if model_path and Path(model_path).exists():
            print(f"📦 Chargement modèle: {model_path}")
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        else:
            print("⚠️  Pas de modèle PPO trouvé, utilisation d'actions nulles")
    
    # Boucle principale
    obs, info = env.reset()
    start_time = time.time()
    step_count = 0
    total_reward = 0
    
    print("\n🚀 Simulation démarrée! Appuyez sur Ctrl+C pour arrêter.\n")
    
    try:
        while True:
            # Calculer action
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(2)
            
            # Step
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception:
                # Fenêtre fermée
                break
            
            total_reward += reward
            step_count += 1
            
            # Afficher stats périodiquement
            if step_count % 60 == 0:
                elapsed = time.time() - start_time
                pitch_deg = np.degrees(info.get("pitch", 0))
                print(f"  t={elapsed:5.1f}s | pitch={pitch_deg:+6.2f}° | "
                      f"reward={total_reward/step_count:.3f}/step")
            
            # Reset si terminé
            if terminated or truncated:
                elapsed = time.time() - start_time
                print(f"\n{'❌ Chute!' if terminated else '✅ Fin normale'} "
                      f"après {elapsed:.1f}s ({step_count} steps)")
                print(f"Récompense moyenne: {total_reward/step_count:.3f}")
                
                if elapsed >= args.duration:
                    break
                
                # Reset pour continuer
                obs, info = env.reset()
                step_count = 0
                total_reward = 0
                print("\n🔄 Nouveau run...\n")
            
            # Timing réaliste (60 Hz)
            time.sleep(1/60)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Arrêt par l'utilisateur")
    
    finally:
        try:
            env.close()
        except Exception:
            pass
        print("👋 Fin de la visualisation")


if __name__ == "__main__":
    main()
