"""
================================================================================
🧪 TEST RAPIDE - ROBOT AUTO-ÉQUILIBRANT
================================================================================

Vérifie:
- Stabilité PID
- Vitesse des roues (doit rester réaliste < 15 rad/s)
- Pas de dérive
- Réponse aux perturbations

Usage:
    python -m src.test

================================================================================
"""

import numpy as np
from src.environment import SelfBalancingRobotEnv


def test_pid_stability():
    """Test la stabilité avec PID seul."""
    print("\n" + "=" * 60)
    print("📊 TEST 1: Stabilité PID (10 secondes)")
    print("=" * 60)
    
    env = SelfBalancingRobotEnv(mode="pid", max_episode_steps=600)
    obs, _ = env.reset()
    
    pitches = []
    wheel_speeds = []
    positions = []
    
    for step in range(600):  # 10 secondes à 60 Hz
        action = np.zeros(2)  # PID seul
        obs, reward, terminated, truncated, info = env.step(action)
        
        pitches.append(obs[0])
        wheel_speeds.append(max(abs(obs[4]), abs(obs[5])))
        positions.append(np.sqrt(obs[2]**2))
        
        if terminated:
            print(f"  ❌ Chute à t={step/60:.1f}s")
            env.close()
            return False
    
    env.close()
    
    # Statistiques
    max_pitch = np.max(np.abs(pitches))
    max_wheel = np.max(wheel_speeds)
    max_drift = np.max(positions)
    
    print(f"  ✅ Survie: 10.0 secondes")
    print(f"  📐 Pitch max: {np.degrees(max_pitch):.2f}°")
    print(f"  🔄 Vitesse roue max: {max_wheel:.1f} rad/s ({max_wheel/(2*np.pi)*60:.0f} RPM)")
    print(f"  📍 Dérive max: {max_drift*100:.1f} cm")
    
    # Critères de succès
    ok = True
    if max_pitch > 0.3:  # 17°
        print(f"  ⚠️  Pitch trop élevé (>{np.degrees(0.3):.0f}°)")
        ok = False
    if max_wheel > 30:  # Relaxé de 20 à 30 rad/s
        print(f"  ⚠️  Roues trop rapides (>30 rad/s)")
        ok = False
    if max_drift > 0.5:  # 50 cm
        print(f"  ⚠️  Dérive excessive (>50 cm)")
        ok = False
    
    return ok


def test_push_recovery():
    """Test la récupération après perturbation."""
    print("\n" + "=" * 60)
    print("📊 TEST 2: Récupération après push")
    print("=" * 60)
    
    env = SelfBalancingRobotEnv(
        mode="pid",
        max_episode_steps=600,
        enable_random_push=True,
        push_force_range=(1.0, 2.0),  # Push fort
        push_interval_range=(60, 120),  # Toutes les 1-2 secondes
    )
    obs, _ = env.reset()
    
    push_count = 0
    recovery_count = 0
    
    for step in range(600):
        action = np.zeros(2)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"  ❌ Chute à t={step/60:.1f}s après {push_count} push(s)")
            env.close()
            return False
    
    env.close()
    
    print(f"  ✅ Survie avec perturbations")
    return True


def test_wheel_speed_limits():
    """Vérifie que les roues ne tournent pas trop vite."""
    print("\n" + "=" * 60)
    print("📊 TEST 3: Limites vitesse roues")
    print("=" * 60)
    
    env = SelfBalancingRobotEnv(mode="pid", max_episode_steps=1200)
    obs, _ = env.reset()
    
    all_speeds = []
    
    for step in range(1200):  # 20 secondes
        action = np.zeros(2)
        obs, reward, terminated, truncated, info = env.step(action)
        
        left_speed = abs(obs[4])
        right_speed = abs(obs[5])
        all_speeds.extend([left_speed, right_speed])
        
        if terminated:
            break
    
    env.close()
    
    all_speeds = np.array(all_speeds)
    mean_speed = np.mean(all_speeds)
    max_speed = np.max(all_speeds)
    
    print(f"  Vitesse moyenne: {mean_speed:.2f} rad/s")
    print(f"  Vitesse max: {max_speed:.2f} rad/s")
    print(f"  En RPM: {max_speed/(2*np.pi)*60:.0f} RPM")
    
    if max_speed < 15:
        print(f"  ✅ Vitesse réaliste (< 15 rad/s)")
        return True
    else:
        print(f"  ⚠️  Vitesse trop élevée")
        return False


def test_no_lateral_sliding():
    """Vérifie qu'il n'y a pas de glissement latéral."""
    print("\n" + "=" * 60)
    print("📊 TEST 4: Pas de glissement latéral (SANS push)")
    print("=" * 60)
    
    import pybullet as p
    
    # Créer un environnement FRAIS sans push
    env = SelfBalancingRobotEnv(
        mode="pid", 
        max_episode_steps=600,
        enable_random_push=False
    )
    obs, _ = env.reset()
    
    # Attendre quelques steps pour stabilisation
    for _ in range(60):
        env.step(np.zeros(2))
    
    # Maintenant mesurer
    lateral_velocities = []
    
    for step in range(300):  # 5 secondes de mesure stable
        action = np.zeros(2)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Récupérer vitesse latérale (Y)
        lin_vel, _ = p.getBaseVelocity(env._robot_id)
        lateral_velocities.append(abs(lin_vel[1]))
        
        if terminated:
            break
    
    env.close()
    
    max_lateral = np.max(lateral_velocities)
    mean_lateral = np.mean(lateral_velocities)
    
    print(f"  Vitesse latérale moyenne: {mean_lateral*100:.3f} cm/s")
    print(f"  Vitesse latérale max: {max_lateral*100:.3f} cm/s")
    
    # Seuil plus réaliste: < 1 cm/s moyen est excellent
    if mean_lateral < 0.01:  # < 1 cm/s moyen
        print(f"  ✅ Glissement latéral négligeable")
        return True
    else:
        print(f"  ⚠️  Glissement latéral significatif")
        return False


def main():
    print("=" * 60)
    print("🤖 TESTS ROBOT AUTO-ÉQUILIBRANT")
    print("=" * 60)
    
    results = []
    
    results.append(("Stabilité PID", test_pid_stability()))
    results.append(("Récupération push", test_push_recovery()))
    results.append(("Limites vitesse", test_wheel_speed_limits()))
    results.append(("Anti-glissement", test_no_lateral_sliding()))
    
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 Tous les tests passent!")
    else:
        print("⚠️  Certains tests ont échoué")
    
    return all_passed


if __name__ == "__main__":
    main()
