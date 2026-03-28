"""
================================================================================
🎛️ TUNING PID EN TEMPS RÉEL
================================================================================

Visualisation avec sliders pour ajuster les coefficients PID en direct.

Usage:
    python -m src.tune_pid

Sliders:
    - Kp, Ki, Kd (inner loop - équilibre)
    - Kpos, Kvel (outer loop - position)
    - Push (appliquer une poussée manuelle)

================================================================================
"""

import time
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path

from src.pid_controller import PIDController, PIDGains


def main():
    print("="*60)
    print("🎛️  TUNING PID EN TEMPS RÉEL")
    print("="*60)
    print("Utilisez les sliders pour ajuster les coefficients!")
    print("="*60)

    # ===== INIT PYBULLET =====
    client = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)

    # Charger sol + robot
    plane = p.loadURDF("plane.urdf")
    urdf_path = str(Path(__file__).parent.parent / "assets" / "robot.urdf")
    robot = p.loadURDF(urdf_path, [0, 0, 0], [0, 0, 0, 1], useFixedBase=False)

    # Trouver les joints des roues
    left_joint = None
    right_joint = None
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        name = info[1].decode("utf-8")
        if "left_wheel" in name:
            left_joint = i
        elif "right_wheel" in name:
            right_joint = i

    # Friction
    for joint in [left_joint, right_joint]:
        p.changeDynamics(robot, joint, lateralFriction=5.0,
                         spinningFriction=0.001, rollingFriction=0.001)
    p.changeDynamics(plane, -1, lateralFriction=5.0)

    # Désactiver le contrôle de vitesse par défaut
    for joint in [left_joint, right_joint]:
        p.resetJointState(robot, joint, 0, 0)
        p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL,
                                targetVelocity=0, force=0)

    # ===== SLIDERS =====
    slider_kp   = p.addUserDebugParameter("Kp",    0.0, 50.0, 25.0)
    slider_ki   = p.addUserDebugParameter("Ki",    0.0, 20.0, 2.5)
    slider_kd   = p.addUserDebugParameter("Kd",    0.0, 10.0, 3.0)
    slider_kpos = p.addUserDebugParameter("Kpos",  0.0, 1.0,  0.15)
    slider_kvel = p.addUserDebugParameter("Kvel",  0.0, 1.0,  0.2)
    slider_push = p.addUserDebugParameter("PUSH!", -5.0, 5.0,  0.0)
    slider_max_torque = p.addUserDebugParameter("Max Torque", 0.1, 2.0, 0.5)

    # ===== INIT PID =====
    pid = PIDController(PIDGains(kp=25.0, ki=2.5, kd=3.0), output_limit=0.5)
    initial_pos = np.array([0.0, 0.0])

    # Stabiliser
    for _ in range(100):
        p.stepSimulation()

    pos, _ = p.getBasePositionAndOrientation(robot)
    initial_pos = np.array(pos[:2])

    # Texte debug
    text_id = p.addUserDebugText("", [0, 0, 0.35], textColorRGB=[1, 1, 1], textSize=1.5)
    text_id2 = p.addUserDebugText("", [0, 0, 0.30], textColorRGB=[0.5, 1, 0.5], textSize=1.2)

    last_push_val = 0.0
    step = 0
    dt = (1/240) * 4  # 60 Hz

    print("\n🚀 Simulation démarrée!\n")
    print("  🎛️  Ajustez les sliders dans la fenêtre PyBullet")
    print("  💨 Glissez 'PUSH!' à gauche/droite pour pousser")
    print("  ⏹️  Ctrl+C pour arreter\n")

    try:
        while True:
            # Lire les sliders
            kp = p.readUserDebugParameter(slider_kp)
            ki = p.readUserDebugParameter(slider_ki)
            kd = p.readUserDebugParameter(slider_kd)
            kpos = p.readUserDebugParameter(slider_kpos)
            kvel = p.readUserDebugParameter(slider_kvel)
            push_val = p.readUserDebugParameter(slider_push)
            max_torque = p.readUserDebugParameter(slider_max_torque)

            # Mettre à jour PID
            pid.g.kp = kp
            pid.g.ki = ki
            pid.g.kd = kd
            pid.output_limit = max_torque
            pid.integral_limit = max_torque * 0.5

            # Mesurer angle
            _, orn = p.getBasePositionAndOrientation(robot)
            euler = p.getEulerFromQuaternion(orn)
            roll = euler[0]

            _, ang_vel = p.getBaseVelocity(robot)
            roll_rate = ang_vel[0]

            # Position
            pos, _ = p.getBasePositionAndOrientation(robot)
            lin_vel, _ = p.getBaseVelocity(robot)
            drift = pos[0] - initial_pos[0]
            vel_x = lin_vel[0]

            # Outer loop (position)
            angle_setpoint = -kpos * drift - kvel * vel_x
            angle_setpoint = np.clip(angle_setpoint, -0.1, 0.1)

            # Inner loop (angle)
            error = roll - angle_setpoint
            if abs(error) < 0.01:
                error = 0.0

            torque_val = pid.update(error, dt, error_rate=roll_rate)
            torque = np.clip(torque_val, -max_torque, max_torque)

            # Appliquer aux moteurs
            p.setJointMotorControl2(robot, left_joint, p.TORQUE_CONTROL, force=torque)
            p.setJointMotorControl2(robot, right_joint, p.TORQUE_CONTROL, force=torque)

            # Push manuel
            if abs(push_val - last_push_val) > 0.1:
                force = push_val - last_push_val
                p.applyExternalForce(robot, -1, [force * 2, 0, 0], [0, 0, 0.12], p.LINK_FRAME)
                last_push_val = push_val

            # Simulation
            for _ in range(4):
                p.stepSimulation()

            step += 1

            # Afficher info
            pitch_deg = np.degrees(roll)
            if step % 30 == 0:
                p.addUserDebugText(
                    f"Pitch: {pitch_deg:+.1f}°  Torque: {torque:.3f}",
                    [0, 0, 0.35], textColorRGB=[1, 1, 1], textSize=1.5,
                    replaceItemUniqueId=text_id
                )
                p.addUserDebugText(
                    f"Kp={kp:.1f}  Ki={ki:.1f}  Kd={kd:.1f}  Kpos={kpos:.2f}  Kvel={kvel:.2f}",
                    [0, 0, 0.30], textColorRGB=[0.5, 1, 0.5], textSize=1.2,
                    replaceItemUniqueId=text_id2
                )

            if step % 120 == 0:
                print(f"  pitch={pitch_deg:+6.2f}° | torque={torque:+.3f} | "
                      f"Kp={kp:.1f} Ki={ki:.1f} Kd={kd:.1f} | "
                      f"Kpos={kpos:.2f} Kvel={kvel:.2f}")

            # Reset si tombé
            if abs(roll) > 0.5:
                print(f"  ❌ Chute! (pitch={pitch_deg:.1f}°) → Reset\n")
                p.resetBasePositionAndOrientation(robot, [0, 0, 0], [0, 0, 0, 1])
                p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0])
                for j in [left_joint, right_joint]:
                    p.resetJointState(robot, j, 0, 0)
                    p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL,
                                            targetVelocity=0, force=0)
                pid.reset()
                pos, _ = p.getBasePositionAndOrientation(robot)
                initial_pos = np.array(pos[:2])
                last_push_val = push_val

            time.sleep(1/60)

    except KeyboardInterrupt:
        print("\n⏹️  Arrêt")
    finally:
        p.disconnect()

    # Afficher les valeurs finales
    print("\n" + "="*60)
    print("📋 VALEURS FINALES À COPIER:")
    print("="*60)
    print(f"  PIDGains(kp={kp:.1f}, ki={ki:.1f}, kd={kd:.1f})")
    print(f"  Kpos = {kpos:.2f}")
    print(f"  Kvel = {kvel:.2f}")
    print(f"  Max Torque = {max_torque:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
