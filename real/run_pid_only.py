"""
Contrôle PID cascade pour robot auto-équilibrant (Raspberry Pi).

Boucle interne: PID angle (pitch → torque)
Boucle externe: cascade position (drift → angle_setpoint)

Usage:
    python run_pid_only.py
    python run_pid_only.py --verbose
    python run_pid_only.py --no-cascade
"""

import time
import argparse
import math

from safety import Safety
from state_estimator import ComplementaryFilter
from motor_bts7960_pwm import MotorDriverBTS7960
from imu_reader import IMUReader

# PID local (copié de src/)
from pid_controller import PIDController, PIDGains

# === Valeurs auto-tunées ===
PID_KP = 20.0
PID_KI = 3.0
PID_KD = 0.5
KPOS = 0.15     # Gain cascade position
KVEL = 0.15     # Gain cascade vitesse
MAX_TORQUE = 0.5


def main():
    parser = argparse.ArgumentParser(description="PID cascade - robot réel")
    parser.add_argument("--verbose", action="store_true", help="Affichage debug")
    parser.add_argument("--hz", type=int, default=200, help="Fréquence contrôle")
    parser.add_argument("--no-cascade", action="store_true", help="Désactiver cascade position")
    args = parser.parse_args()

    print("=" * 60)
    print("🤖 ROBOT AUTO-ÉQUILIBRANT - PID CASCADE")
    print("=" * 60)
    print(f"PID: Kp={PID_KP}, Ki={PID_KI}, Kd={PID_KD}")
    print(f"Cascade: Kpos={KPOS}, Kvel={KVEL}" if not args.no_cascade else "Cascade: DÉSACTIVÉE")
    print(f"Fréquence: {args.hz} Hz")
    print("=" * 60)

    # Hardware
    imu = IMUReader()
    imu.calibrate_gyro(samples=1000)
    filt = ComplementaryFilter(alpha=0.98)
    safety = Safety(max_pitch_rad=0.6)
    motor = MotorDriverBTS7960()

    # PID
    pid = PIDController(PIDGains(kp=PID_KP, ki=PID_KI, kd=PID_KD), output_limit=MAX_TORQUE)

    dt = 1.0 / args.hz
    step_count = 0
    vel_x_estimate = 0.0

    print("\n🚀 Contrôle actif! CTRL+C pour arrêter.\n")

    i2c_errors = 0
    try:
        while True:
            t_start = time.perf_counter()

            # Lecture IMU (avec gestion erreur I2C)
            try:
                ax, ay, az, gx, gy, gz = imu.read_acc_gyro()
                i2c_errors = 0  # Reset compteur sur lecture réussie
            except OSError:
                i2c_errors += 1
                if i2c_errors > 50:
                    print(f"\n[FATAL] {i2c_errors} erreurs I2C consécutives. Vérifiez câblage!")
                    break
                motor.stop()
                time.sleep(0.01)
                continue
            pitch = filt.update(ax, ay, az, gy, dt)
            pitch_rate = gy

            # Sécurité
            if not safety.check(pitch):
                motor.stop()
                pid.reset()
                vel_x_estimate = 0.0
                if args.verbose:
                    print(f"[SAFETY] pitch={math.degrees(pitch):+.1f}° - ARRÊT")
                time.sleep(dt)
                continue

            # Cascade position (boucle externe)
            angle_setpoint = 0.0
            if not args.no_cascade:
                # Estimation vitesse via accélération forward
                vel_x_estimate = vel_x_estimate * 0.95 + ax * 9.81 * dt
                position_drift = vel_x_estimate * dt  # Approximation
                angle_setpoint = -KPOS * position_drift - KVEL * vel_x_estimate

            # PID angle (boucle interne)
            error = pitch - angle_setpoint
            u = pid.update(error, dt, error_rate=pitch_rate)

            # Commande moteurs (même torque gauche/droite)
            motor.command_from_u(u, u, u_max=MAX_TORQUE)

            # Debug
            if args.verbose and step_count % 100 == 0:
                print(f"[{step_count:6d}] pitch={math.degrees(pitch):+5.1f}°  "
                      f"rate={math.degrees(pitch_rate):+6.1f}°/s  "
                      f"u={u:+.3f}  setpoint={math.degrees(angle_setpoint):+.2f}°")

            step_count += 1

            # Timing précis
            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[STOP] Arrêt demandé")
    finally:
        motor.stop()
        motor.cleanup()
        print(f"[OK] Moteurs arrêtés ({step_count} steps, {i2c_errors} erreurs I2C)")


if __name__ == "__main__":
    main()
