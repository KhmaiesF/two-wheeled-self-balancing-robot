"""
================================================================================
🤖 CONTRÔLE RÉEL PID+PPO - Robot Auto-Équilibrant
================================================================================

Le PID fait l'équilibrage principal.
Le PPO ajoute une correction douce (30% max) pour améliorer la stabilité.

PRÉREQUIS:
    pip install numpy onnxruntime
    Modèle ONNX dans: models/ppo_robot.onnx

UTILISATION:
    python3 run_pid_ppo.py              # Mode PID+PPO
    python3 run_pid_ppo.py --pid_only   # Mode PID seul (fallback)
    python3 run_pid_ppo.py --verbose    # Affichage debug
================================================================================
"""

import time
import argparse
import math
import json
import numpy as np
from pathlib import Path
from typing import Tuple

from safety import Safety
from state_estimator import ComplementaryFilter
from motor_bts7960_pwm import MotorDriverBTS7960
from imu_reader import IMUReader
from pid_controller import PIDController, PIDGains
from config import (
    PWM_MAX,
    PWM_DEADZONE,
)

# Essayer d'importer ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("[WARN] onnxruntime non disponible → mode PID uniquement")


# === PID gains (identiques à run_pid_only.py) ===
PID_KP = 10.0
PID_KI = 0.0
PID_KD = 1.0
KPOS = 0.15
KVEL = 0.15
MAX_TORQUE = 0.5
MAX_RAMP = 0.02    # NOUVEAU: limite le changement de couple par step
PPO_RATIO = 0.05   # PPO ajoute max 5% du couple max (safe start)

# Mapping u -> PWM (réel)
U_DEADBAND_DEFAULT = 0.02
PWM_MIN_DEFAULT = 60
PWM_RAMP_MAX_DEFAULT = 15
CALIB_PATH = Path(__file__).parent / "config" / "motor_calib.json"
CATCHUP_PITCH_DEG = 5.0
CATCHUP_PITCH_RAD = math.radians(CATCHUP_PITCH_DEG)

# Timing robuste (dt mesuré clampé)
DT_MIN = 0.002
DT_MAX = 0.02

# PPO safety envelope (réel)
PPO_ALPHA = 0.2                 # u_total = u_pid + alpha * u_rl
PPO_WARMUP_SEC = 3.0            # PPO OFF au démarrage
PPO_DISABLE_PITCH_RAD = 0.15    # ~8.6°
PPO_DISABLE_RATE_RAD_S = 2.0
PPO_COOLDOWN_STEPS = 400        # ~2s à 200 Hz après safety event
PPO_FILTER_BETA = 0.9           # u_rl_filt = beta*prev + (1-beta)*new
PPO_MAX_DU = 0.002              # rate-limit par step pour u_rl filtré


class PPOInference:
    """Inférence PPO légère via ONNX Runtime."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        # Lire la taille d'entrée attendue
        input_shape = self.session.get_inputs()[0].shape
        self.input_dim = input_shape[-1] if len(input_shape) > 1 else input_shape[0]

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.reshape(1, -1).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: obs})
        return outputs[0].flatten()


def load_ppo_model(model_dir: str = "models"):
    """Charge le modèle PPO ONNX."""
    if not HAS_ONNX:
        return None

    # Chercher le modèle
    model_path = Path(model_dir) / "ppo_robot.onnx"
    if not model_path.exists():
        print(f"[WARN] Modèle ONNX non trouvé: {model_path}")
        return None

    try:
        model = PPOInference(str(model_path))
        print(f"[OK] Modèle ONNX chargé: {model_path} (input_dim={model.input_dim})")
        return model
    except Exception as e:
        print(f"[ERROR] Échec chargement modèle: {e}")
        return None


def _gyro_to_rad_s(raw_gyro_value: float, gyro_unit: str) -> Tuple[float, float]:
    """Convertit gyroscope en rad/s et retourne aussi gy_raw en deg/s pour debug.

    Returns:
      gyro_raw_deg_s, gyro_used_rad_s
    """
    gy_raw = float(raw_gyro_value)

    if gyro_unit == "deg":
        return gy_raw, math.radians(gy_raw)

    if gyro_unit == "rad":
        return math.degrees(gy_raw), gy_raw

    # auto: heuristique conservative
    # > 20 -> très probablement deg/s ; sinon considéré rad/s
    if abs(gy_raw) > 20.0:
        return gy_raw, math.radians(gy_raw)
    return math.degrees(gy_raw), gy_raw


def _build_safe_observation(
    input_dim: int,
    pitch: float,
    pitch_rate: float,
    last_u_left: float,
    last_u_right: float,
    angle_setpoint: float,
) -> np.ndarray:
    """Construit une observation fiable sans encodeurs.

    Base fiable (4D):
      [pitch, pitch_rate, last_u_left, last_u_right]

    Si le modèle attend plus (ex: 8D), on met des zéros constants
    sur les dimensions non observables en réel.
    """
    base = np.array([
        float(pitch),
        float(pitch_rate),
        float(last_u_left),
        float(last_u_right),
    ], dtype=np.float32)

    # Option 6D utile si le modèle supporte au moins 6 entrées.
    extra = np.array([
        float(angle_setpoint),
        float(last_u_right - last_u_left),
    ], dtype=np.float32)

    if input_dim <= 4:
        return base[:input_dim]

    if input_dim <= 6:
        obs6 = np.concatenate([base, extra])
        return obs6[:input_dim]

    # input_dim > 6: pad zéro pour dimensions non fiables (wheel speeds, pos/vel).
    obs = np.zeros(input_dim, dtype=np.float32)
    obs[0:4] = base
    obs[4:6] = extra
    return obs


def _load_calibration_pwm_min(default_pwm_min: int) -> int:
    if not CALIB_PATH.exists():
        return default_pwm_min
    try:
        data = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        value = int(data.get("PWM_MIN", default_pwm_min))
        return max(0, min(PWM_MAX, value))
    except Exception:
        return default_pwm_min


def _save_calibration(pwm_min: int, left_min: int, right_min: int):
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "PWM_MIN": int(pwm_min),
        "left_min": int(left_min),
        "right_min": int(right_min),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    CALIB_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ask_rotation(label: str, pwm_value: int) -> bool:
    ans = input(f"{label}: PWM={pwm_value:3d} -> rotation visible? [y/N]: ").strip().lower()
    return ans in ("y", "yes", "o", "oui")


def _calibrate_single_motor(motor: MotorDriverBTS7960, side: str, step: int = 5, pwm_stop: int = 180) -> int:
    print(f"\n[CAL] Calibration {side}")
    for pwm in range(0, pwm_stop + 1, step):
        if side == "left":
            motor.set_raw_pwm(pwm, 0, apply_ramp=False)
        else:
            motor.set_raw_pwm(0, pwm, apply_ramp=False)
        time.sleep(0.35)
        if _ask_rotation(side, pwm):
            motor.stop()
            return pwm
    motor.stop()
    return pwm_stop


def calibrate_deadzone_procedure(motor: MotorDriverBTS7960):
    print("=" * 60)
    print("CALIBRATION DEADZONE MOTEURS")
    print("Robot souleve, roues libres. Appuyer CTRL+C pour arreter.")
    print("=" * 60)
    input("Appuyer ENTREE pour commencer...")

    left_min = _calibrate_single_motor(motor, "left")
    time.sleep(0.6)
    right_min = _calibrate_single_motor(motor, "right")

    pwm_min = max(left_min, right_min)
    _save_calibration(pwm_min, left_min, right_min)

    print("\n[CAL] Resultats:")
    print(f"left_min={left_min}, right_min={right_min}, PWM_MIN(retenu)={pwm_min}")
    print(f"Sauvegarde: {CALIB_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Contrôle réel PID+PPO")
    parser.add_argument("--pid_only", action="store_true", help="Mode PID uniquement")
    parser.add_argument("--verbose", action="store_true", help="Affichage debug")
    parser.add_argument("--hz", type=int, default=200, help="Fréquence de contrôle")
    parser.add_argument("--cascade", action="store_true", help="Activer cascade position (OFF par défaut)")
    parser.add_argument("--gyro-unit", choices=["deg", "rad", "auto"], default="deg",
                        help="Unité gyro brute IMU (defaut: deg)")
    parser.add_argument("--u-deadband", type=float, default=U_DEADBAND_DEFAULT,
                        help="Deadband en commande u (|u|<=deadband => PWM=0)")
    parser.add_argument("--pwm-min", type=int, default=None,
                        help="PWM minimal hors deadzone (si absent: calibration puis fallback)")
    parser.add_argument("--pwm-max", type=int, default=PWM_MAX,
                        help="PWM max (0..255)")
    parser.add_argument("--pwm-ramp-max", type=int, default=PWM_RAMP_MAX_DEFAULT,
                        help="Slew-rate max en PWM par cycle (<=0 desactive)")
    parser.add_argument("--no-advanced-mapping", action="store_true",
                        help="Revient au mapping legacy (pour comparaison)")
    parser.add_argument("--calibrate-deadzone", action="store_true",
                        help="Calibration interactive deadzone -> config/motor_calib.json")
    args = parser.parse_args()

    pwm_min_effective = args.pwm_min
    if pwm_min_effective is None:
        pwm_min_effective = _load_calibration_pwm_min(PWM_MIN_DEFAULT)
    pwm_min_effective = max(0, min(PWM_MAX, int(pwm_min_effective)))
    pwm_max_effective = max(1, min(PWM_MAX, int(args.pwm_max)))
    if pwm_min_effective > pwm_max_effective:
        pwm_min_effective = pwm_max_effective

    print("=" * 60)
    print("🤖 ROBOT AUTO-ÉQUILIBRANT - PID + PPO")
    print("=" * 60)
    print(f"PID: Kp={PID_KP}, Ki={PID_KI}, Kd={PID_KD}")
    print(f"Cascade: Kpos={KPOS}, Kvel={KVEL}" if args.cascade else "Cascade: DÉSACTIVÉE (défaut)")
    if args.cascade:
        print("[WARN] Cascade activée sans encodeurs: drift probable (ax bruité).")

    # Hardware
    imu = IMUReader()
    imu.calibrate_gyro(samples=1000)
    filt = ComplementaryFilter(alpha=0.98)
    safety = Safety(max_pitch_rad=0.6)
    motor = MotorDriverBTS7960()

    if args.calibrate_deadzone:
        try:
            calibrate_deadzone_procedure(motor)
        finally:
            motor.stop()
            motor.cleanup()
        return

    # Configure mapping u -> PWM
    use_advanced_mapping = not args.no_advanced_mapping
    motor.configure_mapping(
        use_advanced=use_advanced_mapping,
        u_deadband=args.u_deadband,
        pwm_min=pwm_min_effective,
        pwm_max=pwm_max_effective,
    )
    # Base ramp policy: en pid_only on veut réactivité max par défaut.
    base_pwm_ramp_max = int(args.pwm_ramp_max)
    if args.pid_only and base_pwm_ramp_max == PWM_RAMP_MAX_DEFAULT:
        base_pwm_ramp_max = 0
    motor.set_pwm_ramp_max(base_pwm_ramp_max)

    # PID
    pid = PIDController(PIDGains(kp=PID_KP, ki=PID_KI, kd=PID_KD), output_limit=MAX_TORQUE)

    # PPO
    ppo_model = None
    if not args.pid_only:
        ppo_model = load_ppo_model()

    use_ppo = (ppo_model is not None)
    mode = "PID+PPO" if use_ppo else "PID seul"
    print(f"\nMode: {mode}")
    print(f"Fréquence: {args.hz} Hz")
    print(f"Gyro input unit: {args.gyro_unit}")
    print(f"Ramp: {'ON (PPO)' if use_ppo else 'OFF (pid_only)'}")
    print(f"Mapping: {'advanced' if use_advanced_mapping else 'legacy'} | U_DEADBAND={args.u_deadband:.3f} | "
          f"PWM_MIN={pwm_min_effective} | PWM_MAX={pwm_max_effective} | PWM_RAMP_BASE={base_pwm_ramp_max}")
    print(f"PPO correction max: {PPO_RATIO*100:.0f}%")
    print(f"PPO alpha: {PPO_ALPHA:.2f}, warmup: {PPO_WARMUP_SEC:.1f}s, cooldown: {PPO_COOLDOWN_STEPS} steps")
    print("=" * 60)
    print("\n🚀 Contrôle actif! CTRL+C pour arrêter.\n")

    dt_target = 1.0 / args.hz
    step_count = 0
    i2c_errors = 0
    vel_x_estimate = 0.0
    last_u_left = 0.0
    last_u_right = 0.0
    u_rl_prev = np.zeros(2, dtype=np.float32)
    u_rl_filt_prev = np.zeros(2, dtype=np.float32)
    start_time = time.time()
    ppo_cooldown_until_step = 0
    t_prev = time.perf_counter()
    last_verbose_log = t_prev
    stat_dt_sum = 0.0
    stat_dt_max = 0.0
    stat_count = 0
    stat_overrun = 0
    current_pwm_ramp_effective = base_pwm_ramp_max
    catchup_mode = False

    try:
        while True:
            t_start = time.perf_counter()
            dt_meas = t_start - t_prev
            t_prev = t_start
            dt = max(DT_MIN, min(DT_MAX, dt_meas))

            stat_dt_sum += dt_meas
            stat_dt_max = max(stat_dt_max, dt_meas)
            stat_count += 1
            if dt_meas > (1.05 * dt_target):
                stat_overrun += 1

            # Lecture IMU (avec retry I2C)
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

            gyro_raw_deg_s, gyro_for_filter_rad_s = _gyro_to_rad_s(gy, args.gyro_unit)
            # Le filtre complémentaire intègre gy*dt avec un angle en rad.
            # Donc gy DOIT être en rad/s ici.
            pitch = float(filt.update(ax, ay, az, gyro_for_filter_rad_s, dt))
            pitch_rate = gyro_for_filter_rad_s  # PID en rad/s

            # Sécurité
            if not safety.check(pitch):
                motor.stop()
                pid.reset()
                filt.reset()
                vel_x_estimate = 0.0
                u_rl_prev[:] = 0.0
                u_rl_filt_prev[:] = 0.0
                ppo_cooldown_until_step = step_count + PPO_COOLDOWN_STEPS
                if args.verbose:
                    print(f"[SAFETY] pitch={math.degrees(pitch):+.1f}° - ARRÊT")
                time.sleep(dt)
                continue

            # ── Cascade position (boucle externe) ──
            angle_setpoint = 0.0
            if args.cascade:
                vel_x_estimate = vel_x_estimate * 0.95 + ax * 9.81 * dt
                position_drift = vel_x_estimate * dt
                angle_setpoint = -KPOS * position_drift - KVEL * vel_x_estimate

            # ── PID angle (boucle interne) ──
            error = pitch - angle_setpoint
            u_pid = pid.update(error, dt, error_rate=pitch_rate)

            # ── Politique rampe PWM adaptative ──
            # Rattrapage: si inclinaison significative, enlever la rampe pour éviter le retard.
            catchup_mode = abs(pitch) > CATCHUP_PITCH_RAD
            pwm_ramp_effective = 0 if catchup_mode else base_pwm_ramp_max
            if use_advanced_mapping and pwm_ramp_effective != current_pwm_ramp_effective:
                motor.set_pwm_ramp_max(pwm_ramp_effective)
                current_pwm_ramp_effective = pwm_ramp_effective

            # ── Correction PPO ──
            ppo_reason_off = ""
            ppo_on = False
            u_rl = np.zeros(2, dtype=np.float32)
            u_rl_raw = np.zeros(2, dtype=np.float32)
            if use_ppo:
                # Gating safety/warmup/cooldown
                now = time.time()
                if now - start_time < PPO_WARMUP_SEC:
                    ppo_reason_off = "warmup"
                elif step_count < ppo_cooldown_until_step:
                    ppo_reason_off = "cooldown"
                elif abs(pitch) > PPO_DISABLE_PITCH_RAD:
                    ppo_reason_off = "pitch_gate"
                elif abs(pitch_rate) > PPO_DISABLE_RATE_RAD_S:
                    ppo_reason_off = "rate_gate"
                else:
                    ppo_on = True

                if ppo_on:
                    obs = _build_safe_observation(
                        ppo_model.input_dim,
                        pitch,
                        pitch_rate,
                        last_u_left,
                        last_u_right,
                        angle_setpoint,
                    )

                    raw_action = ppo_model.predict(obs)
                    ppo_limit = MAX_TORQUE * PPO_RATIO
                    u_rl_raw = np.clip(raw_action, -1.0, 1.0).astype(np.float32) * ppo_limit

                    # Filtre IIR anti-vibration
                    u_rl_filt = PPO_FILTER_BETA * u_rl_filt_prev + (1.0 - PPO_FILTER_BETA) * u_rl_raw

                    # Rate limit dédié PPO
                    du_rl = u_rl_filt - u_rl_prev
                    du_rl = np.clip(du_rl, -PPO_MAX_DU, PPO_MAX_DU)
                    u_rl = u_rl_prev + du_rl

                    u_rl_prev = u_rl.copy()
                    u_rl_filt_prev = u_rl_filt.copy()
                else:
                    # PPO OFF: reset filtre/correction pour repartir proprement
                    u_rl_prev[:] = 0.0
                    u_rl_filt_prev[:] = 0.0

            # ── Commande finale ──
            u_left = u_pid + PPO_ALPHA * u_rl[0]
            u_right = u_pid + PPO_ALPHA * u_rl[1]

            # Clamp total
            u_left = max(-MAX_TORQUE, min(MAX_TORQUE, u_left))
            u_right = max(-MAX_TORQUE, min(MAX_TORQUE, u_right))

            u_before_ramp_left = u_left
            u_before_ramp_right = u_right

            # ── RAMPE: limite les changements brusques ──
            if use_ppo:
                du_left = u_left - last_u_left
                du_right = u_right - last_u_right
                du_left = max(-MAX_RAMP, min(MAX_RAMP, du_left))
                du_right = max(-MAX_RAMP, min(MAX_RAMP, du_right))
                u_left = last_u_left + du_left
                u_right = last_u_right + du_right

            # Mapping avancé + rampe PWM (driver)
            if use_advanced_mapping:
                motor.command_from_u_with_mapping(u_left, u_right, u_max=MAX_TORQUE)
            else:
                motor.command_from_u(u_left, u_right, u_max=MAX_TORQUE)

            last_u_left = u_left
            last_u_right = u_right

            # Debug (max 1 ligne/s)
            now_log = time.perf_counter()
            if args.verbose and (now_log - last_verbose_log) >= 1.0:
                dt_avg = stat_dt_sum / max(1, stat_count)
                overrun_pct = 100.0 * stat_overrun / max(1, stat_count)
                ppo_state = "ON" if ppo_on else f"OFF({ppo_reason_off or 'no_model'})"
                dbg = motor.get_last_debug()
                pwm_l = int(dbg.get("pwm_left", 0))
                pwm_r = int(dbg.get("pwm_right", 0))
                dir_l = str(dbg.get("dir_left", "STOP"))
                dir_r = str(dbg.get("dir_right", "STOP"))
                reason_l = str(dbg.get("reason_left", "?"))
                reason_r = str(dbg.get("reason_right", "?"))
                deadband_note = ""
                if pwm_l == 0 and pwm_r == 0 and (reason_l == "deadband" or reason_r == "deadband"):
                    deadband_note = " deadband=ON"
                print(
                    f"[{step_count:6d}] pitch={math.degrees(pitch):+6.2f}deg/{pitch:+.3f}rad  "
                    f"gyro_raw={gyro_raw_deg_s:+7.2f}deg/s  gyro_used={pitch_rate:+.3f}rad/s  "
                    f"u_pid={u_pid:+.3f}  u_before_ramp=[{u_before_ramp_left:+.3f},{u_before_ramp_right:+.3f}]  "
                    f"u_sent=[{u_left:+.3f},{u_right:+.3f}]  "
                    f"pwm=[{pwm_l:3d},{pwm_r:3d}] dir=[{dir_l},{dir_r}] "
                    f"reason=[{reason_l},{reason_r}]{deadband_note}  PPO={ppo_state}  "
                    f"pwm_ramp={current_pwm_ramp_effective} catchup={'ON' if catchup_mode else 'OFF'}  "
                    f"dt_avg={dt_avg*1000:.2f}ms dt_max={stat_dt_max*1000:.2f}ms overrun={overrun_pct:.1f}%"
                )
                stat_dt_sum = 0.0
                stat_dt_max = 0.0
                stat_count = 0
                stat_overrun = 0
                last_verbose_log = now_log

            step_count += 1

            # Timing
            elapsed = time.perf_counter() - t_start
            sleep_time = dt_target - elapsed
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
