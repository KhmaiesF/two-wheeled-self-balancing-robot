"""
Driver moteur BTS7960 via RPi.GPIO
Supporte 2 moteurs (gauche/droite) pour robot auto-équilibrant.

GPIO:
    Gauche: RPWM=12, LPWM=13, R_EN=24, L_EN=25
    Droit:  RPWM=18, LPWM=19, R_EN=26, L_EN=27
"""
import RPi.GPIO as GPIO
from typing import Dict

from config import (
    LEFT_RPWM, LEFT_LPWM,
    RIGHT_RPWM, RIGHT_LPWM,
    LEFT_R_EN, LEFT_L_EN,
    RIGHT_R_EN, RIGHT_L_EN,
    LEFT_MOTOR_INVERTED, RIGHT_MOTOR_INVERTED,
    PWM_MAX, PWM_DEADZONE,
    MAX_TORQUE_EQUIV
)

PWM_FREQ = 1000  # 1 kHz


class MotorDriverBTS7960:
    """Alias pour compatibilité avec run_pid_only.py et run_pid_ppo.py."""
    def __init__(self):
        self._driver = DualMotorDriver()

    def set_motors(self, left_speed, right_speed):
        self._driver.set_motors(left_speed, right_speed)

    def command_from_u(self, u_left, u_right, u_max=None):
        if u_max is None:
            u_max = MAX_TORQUE_EQUIV
        self._driver.command_from_u(u_left, u_right, u_max)

    def command_from_u_with_mapping(self, u_left, u_right, u_max=None):
        if u_max is None:
            u_max = MAX_TORQUE_EQUIV
        self._driver.command_from_u_with_mapping(u_left, u_right, u_max)

    def configure_mapping(self, use_advanced: bool, u_deadband: float, pwm_min: int, pwm_max: int):
        self._driver.configure_mapping(use_advanced, u_deadband, pwm_min, pwm_max)

    def set_pwm_ramp_max(self, pwm_ramp_max: int):
        self._driver.set_pwm_ramp_max(pwm_ramp_max)

    def set_raw_pwm(self, left_pwm_signed: int, right_pwm_signed: int, apply_ramp: bool = False):
        self._driver.set_raw_pwm(left_pwm_signed, right_pwm_signed, apply_ramp=apply_ramp)

    def get_last_debug(self) -> Dict[str, object]:
        return self._driver.get_last_debug()

    def stop(self):
        self._driver.stop()

    def cleanup(self):
        self._driver.cleanup()


class DualMotorDriver:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup Enable pins (HIGH = driver actif)
        self._en_pins = [LEFT_R_EN, LEFT_L_EN, RIGHT_R_EN, RIGHT_L_EN]
        for pin in self._en_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)  # Activer les drivers
        
        # Setup PWM pins
        GPIO.setup(LEFT_RPWM, GPIO.OUT)
        GPIO.setup(LEFT_LPWM, GPIO.OUT)
        GPIO.setup(RIGHT_RPWM, GPIO.OUT)
        GPIO.setup(RIGHT_LPWM, GPIO.OUT)
        
        # Créer PWM
        self._left_fwd = GPIO.PWM(LEFT_RPWM, PWM_FREQ)
        self._left_bwd = GPIO.PWM(LEFT_LPWM, PWM_FREQ)
        self._right_fwd = GPIO.PWM(RIGHT_RPWM, PWM_FREQ)
        self._right_bwd = GPIO.PWM(RIGHT_LPWM, PWM_FREQ)
        
        # Démarrer à 0
        self._left_fwd.start(0)
        self._left_bwd.start(0)
        self._right_fwd.start(0)
        self._right_bwd.start(0)

        # Mapping avancé u -> PWM (optionnel, OFF par défaut pour compatibilité)
        self.use_advanced_mapping = False
        self.u_deadband = 0.0
        self.pwm_min = PWM_DEADZONE
        self.pwm_max = PWM_MAX
        self.pwm_ramp_max = None
        self._last_left_pwm_signed = 0
        self._last_right_pwm_signed = 0
        self._last_debug = {
            "u_left": 0.0,
            "u_right": 0.0,
            "u_max": float(MAX_TORQUE_EQUIV),
            "pwm_left": 0,
            "pwm_right": 0,
            "dir_left": "STOP",
            "dir_right": "STOP",
            "reason_left": "init",
            "reason_right": "init",
        }
        self._cleaned = False
        
        print("✅ Drivers BTS7960 activés (EN via GPIO)")
    
    def set_motors(self, left_speed: float, right_speed: float):
        """
        Commande les deux moteurs.
        Valeurs entre -100 et +100 (%).
        Positif = avant, Négatif = arrière.
        """
        if LEFT_MOTOR_INVERTED:
            left_speed = -left_speed
        if RIGHT_MOTOR_INVERTED:
            right_speed = -right_speed

        self._set_single(left_speed, self._left_fwd, self._left_bwd)
        self._set_single(right_speed, self._right_fwd, self._right_bwd)
    
    def _set_single(self, speed: float, pwm_fwd, pwm_bwd):
        # Clamp
        speed = max(-100, min(100, speed))
        
        # Deadzone (convertir PWM_DEADZONE 0-255 en %)
        deadzone_pct = PWM_DEADZONE / PWM_MAX * 100
        
        if abs(speed) < deadzone_pct:
            pwm_fwd.ChangeDutyCycle(0)
            pwm_bwd.ChangeDutyCycle(0)
            return
        
        if speed > 0:
            pwm_fwd.ChangeDutyCycle(speed)
            pwm_bwd.ChangeDutyCycle(0)
        else:
            pwm_fwd.ChangeDutyCycle(0)
            pwm_bwd.ChangeDutyCycle(abs(speed))

    def _raw_to_duty(self, pwm_raw: int) -> float:
        pwm_raw = max(0, min(PWM_MAX, int(pwm_raw)))
        return (pwm_raw / float(PWM_MAX)) * 100.0

    def _apply_single_raw(self, pwm_signed: int, pwm_fwd, pwm_bwd):
        pwm_abs = max(0, min(PWM_MAX, int(abs(pwm_signed))))
        duty = self._raw_to_duty(pwm_abs)
        if pwm_signed > 0:
            pwm_fwd.ChangeDutyCycle(duty)
            pwm_bwd.ChangeDutyCycle(0)
            return "FWD", pwm_abs
        if pwm_signed < 0:
            pwm_fwd.ChangeDutyCycle(0)
            pwm_bwd.ChangeDutyCycle(duty)
            return "REV", pwm_abs
        pwm_fwd.ChangeDutyCycle(0)
        pwm_bwd.ChangeDutyCycle(0)
        return "STOP", 0

    def _map_u_to_signed_pwm(self, u: float, u_max: float):
        u = max(-u_max, min(u_max, float(u)))
        abs_u = abs(u)
        if self.use_advanced_mapping:
            if abs_u <= self.u_deadband:
                return 0, "deadband"
            den = max(1e-6, (u_max - self.u_deadband))
            u_eff = (abs_u - self.u_deadband) / den
            u_eff = max(0.0, min(1.0, u_eff))
            pwm = int(round(self.pwm_min + u_eff * (self.pwm_max - self.pwm_min)))
            pwm = max(0, min(self.pwm_max, pwm))
            return pwm if u >= 0 else -pwm, "mapped"

        # Legacy mapping (compatible ancien comportement)
        pwm = int(round((abs_u / max(1e-6, u_max)) * PWM_MAX))
        return pwm if u >= 0 else -pwm, "linear"

    def _apply_pwm_ramp(self, target_signed: int, prev_signed: int):
        if self.pwm_ramp_max is None or self.pwm_ramp_max <= 0:
            return target_signed
        delta = target_signed - prev_signed
        delta = max(-self.pwm_ramp_max, min(self.pwm_ramp_max, delta))
        return prev_signed + delta

    def configure_mapping(self, use_advanced: bool, u_deadband: float, pwm_min: int, pwm_max: int):
        self.use_advanced_mapping = bool(use_advanced)
        self.u_deadband = max(0.0, float(u_deadband))
        self.pwm_max = max(1, min(PWM_MAX, int(pwm_max)))
        self.pwm_min = max(0, min(self.pwm_max, int(pwm_min)))

    def set_pwm_ramp_max(self, pwm_ramp_max: int):
        if pwm_ramp_max is None:
            self.pwm_ramp_max = None
            return
        value = int(pwm_ramp_max)
        self.pwm_ramp_max = value if value > 0 else None

    def set_raw_pwm(self, left_pwm_signed: int, right_pwm_signed: int, apply_ramp: bool = False):
        left = int(max(-PWM_MAX, min(PWM_MAX, left_pwm_signed)))
        right = int(max(-PWM_MAX, min(PWM_MAX, right_pwm_signed)))

        # Inversion sens selon config robot
        if LEFT_MOTOR_INVERTED:
            left = -left
        if RIGHT_MOTOR_INVERTED:
            right = -right

        if apply_ramp:
            left = self._apply_pwm_ramp(left, self._last_left_pwm_signed)
            right = self._apply_pwm_ramp(right, self._last_right_pwm_signed)

        dir_left, pwm_left = self._apply_single_raw(left, self._left_fwd, self._left_bwd)
        dir_right, pwm_right = self._apply_single_raw(right, self._right_fwd, self._right_bwd)
        self._last_left_pwm_signed = left
        self._last_right_pwm_signed = right
        self._last_debug.update({
            "pwm_left": pwm_left,
            "pwm_right": pwm_right,
            "dir_left": dir_left,
            "dir_right": dir_right,
            "reason_left": "raw",
            "reason_right": "raw",
        })

    def command_from_u_with_mapping(self, u_left: float, u_right: float, u_max: float = None):
        if u_max is None:
            u_max = MAX_TORQUE_EQUIV
        u_max = max(1e-6, float(u_max))

        left_signed, reason_left = self._map_u_to_signed_pwm(u_left, u_max)
        right_signed, reason_right = self._map_u_to_signed_pwm(u_right, u_max)

        if LEFT_MOTOR_INVERTED:
            left_signed = -left_signed
        if RIGHT_MOTOR_INVERTED:
            right_signed = -right_signed

        left_signed = self._apply_pwm_ramp(left_signed, self._last_left_pwm_signed)
        right_signed = self._apply_pwm_ramp(right_signed, self._last_right_pwm_signed)

        dir_left, pwm_left = self._apply_single_raw(left_signed, self._left_fwd, self._left_bwd)
        dir_right, pwm_right = self._apply_single_raw(right_signed, self._right_fwd, self._right_bwd)

        self._last_left_pwm_signed = left_signed
        self._last_right_pwm_signed = right_signed

        self._last_debug.update({
            "u_left": float(u_left),
            "u_right": float(u_right),
            "u_max": float(u_max),
            "pwm_left": pwm_left,
            "pwm_right": pwm_right,
            "dir_left": dir_left,
            "dir_right": dir_right,
            "reason_left": reason_left,
            "reason_right": reason_right,
        })

    def get_last_debug(self) -> Dict[str, object]:
        return dict(self._last_debug)

    def command_from_u(self, u_left: float, u_right: float, u_max: float = None):
        """Convertit une commande couple logique [-u_max, u_max] en % moteur."""
        if u_max is None:
            u_max = MAX_TORQUE_EQUIV
        u_max = max(1e-6, float(u_max))

        def to_pct(u):
            u = max(-u_max, min(u_max, float(u)))
            pct = (u / u_max) * 100.0
            return pct

        self.set_motors(to_pct(u_left), to_pct(u_right))

        # Mettre à jour le debug en mode legacy
        self._last_debug.update({
            "u_left": float(u_left),
            "u_right": float(u_right),
            "u_max": float(u_max),
            "reason_left": "legacy",
            "reason_right": "legacy",
        })
    
    def stop(self):
        for pwm in [self._left_fwd, self._left_bwd, self._right_fwd, self._right_bwd]:
            if pwm is None:
                continue
            try:
                pwm.ChangeDutyCycle(0)
            except Exception:
                pass
    
    def disable(self):
        """Désactive les drivers BTS7960 (sécurité)"""
        for pin in self._en_pins:
            try:
                GPIO.output(pin, GPIO.LOW)
            except Exception:
                pass
        print("🔴 Drivers BTS7960 désactivés")
    
    def enable(self):
        """Réactive les drivers BTS7960"""
        for pin in self._en_pins:
            try:
                GPIO.output(pin, GPIO.HIGH)
            except Exception:
                pass
        print("🟢 Drivers BTS7960 activés")

    def _safe_pwm_stop(self, pwm_obj):
        if pwm_obj is None:
            return
        try:
            pwm_obj.ChangeDutyCycle(0)
        except Exception:
            pass
        try:
            pwm_obj.stop()
        except Exception:
            pass
    
    def cleanup(self):
        if self._cleaned:
            return
        self.stop()
        self.disable()  # Couper les drivers avant cleanup

        self._safe_pwm_stop(self._left_fwd)
        self._safe_pwm_stop(self._left_bwd)
        self._safe_pwm_stop(self._right_fwd)
        self._safe_pwm_stop(self._right_bwd)

        # Retirer références pour éviter __del__ tardif sur objets déjà invalidés
        self._left_fwd = None
        self._left_bwd = None
        self._right_fwd = None
        self._right_bwd = None

        try:
            GPIO.cleanup()
        except Exception:
            pass
        self._cleaned = True
