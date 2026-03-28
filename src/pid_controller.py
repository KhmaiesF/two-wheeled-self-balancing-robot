from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PIDGains:
    kp: float = 25
    ki: float = 0.0
    kd: float = 2.0


class PIDController:
    """
    PID discret avec anti-windup simple (clamp sur intégrale + sortie).
    """
    def __init__(self, gains: PIDGains, output_limit: float):
        self.g = gains
        self.output_limit = float(output_limit)

        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

        # anti-windup (limite intégrale)
        self.integral_limit = self.output_limit * 0.5

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error: float, dt: float, error_rate: float = None) -> float:
        """Update PID avec option d'utiliser directement error_rate (moins bruyant)."""
        dt = max(1e-6, float(dt))

        if not self.initialized:
            self.prev_error = error
            self.initialized = True

        # intégrale
        self.integral += error * dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit

        # dérivée (utiliser error_rate si fourni, sinon calculer)
        if error_rate is not None:
            deriv = error_rate
        else:
            deriv = (error - self.prev_error) / dt
        self.prev_error = error

        u = self.g.kp * error + self.g.ki * self.integral + self.g.kd * deriv

        # clamp sortie
        if u > self.output_limit:
            u = self.output_limit
        elif u < -self.output_limit:
            u = -self.output_limit
        return float(u)
