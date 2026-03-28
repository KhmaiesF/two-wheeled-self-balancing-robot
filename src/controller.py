from __future__ import annotations
import numpy as np
from src.pid_controller import PIDController, PIDGains


class HybridController:
    """
    Combine PID + PPO (PPO = correcteur).
    - Mode 'pid': action = u_pid appliqué aux 2 roues (symétrique)
    - Mode 'ppo': action PPO (2 torques)
    - Mode 'pid+ppo': u = u_pid + u_rl
    """
    def __init__(
        self,
        max_torque: float,
        dt: float,
        target_pitch: float = 0.0,
        rl_correction_limit: float = 0.5,
        gains: PIDGains | None = None,
    ):
        self.max_torque = float(max_torque)
        self.dt = float(dt)
        self.target_pitch = float(target_pitch)
        self.rl_correction_limit = float(rl_correction_limit)

        self.pid = PIDController(gains or PIDGains(), output_limit=self.max_torque)

    def reset(self):
        self.pid.reset()

    def pid_action(self, pitch: float, pitch_rate: float) -> np.ndarray:
        # erreur angle (pitch vers 0)
        error = self.target_pitch - float(pitch)
        u = self.pid.update(error, self.dt)

        # option: amortissement direct sur pitch_rate (petit)
        u -= 0.1 * float(pitch_rate)

        # même commande aux 2 roues (balance)
        return np.array([u, u], dtype=np.float32)

    def combine(self, u_pid: np.ndarray, u_rl: np.ndarray) -> np.ndarray:
        u_rl = np.array(u_rl, dtype=np.float32)

        # limiter PPO à une correction (très important pour stabilité + transfert réel)
        u_rl = np.clip(u_rl, -self.rl_correction_limit, self.rl_correction_limit)

        u = u_pid + u_rl
        u = np.clip(u, -self.max_torque, self.max_torque)
        return u.astype(np.float32)
