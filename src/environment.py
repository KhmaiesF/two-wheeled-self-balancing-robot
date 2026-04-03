"""
================================================================================
🤖 ENVIRONNEMENT ROBOT AUTO-ÉQUILIBRANT - VERSION IMU-ONLY
================================================================================

Environnement Gymnasium pour robot à 2 roues auto-équilibrant.

AXE D'EQUILIBRAGE UNIQUE:
- Le robot penche autour de X dans ce projet
- tilt = euler[0]
- tilt_rate = ang_vel[0]

OBSERVATION PAR DEFAUT (4 dimensions):
- obs[0] = tilt_angle (rad)
- obs[1] = tilt_rate (rad/s)
- obs[2] = last_torque_left (Nm)
- obs[3] = last_torque_right (Nm)

OPTION (5 dimensions):
- obs[4] = angle_setpoint (rad) si include_angle_setpoint=True

OBJECTIF:
- Cohérence simulation ↔ réel (Raspberry Pi, IMU sans encodeurs)
- PPO plus simple, réaliste et robuste

================================================================================
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

from src.pid_controller import PIDController, PIDGains


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PhysicsConfig:
    """Configuration physique réaliste."""
    # Friction sol-roues
    lateral_friction: float = 5.0      # Très élevée: empêche glissement latéral
    spinning_friction: float = 0.001   # Très faible: roues tournent librement
    rolling_friction: float = 0.001    # Très faible: roues roulent bien
    
    # Moteurs
    max_torque: float = 0.5            # Nm - couple max (réduit pour vitesse réaliste)
    max_wheel_speed: float = 30.0      # rad/s - ~5 tours/sec
    
    # Simulation
    timestep: float = 1/240            # 240 Hz
    sim_steps_per_action: int = 4      # 60 Hz pour le contrôleur

    # Capteurs/commande pour réduire le mismatch sim ↔ réel
    tilt_noise_std: float = np.deg2rad(0.20)       # Bruit angle IMU
    tilt_rate_noise_std: float = np.deg2rad(0.80)  # Bruit gyro
    gyro_bias_std: float = np.deg2rad(0.30)        # Biais gyro tiré au reset
    action_smoothing_alpha: float = 0.35           # 0=très lent, 1=sans lissage


@dataclass
class RewardConfig:
    """Configuration des récompenses."""
    # Objectif principal: stabiliser l'angle
    w_upright: float = 2.0
    upright_decay: float = 8.0
    small_angle_threshold: float = np.deg2rad(4.0)
    small_angle_bonus: float = 0.8

    # Chute
    angle_limit: float = 0.5           # rad (~28°) - chute si dépassé
    fallen_penalty: float = 8.0

    # Pénalités dynamiques/effort
    w_tilt_rate: float = 0.05
    w_torque: float = 0.015
    w_torque_delta: float = 0.03  # Pénalise oscillation de commande
    w_tilt_rate_delta: float = 0.01  # Pénalise oscillation angulaire

    # Bonus/malus
    alive_bonus: float = 0.05


# ============================================================================
# ENVIRONNEMENT
# ============================================================================

class SelfBalancingRobotEnv(gym.Env):
    """
    Environnement Gymnasium pour robot auto-équilibrant.
    
    Simple, réaliste, extensible.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    BASE_OBS_LABELS = [
        "tilt_angle",
        "tilt_rate",
        "last_torque_left",
        "last_torque_right",
    ]
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        mode: str = "pid+ppo",
        max_episode_steps: int = 2000,
        physics_config: Optional[PhysicsConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        pid_gains: Optional[PIDGains] = None,
        enable_random_push: bool = False,
        push_force_range: Tuple[float, float] = (0.3, 1.2),
        push_interval_range: Tuple[int, int] = (100, 300),
        include_angle_setpoint: bool = False,
        angle_setpoint: float = 0.0,
        debug_observation: Optional[bool] = None,
    ):
        super().__init__()
        
        # Configuration
        self.render_mode = render_mode
        self.mode = mode
        self.max_episode_steps = max_episode_steps
        self.physics = physics_config or PhysicsConfig()
        self.rewards = reward_config or RewardConfig()
        self.pid_gains = pid_gains or PIDGains(kp=8.9, ki=14.0, kd=0.2)
        self.include_angle_setpoint = include_angle_setpoint
        self.angle_setpoint = float(angle_setpoint)
        self.debug_observation = (
            bool(int(os.getenv("PPO_DEBUG_OBS", "0")))
            if debug_observation is None
            else bool(debug_observation)
        )
        
        # Push configuration
        self.enable_random_push = enable_random_push
        self.push_force_range = push_force_range
        self.push_interval_range = push_interval_range

        # Observation: 4D par défaut, 5D si setpoint inclus
        obs_dim = len(self.BASE_OBS_LABELS) + int(self.include_angle_setpoint)
        self._obs_labels = self.BASE_OBS_LABELS + (
            ["angle_setpoint"] if self.include_angle_setpoint else []
        )
        
        # Espaces d'observation et d'action
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # PyBullet
        self._physics_client = None
        self._robot_id = None
        self._plane_id = None
        self._left_joint = None
        self._right_joint = None
        
        # État
        self._step_count = 0
        self._initial_pos = None
        self._last_torque = np.zeros(2)
        self._prev_torque = np.zeros(2)
        self._prev_tilt_rate_true = 0.0
        self._gyro_bias = 0.0
        self._next_push_step = 0
        self._push_count = 0  # Compteur de push
        
        # Contrôleur PID
        self._pid = PIDController(self.pid_gains, self.physics.max_torque)
        
        # URDF path
        self._urdf_path = Path(__file__).parent.parent / "assets" / "robot.urdf"
    
    # ========================================================================
    # API GYMNASIUM
    # ========================================================================
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset l'environnement."""
        super().reset(seed=seed)
        
        # Initialiser PyBullet si nécessaire
        if self._physics_client is None:
            self._init_pybullet()
        
        # Reset robot position
        self._reset_robot()
        
        # Reset état
        self._step_count = 0
        self._last_torque = np.zeros(2)
        self._prev_torque = np.zeros(2)
        self._pid.reset()
        self._push_count = 0  # Reset compteur push
        self._gyro_bias = float(self.np_random.normal(0.0, self.physics.gyro_bias_std))

        _, tilt_rate_true = self._get_tilt_and_rate(noisy=False)
        self._prev_tilt_rate_true = tilt_rate_true
        
        # Planifier premier push
        if self.enable_random_push:
            self._schedule_next_push()
        
        obs = self._get_observation()
        info = {
            "mode": self.mode,
            "obs_labels": self._obs_labels,
            "gyro_bias": float(self._gyro_bias),
        }

        if self.debug_observation:
            self._print_debug_observation(obs)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Exécute une action."""
        action = np.clip(action, -1.0, 1.0)
        
        # Obtenir état actuel
        tilt, tilt_rate = self._get_tilt_and_rate(noisy=True)
        
        # Calculer torque selon le mode
        if self.mode == "pid":
            torque_cmd = self._pid_control(tilt, tilt_rate)
        elif self.mode == "ppo":
            torque_cmd = action * self.physics.max_torque
        else:  # pid+ppo
            pid_torque = self._pid_control(tilt, tilt_rate)
            ppo_correction = action * (self.physics.max_torque * 0.3)  # 30% max
            torque_cmd = pid_torque + ppo_correction
        
        # Lissage + saturation réaliste du couple
        torque = self._smooth_torque_command(torque_cmd)
        
        # Appliquer aux moteurs
        self._apply_motor_torque(torque)
        
        # Push aléatoire si activé
        if self.enable_random_push and self._step_count >= self._next_push_step:
            self._apply_random_push()
            self._schedule_next_push()
        
        # Simulation
        for _ in range(self.physics.sim_steps_per_action):
            p.stepSimulation()
        
        self._step_count += 1
        
        # Observation et reward
        obs = self._get_observation()
        reward, reward_info = self._compute_reward()
        
        # Terminaison
        terminated = self._is_fallen()
        truncated = self._step_count >= self.max_episode_steps

        tilt_true, tilt_rate_true = self._get_tilt_and_rate(noisy=False)
        
        info = {
            "step": self._step_count,
            "tilt": float(tilt_true),
            "tilt_rate": float(tilt_rate_true),
            "angle_setpoint": float(self.angle_setpoint),
            "torque": torque.tolist(),
            **reward_info
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Rendu visuel."""
        if self.render_mode == "rgb_array":
            # Capture image
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0.5, 0.5, 0.3],
                cameraTargetPosition=[0, 0, 0.1],
                cameraUpVector=[0, 0, 1]
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=10.0
            )
            _, _, rgb, _, _ = p.getCameraImage(
                width=480, height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix
            )
            return np.array(rgb)[:, :, :3]
        return None
    
    def close(self):
        """Ferme l'environnement."""
        if self._physics_client is not None:
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass
            self._physics_client = None
    
    # ========================================================================
    # MÉTHODES INTERNES
    # ========================================================================
    
    def _init_pybullet(self):
        """Initialise PyBullet."""
        if self.render_mode == "human":
            self._physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self._physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.physics.timestep)
        
        # Charger sol
        self._plane_id = p.loadURDF("plane.urdf")
        
        # Charger robot
        # Le base_link est à l'origine, les roues sont à z=0.0375 (rayon)
        # Donc pour que les roues touchent le sol, base_link doit être à z=0
        self._robot_id = p.loadURDF(
            str(self._urdf_path),
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False
        )
        
        # Trouver les joints des roues
        for i in range(p.getNumJoints(self._robot_id)):
            info = p.getJointInfo(self._robot_id, i)
            name = info[1].decode("utf-8")
            if "left_wheel" in name:
                self._left_joint = i
            elif "right_wheel" in name:
                self._right_joint = i
        
        # Configurer friction des roues
        for joint in [self._left_joint, self._right_joint]:
            p.changeDynamics(
                self._robot_id, joint,
                lateralFriction=self.physics.lateral_friction,
                spinningFriction=self.physics.spinning_friction,
                rollingFriction=self.physics.rolling_friction
            )
        
        # Friction sol
        p.changeDynamics(
            self._plane_id, -1,
            lateralFriction=self.physics.lateral_friction
        )
    
    def _reset_robot(self):
        """Reset la position du robot."""
        # Le base_link est à l'origine, roues à z=0.0375
        # Pour que les roues touchent le sol, base_link à z=0
        p.resetBasePositionAndOrientation(
            self._robot_id,
            [0, 0, 0],
            [0, 0, 0, 1]
        )
        p.resetBaseVelocity(self._robot_id, [0, 0, 0], [0, 0, 0])
        
        # Conservé pour compatibilité et instrumentation éventuelle
        self._initial_pos = np.array([0.0, 0.0])
        
        # Reset roues
        for joint in [self._left_joint, self._right_joint]:
            p.resetJointState(self._robot_id, joint, 0, 0)
            p.setJointMotorControl2(
                self._robot_id, joint,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )
        
        # Stabiliser avec contrôle PID actif
        self._pid.reset()
        for _ in range(100):
            tilt, tilt_rate = self._get_tilt_and_rate(noisy=False)
            torque = self._pid_control(tilt, tilt_rate)
            self._apply_motor_torque(torque)
            p.stepSimulation()
        
        # Mémoriser position finale comme position initiale
        pos, _ = p.getBasePositionAndOrientation(self._robot_id)
        self._initial_pos = np.array(pos[:2])
    
    def _get_tilt_and_rate(self, noisy: bool = True) -> Tuple[float, float]:
        """Retourne tilt et tilt_rate autour de X (axe d'équilibrage)."""
        _, orn = p.getBasePositionAndOrientation(self._robot_id)
        euler = p.getEulerFromQuaternion(orn)
        tilt = euler[0]
        
        _, ang_vel = p.getBaseVelocity(self._robot_id)
        tilt_rate = ang_vel[0]

        if noisy:
            tilt += float(self.np_random.normal(0.0, self.physics.tilt_noise_std))
            tilt_rate += float(self.np_random.normal(0.0, self.physics.tilt_rate_noise_std))
            tilt_rate += self._gyro_bias

        return float(tilt), float(tilt_rate)
        
    def _get_pitch_and_rate(self) -> Tuple[float, float]:
        """Alias rétrocompatible: retourne tilt/tilt_rate sur l'axe X."""
        return self._get_tilt_and_rate(noisy=False)
    
    def _get_observation(self) -> np.ndarray:
        """Construit le vecteur d'observation."""
        tilt, tilt_rate = self._get_tilt_and_rate(noisy=True)

        obs_values = [
            tilt,
            tilt_rate,
            float(self._last_torque[0]),
            float(self._last_torque[1]),
        ]
        if self.include_angle_setpoint:
            obs_values.append(float(self.angle_setpoint))

        obs = np.array(obs_values, dtype=np.float32)
        
        return obs
    
    def _pid_control(self, tilt: float, tilt_rate: float) -> np.ndarray:
        """Calcule le couple PID d'équilibrage autour de l'axe X."""
        error = tilt - self.angle_setpoint
        
        # Zone morte (deadband) pour éviter micro-corrections
        if abs(error) < 0.01:  # ~0.5° - ignore les très petites erreurs
            error = 0.0
        
        dt = self.physics.timestep * self.physics.sim_steps_per_action
        
        torque = self._pid.update(error, dt, error_rate=tilt_rate)
        
        # Même couple pour les deux roues (équilibrage)
        return np.array([torque, torque])

    def _smooth_torque_command(self, torque_cmd: np.ndarray) -> np.ndarray:
        """Lissage first-order lag + saturation du couple moteur."""
        alpha = float(np.clip(self.physics.action_smoothing_alpha, 0.0, 1.0))
        self._prev_torque = self._last_torque.copy()

        smoothed = (1.0 - alpha) * self._last_torque + alpha * torque_cmd
        clipped = np.clip(smoothed, -self.physics.max_torque, self.physics.max_torque)
        self._last_torque = clipped.astype(np.float64)
        return self._last_torque.copy()
    
    def _apply_motor_torque(self, torque: np.ndarray):
        """Applique le couple aux moteurs."""
        # Roue gauche
        p.setJointMotorControl2(
            self._robot_id, self._left_joint,
            p.TORQUE_CONTROL,
            force=torque[0]
        )
        # Roue droite
        p.setJointMotorControl2(
            self._robot_id, self._right_joint,
            p.TORQUE_CONTROL,
            force=torque[1]
        )
    
    def _compute_reward(self) -> Tuple[float, Dict]:
        """Calcule la récompense."""
        tilt, tilt_rate = self._get_tilt_and_rate(noisy=False)
        tilt_rate_delta = tilt_rate - self._prev_tilt_rate_true
        self._prev_tilt_rate_true = tilt_rate

        # Bonus fort près de la verticale
        upright_reward = self.rewards.w_upright * np.exp(
            -self.rewards.upright_decay * (tilt ** 2)
        )
        small_angle_bonus = (
            self.rewards.small_angle_bonus
            if abs(tilt) < self.rewards.small_angle_threshold
            else 0.0
        )

        # Pénalités de dynamique/effort
        tilt_rate_penalty = self.rewards.w_tilt_rate * (tilt_rate ** 2)
        torque_penalty = self.rewards.w_torque * float(np.sum(self._last_torque ** 2))
        torque_delta_penalty = self.rewards.w_torque_delta * float(
            np.sum((self._last_torque - self._prev_torque) ** 2)
        )
        tilt_rate_delta_penalty = self.rewards.w_tilt_rate_delta * (tilt_rate_delta ** 2)

        fallen_penalty = self.rewards.fallen_penalty if self._is_fallen() else 0.0

        reward = (
            self.rewards.alive_bonus
            + upright_reward
            + small_angle_bonus
            - tilt_rate_penalty
            - torque_penalty
            - torque_delta_penalty
            - tilt_rate_delta_penalty
            - fallen_penalty
        )
        
        info = {
            "reward_alive": self.rewards.alive_bonus,
            "reward_upright": upright_reward,
            "reward_small_angle": small_angle_bonus,
            "penalty_tilt_rate": tilt_rate_penalty,
            "penalty_torque": torque_penalty,
            "penalty_torque_delta": torque_delta_penalty,
            "penalty_tilt_rate_delta": tilt_rate_delta_penalty,
            "penalty_fallen": fallen_penalty,
        }
        
        return float(reward), info
    
    def _is_fallen(self) -> bool:
        """Vérifie si le robot est tombé."""
        tilt, _ = self._get_tilt_and_rate(noisy=False)
        return abs(tilt) > self.rewards.angle_limit

    def get_observation_labels(self) -> List[str]:
        """Retourne la description de chaque dimension d'observation."""
        return list(self._obs_labels)

    def _print_debug_observation(self, obs: np.ndarray):
        """Affiche un exemple d'observation et sa signification."""
        print("\n[DEBUG OBS] Observation au reset")
        for idx, (label, value) in enumerate(zip(self._obs_labels, obs.tolist())):
            print(f"  obs[{idx}] {label:<18} = {value:+.5f}")
        print()
    
    def _schedule_next_push(self):
        """Planifie le prochain push."""
        interval = self.np_random.integers(*self.push_interval_range)
        self._next_push_step = self._step_count + interval
    
    def _apply_random_push(self):
        """Applique une force aléatoire."""
        force = self.np_random.uniform(*self.push_force_range)
        direction = self.np_random.choice([-1, 1])
        
        self._push_count += 1
        dir_str = "→" if direction > 0 else "←"
        print(f"  💨 PUSH #{self._push_count}: {force:.2f}N {dir_str} (t={self._step_count/60:.1f}s)")
        
        # Force horizontale sur le centre de masse
        p.applyExternalForce(
            self._robot_id, -1,
            forceObj=[direction * force, 0, 0],
            posObj=[0, 0, 0.12],  # Centre de masse
            flags=p.LINK_FRAME
        )


# ============================================================================
# REGISTRATION
# ============================================================================

def make_env(
    render_mode: Optional[str] = None,
    mode: str = "pid+ppo",
    **kwargs
) -> SelfBalancingRobotEnv:
    """Factory function pour créer l'environnement."""
    return SelfBalancingRobotEnv(render_mode=render_mode, mode=mode, **kwargs)
