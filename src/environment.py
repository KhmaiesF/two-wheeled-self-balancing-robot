"""
================================================================================
🤖 ENVIRONNEMENT ROBOT AUTO-ÉQUILIBRANT - VERSION PROPRE
================================================================================

Environnement Gymnasium pour robot à 2 roues auto-équilibrant.

PHYSIQUE RÉALISTE:
- Robot veut maintenir angle = 0° (vertical)
- Roues tournent avec le torque NÉCESSAIRE (pas plus)
- Friction latérale élevée: pas de glissement perpendiculaire aux roues
- Friction de roulement faible: roues peuvent rouler librement
- Réponse réaliste aux perturbations externes (push)

OBSERVATIONS (8 dimensions):
- pitch: angle d'inclinaison (rad)
- pitch_rate: vitesse angulaire pitch (rad/s)
- position_x: dérive avant/arrière (m)
- velocity_x: vitesse avant/arrière (m/s)
- left_wheel_speed: vitesse roue gauche (rad/s)
- right_wheel_speed: vitesse roue droite (rad/s)
- last_torque_left: dernier couple appliqué gauche
- last_torque_right: dernier couple appliqué droit

ACTIONS (2 dimensions continues):
- Ajustement couple roue gauche [-1, 1]
- Ajustement couple roue droite [-1, 1]

MODES:
- "pid": PID seul (baseline)
- "ppo": PPO seul (from scratch)  
- "pid+ppo": PID + correction PPO (recommandé)

================================================================================
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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


@dataclass
class RewardConfig:
    """Configuration des récompenses."""
    # Objectif principal: rester vertical
    w_upright: float = 1.0
    angle_limit: float = 0.5           # rad (~28°) - chute si dépassé
    
    # Pénalités oscillation
    w_pitch_rate: float = 0.05
    
    # Pénalités dérive
    w_position: float = 0.1
    w_velocity: float = 0.02
    
    # Pénalités effort
    w_torque: float = 0.01
    w_wheel_speed: float = 0.001
    wheel_speed_soft_limit: float = 15.0  # rad/s - pénalité croissante au-delà
    
    # Bonus/malus
    fallen_penalty: float = 5.0


# ============================================================================
# ENVIRONNEMENT
# ============================================================================

class SelfBalancingRobotEnv(gym.Env):
    """
    Environnement Gymnasium pour robot auto-équilibrant.
    
    Simple, réaliste, extensible.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
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
    ):
        super().__init__()
        
        # Configuration
        self.render_mode = render_mode
        self.mode = mode
        self.max_episode_steps = max_episode_steps
        self.physics = physics_config or PhysicsConfig()
        self.rewards = reward_config or RewardConfig()
        self.pid_gains = pid_gains or PIDGains(kp=8.9, ki=14.0, kd=0.2)
        
        # Push configuration
        self.enable_random_push = enable_random_push
        self.push_force_range = push_force_range
        self.push_interval_range = push_interval_range
        
        # Espaces d'observation et d'action
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
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
        self._pid.reset()
        self._push_count = 0  # Reset compteur push
        
        # Planifier premier push
        if self.enable_random_push:
            self._schedule_next_push()
        
        obs = self._get_observation()
        info = {"mode": self.mode}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Exécute une action."""
        action = np.clip(action, -1.0, 1.0)
        
        # Obtenir état actuel
        pitch, pitch_rate = self._get_pitch_and_rate()
        
        # Calculer torque selon le mode
        if self.mode == "pid":
            torque = self._pid_control(pitch, pitch_rate)
        elif self.mode == "ppo":
            torque = action * self.physics.max_torque
        else:  # pid+ppo
            pid_torque = self._pid_control(pitch, pitch_rate)
            ppo_correction = action * (self.physics.max_torque * 0.3)  # 30% max
            torque = pid_torque + ppo_correction
        
        # Limiter le torque
        torque = np.clip(torque, -self.physics.max_torque, self.physics.max_torque)
        self._last_torque = torque.copy()
        
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
        
        info = {
            "step": self._step_count,
            "pitch": float(pitch),
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
        
        # Initialiser position initiale AVANT le PID (pour éviter NoneType)
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
            roll, roll_rate = self._get_pitch_and_rate()
            torque = self._pid_control(roll, roll_rate)
            self._apply_motor_torque(torque)
            p.stepSimulation()
        
        # Mémoriser position finale comme position initiale
        pos, _ = p.getBasePositionAndOrientation(self._robot_id)
        self._initial_pos = np.array(pos[:2])
    
    def _get_pitch_and_rate(self) -> Tuple[float, float]:
        """Retourne l'angle d'inclinaison et sa vitesse.
        
        Note: Le robot a ses roues sur l'axe X, donc il penche autour de X.
        C'est le ROLL (euler[0]), pas le pitch.
        """
        _, orn = p.getBasePositionAndOrientation(self._robot_id)
        euler = p.getEulerFromQuaternion(orn)
        # Roll = rotation autour de X (axe des roues)
        roll = euler[0]
        
        _, ang_vel = p.getBaseVelocity(self._robot_id)
        roll_rate = ang_vel[0]  # Vitesse angulaire autour de X
        
        return roll, roll_rate
    
    def _get_observation(self) -> np.ndarray:
        """Construit le vecteur d'observation."""
        pos, orn = p.getBasePositionAndOrientation(self._robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self._robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Vitesses des roues
        left_state = p.getJointState(self._robot_id, self._left_joint)
        right_state = p.getJointState(self._robot_id, self._right_joint)
        
        obs = np.array([
            euler[1],                      # pitch
            ang_vel[1],                    # pitch_rate
            pos[0] - self._initial_pos[0], # position_x (dérive)
            lin_vel[0],                    # velocity_x
            left_state[1],                 # left_wheel_speed
            right_state[1],                # right_wheel_speed
            self._last_torque[0],          # last_torque_left
            self._last_torque[1],          # last_torque_right
        ], dtype=np.float32)
        
        return obs
    
    def _pid_control(self, pitch: float, pitch_rate: float) -> np.ndarray:
        """Calcule le couple PID avec contrôle de position (cascade).
        
        Note: pitch est en fait roll (angle autour de X).
        Le signe du couple dépend de la convention de l'URDF.
        
        Contrôle en cascade:
        - Outer loop: position → angle_setpoint (pour éviter la dérive)
        - Inner loop: angle → torque (pour l'équilibre)
        """
        # Récupérer la position actuelle pour le contrôle de dérive
        pos, _ = p.getBasePositionAndOrientation(self._robot_id)
        lin_vel, _ = p.getBaseVelocity(self._robot_id)
        
        position_drift = pos[0] - self._initial_pos[0]  # Dérive en X
        velocity_x = lin_vel[0]
        
        # Contrôle de position (outer loop)
        # Si on dérive vers +X, on veut pencher vers -X (angle négatif)
        Kpos = 0.40  # Gain position
        Kvel = 0.42  # Gain vitesse
        
        angle_setpoint = -Kpos * position_drift - Kvel * velocity_x
        angle_setpoint = np.clip(angle_setpoint, -0.1, 0.1)  # Limiter à ~6°
        
        # Contrôle d'angle (inner loop)
        # Erreur: on veut angle = angle_setpoint
        error = pitch - angle_setpoint
        
        # Zone morte (deadband) pour éviter micro-corrections
        if abs(error) < 0.01:  # ~0.5° - ignore les très petites erreurs
            error = 0.0
        
        dt = self.physics.timestep * self.physics.sim_steps_per_action
        
        # Utiliser pitch_rate directement pour la dérivée (moins bruité)
        torque = self._pid.update(error, dt, error_rate=pitch_rate)
        
        # Même couple pour les deux roues (équilibrage)
        return np.array([torque, torque])
    
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
        pitch, pitch_rate = self._get_pitch_and_rate()
        pos, _ = p.getBasePositionAndOrientation(self._robot_id)
        lin_vel, _ = p.getBaseVelocity(self._robot_id)
        
        left_state = p.getJointState(self._robot_id, self._left_joint)
        right_state = p.getJointState(self._robot_id, self._right_joint)
        wheel_speeds = np.array([abs(left_state[1]), abs(right_state[1])])
        
        # Récompense pour rester vertical
        upright_reward = self.rewards.w_upright * np.exp(-5 * pitch**2)
        
        # Pénalité oscillation
        pitch_rate_penalty = self.rewards.w_pitch_rate * pitch_rate**2
        
        # Pénalité dérive position
        drift = np.sqrt(pos[0]**2 + pos[1]**2)
        position_penalty = self.rewards.w_position * drift**2
        
        # Pénalité vitesse
        velocity_penalty = self.rewards.w_velocity * (lin_vel[0]**2 + lin_vel[1]**2)
        
        # Pénalité effort moteur
        torque_penalty = self.rewards.w_torque * np.sum(self._last_torque**2)
        
        # Pénalité vitesse roues excessive
        excess = np.maximum(0, wheel_speeds - self.rewards.wheel_speed_soft_limit)
        wheel_penalty = self.rewards.w_wheel_speed * np.sum(excess**2)
        
        # Pénalité chute
        if self._is_fallen():
            fallen_penalty = self.rewards.fallen_penalty
        else:
            fallen_penalty = 0.0
        
        # Total
        reward = (
            upright_reward
            - pitch_rate_penalty
            - position_penalty
            - velocity_penalty
            - torque_penalty
            - wheel_penalty
            - fallen_penalty
        )
        
        info = {
            "reward_upright": upright_reward,
            "penalty_pitch_rate": pitch_rate_penalty,
            "penalty_position": position_penalty,
            "penalty_velocity": velocity_penalty,
            "penalty_torque": torque_penalty,
            "penalty_wheel": wheel_penalty,
        }
        
        return float(reward), info
    
    def _is_fallen(self) -> bool:
        """Vérifie si le robot est tombé."""
        pitch, _ = self._get_pitch_and_rate()
        return abs(pitch) > self.rewards.angle_limit
    
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
