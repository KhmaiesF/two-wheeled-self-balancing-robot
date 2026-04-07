"""
PyBullet visualization focused on IMU-only PPO models.

Key features:
- Loads SB3 PPO model (.zip)
- Auto-loads VecNormalize stats (vecnormalize.pkl) when available
- Runs deterministic policy in PyBullet GUI

Usage:
    python -m src.visualize_imu_ppo
    python -m src.visualize_imu_ppo --model models/imu_only_v1/final_model.zip
    python -m src.visualize_imu_ppo --mode ppo --push --duration 60
    python -m src.visualize_imu_ppo --no-vecnorm
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment import SelfBalancingRobotEnv


DEFAULT_IMU_MODEL = Path("models/imu_only_v2/final_model.zip")


def _build_vecnorm_loader_env(mode: str) -> DummyVecEnv:
    """Create a tiny non-render env only used to load VecNormalize stats."""
    return DummyVecEnv([
        lambda: SelfBalancingRobotEnv(
            render_mode=None,
            mode=mode,
            enable_random_push=False,
            max_episode_steps=10,
        )
    ])


def _resolve_vecnorm_path(model_path: Path, explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None

    auto_path = model_path.with_name("vecnormalize.pkl")
    if auto_path.exists():
        return auto_path
    return None


def _load_model_and_vecnorm(
    mode: str,
    model_path: Path,
    vecnorm_path: Optional[Path],
) -> tuple[Optional[PPO], Optional[VecNormalize], Optional[DummyVecEnv]]:
    if mode == "pid":
        return None, None, None

    if not model_path.exists():
        print(f"[WARN] Model not found: {model_path}")
        print("[WARN] Falling back to zero PPO action.")
        return None, None, None

    print(f"[OK] Loading PPO model: {model_path}")
    model = PPO.load(str(model_path))

    if vecnorm_path is None:
        print("[INFO] No vecnormalize.pkl found -> using raw observations.")
        return model, None, None

    loader_env = _build_vecnorm_loader_env(mode)
    try:
        vecnorm = VecNormalize.load(str(vecnorm_path), loader_env)
        vecnorm.training = False
        vecnorm.norm_reward = False
        print(f"[OK] Loaded VecNormalize stats: {vecnorm_path}")
        return model, vecnorm, loader_env
    except Exception as exc:
        loader_env.close()
        print(f"[WARN] Failed to load VecNormalize ({exc}) -> using raw observations.")
        return model, None, None


def _policy_action(
    model: Optional[PPO],
    vecnorm: Optional[VecNormalize],
    obs: np.ndarray,
) -> np.ndarray:
    if model is None:
        return np.zeros(2, dtype=np.float32)

    if vecnorm is None:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    obs_2d = obs.reshape(1, -1)
    obs_norm = vecnorm.normalize_obs(obs_2d)
    action, _ = model.predict(obs_norm[0], deterministic=True)
    return np.asarray(action, dtype=np.float32)


def _add_or_replace_line(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    color_rgb: tuple[float, float, float],
    state: dict,
    key: str,
    line_width: float = 2.0,
) -> None:
    """Create or update a debug line in the active PyBullet GUI."""
    client_id = int(state["client_id"])
    old_id = int(state["items"].get(key, -1))
    item_id = p.addUserDebugLine(
        start_xyz.tolist(),
        end_xyz.tolist(),
        lineColorRGB=list(color_rgb),
        lineWidth=line_width,
        lifeTime=0,
        replaceItemUniqueId=old_id,
        physicsClientId=client_id,
    )
    state["items"][key] = item_id


def _add_or_replace_text(
    text: str,
    text_xyz: np.ndarray,
    color_rgb: tuple[float, float, float],
    state: dict,
    key: str,
    text_size: float = 1.2,
) -> None:
    """Create or update a debug text in the active PyBullet GUI."""
    client_id = int(state["client_id"])
    old_id = int(state["items"].get(key, -1))
    item_id = p.addUserDebugText(
        text,
        text_xyz.tolist(),
        textColorRGB=list(color_rgb),
        textSize=text_size,
        lifeTime=0,
        replaceItemUniqueId=old_id,
        physicsClientId=client_id,
    )
    state["items"][key] = item_id


def _set_default_camera(state: dict) -> None:
    """Set a camera angle that clearly shows URDF robot and world axes."""
    client_id = int(state["client_id"])
    p.resetDebugVisualizerCamera(
        cameraDistance=0.90,
        cameraYaw=50.0,
        cameraPitch=-22.0,
        cameraTargetPosition=[0.0, 0.0, 0.08],
        physicsClientId=client_id,
    )


def _draw_world_axes(state: dict, axis_length: float) -> None:
    """Draw static world-frame X/Y/Z axes at origin."""
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    x_tip = np.array([axis_length, 0.0, 0.0], dtype=np.float64)
    y_tip = np.array([0.0, axis_length, 0.0], dtype=np.float64)
    z_tip = np.array([0.0, 0.0, axis_length], dtype=np.float64)

    _add_or_replace_line(origin, x_tip, (1.0, 0.2, 0.2), state, key="world_x", line_width=3.0)
    _add_or_replace_line(origin, y_tip, (0.2, 1.0, 0.2), state, key="world_y", line_width=3.0)
    _add_or_replace_line(origin, z_tip, (0.2, 0.4, 1.0), state, key="world_z", line_width=3.0)

    _add_or_replace_text("X", x_tip + np.array([0.03, 0.0, 0.0]), (1.0, 0.2, 0.2), state, key="world_x_txt")
    _add_or_replace_text("Y", y_tip + np.array([0.0, 0.03, 0.0]), (0.2, 1.0, 0.2), state, key="world_y_txt")
    _add_or_replace_text("Z", z_tip + np.array([0.0, 0.0, 0.03]), (0.2, 0.4, 1.0), state, key="world_z_txt")


def _update_overlay(
    env: SelfBalancingRobotEnv,
    state: dict,
    obs: np.ndarray,
    info: dict,
    action: np.ndarray,
    elapsed_s: float,
    step: int,
    reward_sum: float,
    mode: str,
    show_axes: bool,
    show_hud: bool,
    axis_length: float,
) -> None:
    """Update robot-local axes and numeric HUD in the GUI."""
    robot_id = getattr(env, "_robot_id", None)
    if robot_id is None:
        return

    client_id = int(state["client_id"])
    pos, orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=client_id)
    pos = np.asarray(pos, dtype=np.float64)

    if show_axes:
        rot = p.getMatrixFromQuaternion(orn)
        x_axis = np.array([rot[0], rot[3], rot[6]], dtype=np.float64)
        y_axis = np.array([rot[1], rot[4], rot[7]], dtype=np.float64)
        z_axis = np.array([rot[2], rot[5], rot[8]], dtype=np.float64)

        origin = pos + np.array([0.0, 0.0, 0.05], dtype=np.float64)
        local_len = float(axis_length * 0.7)
        x_tip = origin + local_len * x_axis
        y_tip = origin + local_len * y_axis
        z_tip = origin + local_len * z_axis

        _add_or_replace_line(origin, x_tip, (1.0, 0.4, 0.4), state, key="robot_x", line_width=2.5)
        _add_or_replace_line(origin, y_tip, (0.4, 1.0, 0.4), state, key="robot_y", line_width=2.5)
        _add_or_replace_line(origin, z_tip, (0.4, 0.6, 1.0), state, key="robot_z", line_width=2.5)

        _add_or_replace_text("x_r", x_tip, (1.0, 0.4, 0.4), state, key="robot_x_txt")
        _add_or_replace_text("y_r", y_tip, (0.4, 1.0, 0.4), state, key="robot_y_txt")
        _add_or_replace_text("z_r", z_tip, (0.4, 0.6, 1.0), state, key="robot_z_txt")

    if show_hud:
        euler = p.getEulerFromQuaternion(orn)
        roll_deg = float(np.degrees(euler[0]))
        pitch_deg = float(np.degrees(euler[1]))
        yaw_deg = float(np.degrees(euler[2]))

        tilt = float(obs[0]) if obs.size > 0 else 0.0
        tilt_rate = float(obs[1]) if obs.size > 1 else 0.0
        last_u_left = float(obs[2]) if obs.size > 2 else 0.0
        last_u_right = float(obs[3]) if obs.size > 3 else 0.0

        avg_reward = reward_sum / max(1, step)
        effort = float(info.get("motor_effort", 0.0))

        base = pos + np.array([-0.48, 0.32, 0.48], dtype=np.float64)
        dz = np.array([0.0, 0.0, -0.06], dtype=np.float64)

        _add_or_replace_text(
            f"t={elapsed_s:6.2f}s  step={step:5d}  mode={mode}",
            base,
            (1.0, 1.0, 1.0),
            state,
            key="hud_1",
            text_size=1.15,
        )
        _add_or_replace_text(
            f"Euler deg: roll={roll_deg:+7.3f}  pitch={pitch_deg:+7.3f}  yaw={yaw_deg:+7.3f}",
            base + 1.0 * dz,
            (1.0, 0.9, 0.3),
            state,
            key="hud_2",
            text_size=1.15,
        )
        _add_or_replace_text(
            f"IMU obs: tilt={tilt:+8.5f} rad  rate={tilt_rate:+8.5f} rad/s",
            base + 2.0 * dz,
            (0.9, 1.0, 1.0),
            state,
            key="hud_3",
            text_size=1.15,
        )
        _add_or_replace_text(
            f"last_u=[{last_u_left:+7.4f}, {last_u_right:+7.4f}]  action=[{action[0]:+7.4f}, {action[1]:+7.4f}]",
            base + 3.0 * dz,
            (0.8, 1.0, 0.8),
            state,
            key="hud_4",
            text_size=1.15,
        )
        _add_or_replace_text(
            f"motor_effort={effort:7.4f}  avg_reward/step={avg_reward:+9.6f}",
            base + 4.0 * dz,
            (0.8, 0.9, 1.0),
            state,
            key="hud_5",
            text_size=1.15,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize IMU-only PPO in PyBullet")
    parser.add_argument("--mode", choices=["pid", "ppo", "pid+ppo"], default="pid+ppo")
    parser.add_argument("--model", type=str, default=str(DEFAULT_IMU_MODEL))
    parser.add_argument("--vecnorm", type=str, default=None,
                        help="Path to vecnormalize.pkl (default: auto-detect next to model)")
    parser.add_argument("--no-vecnorm", action="store_true",
                        help="Disable observation normalization even if vecnormalize.pkl exists")
    parser.add_argument("--duration", type=int, default=45, help="Simulation duration in seconds")
    parser.add_argument("--push", action="store_true", help="Enable random pushes")
    parser.add_argument("--ppo-scale", type=float, default=1.0, help="PPO authority scale in pid+ppo mode")
    parser.add_argument("--fps", type=float, default=60.0, help="Display/loop rate")
    parser.add_argument("--show-axes", action=argparse.BooleanOptionalAction, default=True,
                        help="Show world + robot local axes in GUI")
    parser.add_argument("--show-hud", action=argparse.BooleanOptionalAction, default=True,
                        help="Show numeric angle/action HUD text in GUI")
    parser.add_argument("--axis-length", type=float, default=0.30,
                        help="World axis length in meters")
    parser.add_argument("--auto-camera", action=argparse.BooleanOptionalAction, default=True,
                        help="Reset camera to a useful default view")
    args = parser.parse_args()

    model_path = Path(args.model)
    vecnorm_path = None if args.no_vecnorm else _resolve_vecnorm_path(model_path, args.vecnorm)

    print("=" * 72)
    print("PyBullet Visualization - IMU PPO")
    print("=" * 72)
    print(f"mode={args.mode}")
    print(f"duration={args.duration}s")
    print(f"push={'ON' if args.push else 'OFF'}")
    print(f"ppo_scale={args.ppo_scale:.2f}")
    print(f"model={model_path}")
    print(f"vecnorm={vecnorm_path if vecnorm_path else 'None'}")
    print("=" * 72)

    model, vecnorm, vecnorm_loader_env = _load_model_and_vecnorm(
        mode=args.mode,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
    )

    env = SelfBalancingRobotEnv(
        render_mode="human",
        mode=args.mode,
        enable_random_push=args.push,
        max_episode_steps=max(1, int(args.duration * 60)),
        ppo_scale=args.ppo_scale,
    )

    obs, info = env.reset()
    overlay_state = {
        "client_id": int(getattr(env, "_physics_client", 0)),
        "items": {},
    }
    if args.auto_camera:
        _set_default_camera(overlay_state)
    if args.show_axes:
        _draw_world_axes(overlay_state, axis_length=float(args.axis_length))

    obs_labels = info.get("obs_labels", [])
    if obs_labels:
        print(f"obs_dim={len(obs_labels)} labels={obs_labels}")

    started = time.time()
    step = 0
    reward_sum = 0.0

    loop_dt = 1.0 / max(1.0, float(args.fps))

    print("\n[RUN] Ctrl+C to stop\n")

    try:
        while True:
            action = _policy_action(model, vecnorm, obs)
            obs, reward, terminated, truncated, info = env.step(action)

            reward_sum += float(reward)
            step += 1

            elapsed = time.time() - started
            if args.show_axes or args.show_hud:
                _update_overlay(
                    env=env,
                    state=overlay_state,
                    obs=obs,
                    info=info,
                    action=action,
                    elapsed_s=elapsed,
                    step=step,
                    reward_sum=reward_sum,
                    mode=args.mode,
                    show_axes=bool(args.show_axes),
                    show_hud=bool(args.show_hud),
                    axis_length=float(args.axis_length),
                )

            if step % 60 == 0:
                pitch_deg = float(np.degrees(info.get("tilt", 0.0)))
                effort = float(info.get("motor_effort", 0.0))
                print(
                    f"t={elapsed:6.1f}s | pitch={pitch_deg:+7.3f} deg | "
                    f"action=[{action[0]:+7.3f},{action[1]:+7.3f}] | effort={effort:6.3f}"
                )

            if elapsed >= args.duration:
                print("\n[DONE] Requested duration reached.")
                break

            if terminated or truncated:
                status = "FALL" if terminated else "TRUNCATED"
                print(f"\n[{status}] episode_end step={step} t={elapsed:.2f}s")
                print(f"avg_reward_per_step={reward_sum / max(1, step):.5f}")
                obs, info = env.reset()
                if args.auto_camera:
                    _set_default_camera(overlay_state)
                reward_sum = 0.0
                step = 0
                started = time.time()

            time.sleep(loop_dt)

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
    finally:
        env.close()
        if vecnorm_loader_env is not None:
            vecnorm_loader_env.close()


if __name__ == "__main__":
    main()
