"""Numerical inverse kinematics for the G1 right arm."""

from __future__ import annotations

import mujoco
import numpy as np


def solve_position_ik(
    model: mujoco.MjModel,
    source_data: mujoco.MjData,
    site_id: int,
    arm_qpos_addresses: np.ndarray,
    arm_dof_addresses: np.ndarray,
    arm_ranges: np.ndarray,
    target_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve 3D site position with damped least-squares Jacobian IK."""
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = source_data.qpos
    ik_data.qvel[:] = 0.0

    tolerance = 1e-5
    damping = 1e-3
    max_joint_update = 0.05
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)

    for iteration in range(1, 101):
        mujoco.mj_forward(model, ik_data)
        error = target_position - ik_data.site_xpos[site_id]
        if np.linalg.norm(error) <= tolerance:
            break

        mujoco.mj_jacSite(
            model,
            ik_data,
            jacobian_position,
            jacobian_rotation,
            site_id,
        )
        jacobian = jacobian_position[:, arm_dof_addresses]
        regularized = jacobian @ jacobian.T + damping**2 * np.eye(3)
        joint_update = jacobian.T @ np.linalg.solve(regularized, error)
        joint_update = np.clip(
            joint_update,
            -max_joint_update,
            max_joint_update,
        )
        next_q = ik_data.qpos[arm_qpos_addresses] + joint_update
        ik_data.qpos[arm_qpos_addresses] = np.clip(
            next_q,
            arm_ranges[:, 0],
            arm_ranges[:, 1],
        )
    else:
        iteration = 100

    mujoco.mj_forward(model, ik_data)
    return (
        ik_data.qpos[arm_qpos_addresses].copy(),
        ik_data.site_xpos[site_id].copy(),
        iteration,
    )


def rotation_error_world(
    target_rotation: np.ndarray,
    current_rotation: np.ndarray,
) -> np.ndarray:
    """Return the axis-angle rotation taking current to target in world axes."""
    delta = target_rotation @ current_rotation.T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew_vector = np.array(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * skew_vector
    return angle / (2.0 * np.sin(angle)) * skew_vector


def world_axis_rotation(axis_index: int, angle: float) -> np.ndarray:
    """Construct a rotation matrix about one of the world coordinate axes."""
    axis = np.zeros(3, dtype=np.float64)
    axis[axis_index] = 1.0
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def solve_pose_ik(
    model: mujoco.MjModel,
    source_data: mujoco.MjData,
    site_id: int,
    arm_qpos_addresses: np.ndarray,
    arm_dof_addresses: np.ndarray,
    arm_ranges: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Solve TCP position and orientation using the seven right-arm joints."""
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = source_data.qpos
    ik_data.qvel[:] = 0.0

    position_tolerance = 1e-5
    rotation_tolerance = np.deg2rad(0.01)
    damping = 1e-3
    max_joint_update = 0.03
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)

    for iteration in range(1, 151):
        mujoco.mj_forward(model, ik_data)
        current_position = ik_data.site_xpos[site_id]
        current_rotation = ik_data.site_xmat[site_id].reshape(3, 3)
        position_error = target_position - current_position
        orientation_error = rotation_error_world(target_rotation, current_rotation)
        if (
            np.linalg.norm(position_error) <= position_tolerance
            and np.linalg.norm(orientation_error) <= rotation_tolerance
        ):
            break

        mujoco.mj_jacSite(
            model,
            ik_data,
            jacobian_position,
            jacobian_rotation,
            site_id,
        )
        jacobian = np.vstack(
            (
                jacobian_position[:, arm_dof_addresses],
                jacobian_rotation[:, arm_dof_addresses],
            )
        )
        error = np.concatenate((position_error, orientation_error))
        regularized = jacobian @ jacobian.T + damping**2 * np.eye(6)
        joint_update = jacobian.T @ np.linalg.solve(regularized, error)
        joint_update = np.clip(
            joint_update,
            -max_joint_update,
            max_joint_update,
        )
        next_q = ik_data.qpos[arm_qpos_addresses] + joint_update
        ik_data.qpos[arm_qpos_addresses] = np.clip(
            next_q,
            arm_ranges[:, 0],
            arm_ranges[:, 1],
        )
    else:
        iteration = 150

    mujoco.mj_forward(model, ik_data)
    return (
        ik_data.qpos[arm_qpos_addresses].copy(),
        ik_data.site_xpos[site_id].copy(),
        ik_data.site_xmat[site_id].reshape(3, 3).copy(),
        iteration,
    )
