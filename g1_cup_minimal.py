"""Minimal fixed-base G1 + Dex3 white-cup MuJoCo scene.

Keyboard control supports TCP translation and world-axis roll/pitch/yaw via IK.
One-shot roll, pitch, and yaw checks are available in headless mode.
Interactive adaptive grasping and 25 Hz state/action NPZ recording are available.
Recorded demonstrations can be replayed/reviewed, and a local ACT checkpoint can
be evaluated in a single offscreen closed-loop rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

RIGHT_HAND_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

CONTROLLED_JOINTS = RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS
TCP_SITE = "right_hand_tcp"
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 848
KEYBOARD_POSITION_STEP = 0.005
KEYBOARD_YAW_STEP = np.deg2rad(5.0)
KEYBOARD_ROLL_STEP = np.deg2rad(5.0)
KEYBOARD_PITCH_STEP = np.deg2rad(2.0)
ORIENTATION_TEST_ANGLE = np.deg2rad(5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing unitree_ros/.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the checks, save RGB, and exit without opening a viewer.",
    )
    parser.add_argument(
        "--rgb-out",
        type=Path,
        default=None,
        help=(
            "Headless output PNG path. Defaults to "
            "<project-dir>/minimal_head_rgb.png."
        ),
    )
    parser.add_argument(
        "--record-dir", type=Path, default=None,
        help="Directory for interactive state/action NPZ episodes (default: <project-dir>/demonstrations_npz).",
    )
    parser.add_argument(
        "--record-fps", type=float, default=25.0,
        help="Interactive state/action recording frequency in simulation-time Hz (default: 25).",
    )
    parser.add_argument(
        "--keyboard-move-seconds", type=float, default=0.4,
        help="Seconds used to smoothly execute each queued arm key command (default: 0.4).",
    )
    parser.add_argument(
        "--cup-x-min", type=float, default=0.220,
        help="Minimum randomized cup-center world X in metres (default: 0.220).",
    )
    parser.add_argument(
        "--cup-x-max", type=float, default=0.720,
        help="Maximum randomized cup-center world X in metres (default: 0.720).",
    )
    parser.add_argument(
        "--cup-y-min", type=float, default=-0.325,
        help="Minimum randomized cup-center world Y in metres (default: -0.325).",
    )
    parser.add_argument(
        "--cup-y-max", type=float, default=0.175,
        help="Maximum randomized cup-center world Y in metres (default: 0.175).",
    )
    parser.add_argument(
        "--cup-random-seed", type=int, default=None,
        help="Optional reproducible cup-position seed; omitted uses a new seed each run.",
    )
    parser.add_argument(
        "--record-target-episodes", type=int, default=30,
        help="Interactive recording progress target (default: 30 episodes).",
    )
    parser.add_argument(
        "--replay-episode", type=Path, default=None,
        help="Replay a recorded NPZ episode in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--review-episode", type=Path, default=None,
        help="Create a synchronized side-by-side third-person/head-camera review MP4.",
    )
    parser.add_argument(
        "--act-policy", type=Path, default=None,
        help="Run one offscreen closed-loop rollout with this ACT pretrained_model directory.",
    )
    parser.add_argument(
        "--act-execution-steps", type=int, default=5,
        help="ACT actions executed before replanning at 25 Hz (default: 5).",
    )
    parser.add_argument(
        "--act-rollout-seconds", type=float, default=20.0,
        help="Maximum ACT rollout simulation time in seconds (default: 20).",
    )
    parser.add_argument(
        "--act-video-out", type=Path, default=None,
        help="ACT third-person diagnostic MP4 (default: <project-dir>/act_rollout.mp4).",
    )
    parser.add_argument(
        "--act-no-video", action="store_true",
        help="Skip third-person MP4 encoding while retaining ACT diagnostics CSV.",
    )
    parser.add_argument(
        "--act-cup-x", type=float, default=None,
        help="Fixed cup-center world X for an ACT rollout.",
    )
    parser.add_argument(
        "--act-cup-y", type=float, default=None,
        help="Fixed cup-center world Y for an ACT rollout.",
    )
    parser.add_argument(
        "--select-cup-region", action="store_true",
        help="Interactively select and save a rectangular cup randomization region.",
    )
    parser.add_argument(
        "--cup-region-file", type=Path, default=None,
        help="Cup-region JSON path (default: <project-dir>/cup_region.json).",
    )
    parser.add_argument(
        "--render-episode-rgb", type=Path, default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ik-test-axis",
        choices=("x", "y", "z"),
        default=None,
        help="Run one position-only IK test of +5 mm along this world axis.",
    )
    parser.add_argument(
        "--orientation-test-axis",
        choices=("roll", "pitch", "yaw"),
        default=None,
        help=(
            "Keep the TCP position fixed and test a +5 degree rotation "
            "about world X (roll), Y (pitch), or Z (yaw)."
        ),
    )
    parser.add_argument(
        "--control-regression",
        action="store_true",
        help="Run keyboard IK/PD regression without a viewer or RGB rendering.",
    )
    parser.add_argument(
        "--finger-test",
        action="store_true",
        help="Without rendering, move index_0 by +0.05 rad and back using PD.",
    )
    parser.add_argument(
        "--finger-debug", action="store_true",
        help="Viewer finger diagnostics: 0 selects joint, 8/7 changes target by +/-0.05 rad.",
    )
    parser.add_argument(
        "--hand-close-scan", action="store_true",
        help="Without rendering, scan hand closure in 0.05 rad steps up to 0.50 rad; stop at first contact.",
    )
    parser.add_argument(
        "--thumb-close-scan", action="store_true",
        help="Without rendering, prepare hand at 0.30 rad then scan thumb to -0.80 rad; stop at first contact.",
    )
    parser.add_argument(
        "--finger-contact-scan", action="store_true",
        help="Kinematically scan coupled index and middle joints for first cup contact without advancing physics.",
    )
    parser.add_argument(
        "--hand-hold-test", action="store_true",
        help="Without rendering, ramp three-finger closure and check a 1 s tabletop hold.",
    )
    parser.add_argument(
        "--hand-lift-test", action="store_true",
        help="Run tabletop hold, then lift TCP 5 mm over 2 s and hold 1 s without rendering.",
    )
    parser.add_argument(
        "--lift-arm-kp-scale", type=float, choices=(1.0, 2.0, 4.0), default=1.0,
        help="Lift-test right-arm Kp multiplier; Kd scales by its square root. Hand gains and torque limits unchanged.",
    )
    parser.add_argument(
        "--lift-distance-mm", type=float, choices=(5.0, 10.0), default=5.0,
        help="Lift-test TCP rise target in millimetres.",
    )
    parser.add_argument(
        "--trajectory-gif", type=Path, default=None,
        help="Save a third-person 25 FPS GIF of the tabletop hold and lift test.",
    )
    parser.add_argument(
        "--pregrasp-x-mm", type=float, default=0.0,
        help="Move the open hand forward along world +X before closing in a lift test.",
    )
    parser.add_argument(
        "--thumb-close-seconds", type=float, default=3.0,
        help="Seconds for the thumb joints to reach their hold target (default: 3.0).",
    )
    parser.add_argument(
        "--finger-close-seconds", type=float, default=2.0,
        help="Seconds for the index and middle joints to reach their hold targets (default: 2.0).",
    )
    parser.add_argument(
        "--adaptive-close", action="store_true",
        help="Close all three fingers at one speed, latch each on cup contact, then preload.",
    )
    parser.add_argument(
        "--adaptive-close-speed", type=float, default=0.20,
        help="Adaptive-close joint-target speed in rad/s (default: 0.20).",
    )
    parser.add_argument(
        "--thumb-close-ratios", type=float, nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("THUMB_0", "THUMB_1", "THUMB_2"),
        help="Adaptive thumb joint speed ratios (default: 1.0 1.0 1.0).",
    )
    parser.add_argument(
        "--adaptive-contact-force", type=float, default=0.10,
        help="Per-finger normal-force threshold in N (default: 0.10).",
    )
    parser.add_argument(
        "--adaptive-contact-seconds", type=float, default=0.02,
        help="Continuous contact time required to latch a finger (default: 0.02).",
    )
    parser.add_argument(
        "--adaptive-preload-rad", type=float, default=0.03,
        help="Extra joint travel after all fingers contact (default: 0.03 rad).",
    )
    parser.add_argument(
        "--adaptive-preload-seconds", type=float, default=0.50,
        help="Seconds used to apply adaptive preload (default: 0.50).",
    )
    parser.add_argument(
        "--lift-thumb-target", type=float, choices=(-0.58, -0.63), default=-0.58,
        help="Lift-test target for all three thumb joints. Index/middle remain +0.35 rad.",
    )
    parser.add_argument(
        "--lift-middle-target", type=float, choices=(0.35, 0.375, 0.40), default=0.35,
        help="Lift-test target for both middle-finger joints. Index remains +0.35 rad.",
    )
    parser.add_argument(
        "--lift-middle-0-target", type=float, choices=(0.35, 0.375, 0.40), default=None,
        help="Optional lift-test override for the middle-finger proximal joint.",
    )
    parser.add_argument(
        "--lift-middle-1-target", type=float, choices=(0.35, 0.375, 0.40), default=None,
        help="Optional lift-test override for the middle-finger distal joint.",
    )
    args = parser.parse_args()
    if not 1.0 <= args.record_fps <= 100.0:
        parser.error("--record-fps must be between 1 and 100 Hz")
    if not 0.05 <= args.keyboard_move_seconds <= 2.0:
        parser.error("--keyboard-move-seconds must be between 0.05 and 2.0 s")
    if not args.cup_x_min < args.cup_x_max:
        parser.error("--cup-x-min must be smaller than --cup-x-max")
    if not args.cup_y_min < args.cup_y_max:
        parser.error("--cup-y-min must be smaller than --cup-y-max")
    if not 1 <= args.record_target_episodes <= 10000:
        parser.error("--record-target-episodes must be between 1 and 10000")
    if (args.replay_episode is not None or args.review_episode is not None) and args.headless:
        parser.error("episode replay/review cannot be combined with --headless")
    if args.replay_episode is not None and args.review_episode is not None:
        parser.error("choose --replay-episode or --review-episode")
    if not 1 <= args.act_execution_steps <= 100:
        parser.error("--act-execution-steps must be between 1 and 100")
    if not 1.0 <= args.act_rollout_seconds <= 120.0:
        parser.error("--act-rollout-seconds must be between 1 and 120 s")
    if args.act_video_out is not None and args.act_policy is None:
        parser.error("--act-video-out requires --act-policy")
    if args.act_no_video and args.act_policy is None:
        parser.error("--act-no-video requires --act-policy")
    if (args.act_cup_x is None) != (args.act_cup_y is None):
        parser.error("--act-cup-x and --act-cup-y must be provided together")
    if args.act_cup_x is not None and args.act_policy is None:
        parser.error("--act-cup-x/--act-cup-y require --act-policy")
    if args.act_policy is not None and (
        args.headless
        or args.replay_episode is not None
        or args.review_episode is not None
        or args.select_cup_region
        or args.render_episode_rgb is not None
        or args.ik_test_axis is not None
        or args.orientation_test_axis is not None
        or args.control_regression
        or args.finger_test
        or args.finger_debug
        or args.hand_close_scan
        or args.thumb_close_scan
        or args.finger_contact_scan
        or args.hand_hold_test
        or args.hand_lift_test
    ):
        parser.error("--act-policy is an exclusive offscreen evaluation mode")
    if args.select_cup_region and (
        args.headless
        or args.replay_episode is not None
        or args.review_episode is not None
    ):
        parser.error("--select-cup-region requires the viewer and cannot replay")
    if args.render_episode_rgb is not None and (
        args.headless
        or args.replay_episode is not None
        or args.review_episode is not None
        or args.select_cup_region
    ):
        parser.error("--render-episode-rgb is an exclusive internal mode")
    if args.lift_arm_kp_scale != 1.0 and not args.hand_lift_test:
        parser.error("--lift-arm-kp-scale requires --hand-lift-test")
    if args.lift_distance_mm != 5.0 and not args.hand_lift_test:
        parser.error("--lift-distance-mm requires --hand-lift-test")
    if args.trajectory_gif is not None and not args.hand_lift_test:
        parser.error("--trajectory-gif requires --hand-lift-test")
    if args.pregrasp_x_mm != 0.0 and not (
        args.hand_lift_test or args.finger_contact_scan
    ):
        parser.error("--pregrasp-x-mm requires --hand-lift-test or --finger-contact-scan")
    if not 0.0 <= args.pregrasp_x_mm <= 30.0:
        parser.error("--pregrasp-x-mm must be between 0 and 30 mm")
    if not 0.25 <= args.thumb_close_seconds <= 5.0:
        parser.error("--thumb-close-seconds must be between 0.25 and 5.0")
    if not 0.25 <= args.finger_close_seconds <= 5.0:
        parser.error("--finger-close-seconds must be between 0.25 and 5.0")
    if args.adaptive_close and not (args.hand_hold_test or args.hand_lift_test):
        parser.error("--adaptive-close requires --hand-hold-test or --hand-lift-test")
    if not 0.01 <= args.adaptive_close_speed <= 1.0:
        parser.error("--adaptive-close-speed must be between 0.01 and 1.0 rad/s")
    if any(ratio <= 0.0 or ratio > 2.0 for ratio in args.thumb_close_ratios):
        parser.error("--thumb-close-ratios values must be greater than 0 and at most 2")
    if not 0.01 <= args.adaptive_contact_force <= 5.0:
        parser.error("--adaptive-contact-force must be between 0.01 and 5.0 N")
    if not 0.002 <= args.adaptive_contact_seconds <= 0.5:
        parser.error("--adaptive-contact-seconds must be between 0.002 and 0.5 s")
    if not 0.0 <= args.adaptive_preload_rad <= 0.20:
        parser.error("--adaptive-preload-rad must be between 0 and 0.20 rad")
    if not 0.05 <= args.adaptive_preload_seconds <= 2.0:
        parser.error("--adaptive-preload-seconds must be between 0.05 and 2.0 s")
    if args.lift_thumb_target != -0.58 and not args.hand_lift_test:
        parser.error("--lift-thumb-target requires --hand-lift-test")
    if args.lift_middle_target != 0.35 and not args.hand_lift_test:
        parser.error("--lift-middle-target requires --hand-lift-test")
    if (
        args.lift_middle_0_target is not None or args.lift_middle_1_target is not None
    ) and not args.hand_lift_test:
        parser.error("Middle-joint target overrides require --hand-lift-test")
    if args.hand_lift_test and args.hand_hold_test:
        parser.error("Choose --hand-lift-test or --hand-hold-test")
    if args.finger_contact_scan and (
        args.hand_hold_test or args.hand_lift_test or args.hand_close_scan
        or args.thumb_close_scan or args.finger_debug or args.finger_test
        or args.control_regression or args.ik_test_axis is not None
        or args.orientation_test_axis is not None
    ):
        parser.error("--finger-contact-scan cannot be combined with other motion tests")
    if (args.hand_hold_test or args.hand_lift_test) and (
        args.hand_close_scan or args.thumb_close_scan or args.finger_debug
        or args.finger_test or args.control_regression
        or args.ik_test_axis is not None or args.orientation_test_axis is not None
    ):
        parser.error("--hand-hold-test cannot be combined with other motion tests")
    if args.hand_close_scan and args.thumb_close_scan:
        parser.error("Choose either --hand-close-scan or --thumb-close-scan")
    if (args.hand_close_scan or args.thumb_close_scan) and (
        args.finger_debug or args.finger_test or args.control_regression
        or args.ik_test_axis is not None or args.orientation_test_axis is not None
    ):
        parser.error("Closure scans cannot be combined with other motion tests")
    if args.finger_debug and (
        args.headless or args.finger_test or args.control_regression
        or args.ik_test_axis is not None or args.orientation_test_axis is not None
    ):
        parser.error("--finger-debug requires interactive mode without other tests")
    if args.finger_test and (
        args.control_regression
        or args.ik_test_axis is not None
        or args.orientation_test_axis is not None
    ):
        parser.error("--finger-test cannot be combined with other motion tests")
    if args.control_regression and (
        args.ik_test_axis is not None or args.orientation_test_axis is not None
    ):
        parser.error("--control-regression cannot be combined with one-shot IK tests")
    return args


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


def finger_cup_contacts(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> dict[str, tuple[int, float]]:
    """Count current finger/cup contacts and sum their normal forces in N.

    Call after mj_forward or the constraint solve. This is an instantaneous
    diagnostic, not a grasp-success check or a net force vector.
    """
    cup_body_id = model.body("cup").id
    summary = {finger: (0, 0.0) for finger in ("thumb", "index", "middle")}
    wrench = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 == cup_body_id:
            other_body = body2
        elif body2 == cup_body_id:
            other_body = body1
        else:
            continue
        finger = None
        while other_body != 0:
            name = model.body(other_body).name or ""
            finger = next(
                (part for part in summary if name.startswith(f"right_hand_{part}_")),
                None,
            )
            if finger is not None:
                break
            other_body = int(model.body_parentid[other_body])
        if finger is None:
            continue
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        count, force = summary[finger]
        summary[finger] = (count + 1, force + max(0.0, float(wrench[0])))
    return summary


def finger_cup_distances(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> dict[str, tuple[float, str] | None]:
    """Nearest signed collision-geometry distance per finger, within 1 metre.

    Positive means separated, negative means overlapping. Visual-only geoms
    are excluded. Call mj_forward before querying after a state change.
    """
    cup_geom = model.geom("cup_geom").id
    result = {part: None for part in ("thumb", "index", "middle")}
    for geom_id in range(model.ngeom):
        if not (
            (model.geom_contype[geom_id] & model.geom_conaffinity[cup_geom])
            or (model.geom_contype[cup_geom] & model.geom_conaffinity[geom_id])
        ):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        link_name = model.body(body_id).name or f"body_{body_id}"
        part = None
        while body_id != 0:
            name = model.body(body_id).name or ""
            part = next((p for p in result if name.startswith(f"right_hand_{p}_")), None)
            if part is not None:
                break
            body_id = int(model.body_parentid[body_id])
        if part is None:
            continue
        distance = float(mujoco.mj_geomDistance(model, data, geom_id, cup_geom, 1.0, None))
        if not np.isfinite(distance):
            raise RuntimeError(f"Non-finite finger-cup distance for geom {geom_id}")
        if distance < 1.0 and (result[part] is None or distance < result[part][0]):
            result[part] = (distance, f"{link_name}/geom_{geom_id}")
    return result


def finger_cup_slip_speeds(
    model: mujoco.MjModel, data: mujoco.MjData, force_threshold: float = 0.01,
) -> dict[str, tuple[int, float, float, float]]:
    """Return loaded contact count and relative point speeds per finger.

    Each tuple is ``(points, force-weighted tangential speed,
    maximum tangential speed, force-weighted absolute normal speed)`` in m/s.
    The velocity is evaluated at each instantaneous MuJoCo contact point.
    """
    cup_body_id = model.body("cup").id
    raw = {part: [] for part in ("thumb", "index", "middle")}
    jac_cup = np.zeros((3, model.nv), dtype=np.float64)
    jac_finger = np.zeros((3, model.nv), dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 == cup_body_id:
            finger_body = body2
        elif body2 == cup_body_id:
            finger_body = body1
        else:
            continue
        part = None
        ancestor = finger_body
        while ancestor != 0:
            name = model.body(ancestor).name or ""
            part = next((p for p in raw if name.startswith(f"right_hand_{p}_")), None)
            if part is not None:
                break
            ancestor = int(model.body_parentid[ancestor])
        if part is None:
            continue
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        normal_force = max(0.0, float(wrench[0]))
        if normal_force <= force_threshold:
            continue
        mujoco.mj_jac(model, data, jac_cup, None, contact.pos, cup_body_id)
        mujoco.mj_jac(model, data, jac_finger, None, contact.pos, finger_body)
        relative_velocity = (jac_finger - jac_cup) @ data.qvel
        normal = contact.frame.reshape(3, 3)[0]
        normal_speed = abs(float(relative_velocity @ normal))
        tangential_velocity = relative_velocity - (relative_velocity @ normal) * normal
        tangential_speed = float(np.linalg.norm(tangential_velocity))
        raw[part].append((normal_force, tangential_speed, normal_speed))

    result = {}
    for part, values in raw.items():
        if not values:
            result[part] = (0, 0.0, 0.0, 0.0)
            continue
        samples = np.asarray(values)
        weights = samples[:, 0]
        result[part] = (
            len(values),
            float(np.average(samples[:, 1], weights=weights)),
            float(np.max(samples[:, 1])),
            float(np.average(samples[:, 2], weights=weights)),
        )
    return result


def build_model(project_dir: Path) -> mujoco.MjModel:
    source = (
        project_dir
        / "unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.xml"
    )
    if not source.is_file():
        raise FileNotFoundError(f"G1 MJCF not found: {source}")

    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("G1 MJCF has no <compiler> element")
    meshdir = compiler.get("meshdir", "meshes")
    compiler.set("meshdir", str((source.parent / meshdir).resolve()))

    world = root.find("worldbody")
    actuator = root.find("actuator")
    if world is None or actuator is None:
        raise RuntimeError("G1 MJCF must contain <worldbody> and <actuator>")

    # Keep the complete G1 visual tree, but weld the floating base and every
    # joint except the right arm and Dex3 hand at their reference poses.
    controlled = set(CONTROLLED_JOINTS)
    for parent in world.iter():
        for child in list(parent):
            if child.tag == "freejoint":
                parent.remove(child)
            elif child.tag == "joint" and child.get("name") not in controlled:
                parent.remove(child)

    for motor in list(actuator):
        if motor.get("joint") not in controlled:
            actuator.remove(motor)

    # These elements refer to the original full-model state dimensions.
    for tag in ("sensor", "keyframe"):
        for element in list(root.findall(tag)):
            root.remove(element)

    torso = next(
        (body for body in world.iter("body") if body.get("name") == "torso_link"),
        None,
    )
    if torso is None:
        raise RuntimeError("G1 MJCF has no torso_link body")

    wrist = next(
        (
            body
            for body in world.iter("body")
            if body.get("name") == "right_wrist_yaw_link"
        ),
        None,
    )
    if wrist is None:
        raise RuntimeError("G1 MJCF has no right_wrist_yaw_link body")

    # Initial tool-center-point candidate. It lies forward of the palm origin,
    # between the thumb and the index/middle fingers. Stage 3 will use this site
    # for translational Jacobian IK after its placement is visually confirmed.
    ET.SubElement(
        wrist,
        "site",
        name=TCP_SITE,
        type="sphere",
        pos="0.115 0.029 0",
        size="0.010",
        rgba="0.1 1 0.1 1",
        group="2",
    )

    # Official D435 mount translation from g1_29dof_with_hand_rev_1_0.urdf.
    # MuJoCo cameras look along local -Z with local +Y pointing upward.
    ET.SubElement(
        torso,
        "camera",
        name="head_camera",
        pos="0.0576235 0.01753 0.42987",
        xyaxes="0 -1 0 0.6896 0 0.7242",
        fovy="48",
    )

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(IMAGE_WIDTH))
    global_visual.set("offheight", str(IMAGE_HEIGHT))

    # Fixed table: its near edge is x=0.18 m from the robot origin; center
    # z=0.80 and half-height=0.03 put the top surface at z=0.83 m.
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos="0.73 -0.085 0.80",
        size="0.55 0.75 0.03",
        rgba="0.62 0.58 0.48 1",
        friction="1.0 0.01 0.001",
    )

    # Dynamic white cup. Its bottom starts 1 mm above the table and settles.
    cup = ET.SubElement(world, "body", name="cup", pos="0.36 -0.075 0.881")
    ET.SubElement(cup, "freejoint", name="cup_freejoint")
    ET.SubElement(
        cup,
        "geom",
        name="cup_geom",
        type="cylinder",
        size="0.036 0.05",
        mass="0.12",
        rgba="0.95 0.95 0.95 1",
        friction="1.2 0.01 0.001",
        condim="4",
    )
    ET.SubElement(
        cup,
        "geom",
        name="cup_inner_visual",
        type="cylinder",
        pos="0 0 0.0505",
        size="0.029 0.0005",
        density="0",
        contype="0",
        conaffinity="0",
        rgba="0.55 0.65 0.68 1",
    )

    # Hidden below the table during normal runs. Region-selection mode moves
    # and resizes this non-colliding translucent box to show the chosen area.
    region_marker = ET.SubElement(
        world,
        "body",
        name="cup_region_marker",
        mocap="true",
        pos="0 0 -1",
    )
    ET.SubElement(
        region_marker,
        "geom",
        name="cup_region_marker_geom",
        type="box",
        size="0.001 0.001 0.001",
        rgba="0.1 0.8 0.2 0.28",
        contype="0",
        conaffinity="0",
        group="2",
    )

    for geom in root.iter("geom"):
        if geom.get("contype", "1") != "0":
            geom.set("solref", "0.002 1")
            geom.set("solimp", "0.95 0.99 0.001")

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = 0.0002
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = 100
    model.opt.noslip_iterations = 10
    return model


def compress_recorded_episode(
    timestamps: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    fps: float,
    *,
    leading_seconds: float = 0.5,
    trailing_seconds: float = 1.0,
    idle_threshold_seconds: float = 0.6,
    idle_keep_seconds: float = 0.2,
) -> dict[str, np.ndarray | int]:
    """Trim an episode and shorten long runs where action and state are still."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if (
        timestamps.ndim != 1
        or states.shape != (timestamps.size, 14)
        or actions.shape != (timestamps.size, 14)
        or timestamps.size < 2
        or not np.isfinite(timestamps).all()
        or not np.isfinite(states).all()
        or not np.isfinite(actions).all()
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("invalid timestamps/state/action arrays")

    dynamic = np.zeros(timestamps.size, dtype=bool)
    action_step = np.max(np.abs(np.diff(actions, axis=0)), axis=1)
    state_step = np.max(np.abs(np.diff(states, axis=0)), axis=1)
    if not np.any(action_step > 1e-6):
        raise ValueError("episode contains no action changes")
    dynamic[1:] = (action_step > 1e-6) | (state_step > 0.01 / fps)
    changed_indices = np.flatnonzero(dynamic)

    leading_frames = max(0, round(leading_seconds * fps))
    trailing_frames = max(0, round(trailing_seconds * fps))
    start = max(0, int(changed_indices[0]) - leading_frames)
    stop = min(timestamps.size, int(changed_indices[-1]) + 1 + trailing_frames)

    changed = dynamic[start:stop]
    keep = np.ones(stop - start, dtype=bool)
    threshold_frames = max(1, round(idle_threshold_seconds * fps))
    kept_idle_frames = max(2, round(idle_keep_seconds * fps))
    index = 0
    while index < changed.size:
        if changed[index]:
            index += 1
            continue
        run_start = index
        while index < changed.size and not changed[index]:
            index += 1
        run_stop = index
        run_length = run_stop - run_start
        # Runs at either cropped boundary are the requested leading/trailing
        # context and are deliberately left intact.
        if (
            run_start > 0
            and run_stop < changed.size
            and run_length > threshold_frames
            and run_length > kept_idle_frames
        ):
            keep_left = (kept_idle_frames + 1) // 2
            keep_right = kept_idle_frames // 2
            keep[run_start + keep_left:run_stop - keep_right] = False

    original_indices = np.arange(start, stop, dtype=np.int64)[keep]
    frame_index = np.arange(original_indices.size, dtype=np.int64)
    return {
        "timestamp": frame_index.astype(np.float64) / fps,
        "sim_timestamp": timestamps[start:stop][keep],
        "frame_index": frame_index,
        "original_frame_index": original_indices,
        "observation_state": states[start:stop][keep],
        "action": actions[start:stop][keep],
        "original_frame_count": int(timestamps.size),
        "trim_start_frame": int(start),
    }


def replay_recorded_episode(
    episode_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_qpos_addresses: np.ndarray,
    cup_qpos_address: int,
) -> None:
    episode_path = episode_path.expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"Replay episode does not exist: {episode_path}")
    with np.load(episode_path, allow_pickle=False) as episode:
        required = {"timestamp", "observation_state", "cup_pose"}
        missing = sorted(required.difference(episode.files))
        if missing:
            raise ValueError(
                f"Replay episode is missing {missing}; old recordings without "
                "per-frame cup_pose cannot be replayed completely"
            )
        timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
        states = np.asarray(episode["observation_state"], dtype=np.float64)
        cup_poses = np.asarray(episode["cup_pose"], dtype=np.float64)
        fps = float(episode["fps"]) if "fps" in episode.files else 25.0
    if (
        timestamps.ndim != 1
        or timestamps.size == 0
        or states.shape != (timestamps.size, 14)
        or cup_poses.shape != (timestamps.size, 7)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(states).all()
        or not np.isfinite(cup_poses).all()
        or fps <= 0.0
    ):
        raise ValueError("Replay episode has invalid timestamps/state/cup_pose arrays")

    from mujoco import viewer as mj_viewer

    print(
        f"Replay: {episode_path}; frames={timestamps.size}, fps={fps:g}, "
        f"duration={timestamps[-1]:.3f} s"
    )
    print("Replay starts immediately and holds the final frame; close the viewer to exit.")
    with mj_viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=False
    ) as viewer:
        viewer.cam.lookat[:] = [0.33, -0.08, 0.90]
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -20
        period = 1.0 / fps
        for frame in range(timestamps.size):
            if not viewer.is_running():
                break
            frame_started = time.monotonic()
            with viewer.lock():
                data.qpos[controlled_qpos_addresses] = states[frame]
                data.qpos[cup_qpos_address:cup_qpos_address + 7] = cup_poses[frame]
                data.qvel[:] = 0.0
                data.time = float(timestamps[frame])
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(0.0, period - (time.monotonic() - frame_started)))
        if viewer.is_running():
            print("Replay finished; final frame is held.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    print("Replay viewer closed; exiting cleanly.")
    sys.stdout.flush()
    os._exit(0)


def render_episode_head_rgb(
    episode_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_qpos_addresses: np.ndarray,
    cup_qpos_address: int,
) -> Path:
    """Render frame-aligned head-camera video from a saved simulator trajectory."""
    import av
    from fractions import Fraction

    episode_path = episode_path.expanduser().resolve()
    with np.load(episode_path, allow_pickle=False) as episode:
        required = {"timestamp", "observation_state", "cup_pose", "fps"}
        missing = sorted(required.difference(episode.files))
        if missing:
            raise ValueError(f"Cannot render RGB; episode is missing {missing}")
        timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
        states = np.asarray(episode["observation_state"], dtype=np.float64)
        cup_poses = np.asarray(episode["cup_pose"], dtype=np.float64)
        fps = float(episode["fps"])
    if (
        timestamps.ndim != 1
        or timestamps.size == 0
        or states.shape != (timestamps.size, 14)
        or cup_poses.shape != (timestamps.size, 7)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(states).all()
        or not np.isfinite(cup_poses).all()
        or fps <= 0.0
    ):
        raise ValueError("Cannot render RGB; invalid timestamp/state/cup_pose arrays")

    output_path = episode_path.with_suffix(".head_rgb.mp4")
    temporary_path = output_path.with_name(output_path.stem + ".tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    container = None
    try:
        container = av.open(str(temporary_path), "w", format="mp4")
        stream = container.add_stream(
            "libx264", rate=Fraction(str(fps)).limit_denominator(1000)
        )
        stream.width = IMAGE_WIDTH
        stream.height = IMAGE_HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        camera_scene_option = mujoco.MjvOption()
        camera_scene_option.sitegroup[:] = 0
        with mujoco.Renderer(
            model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
        ) as renderer:
            for frame_index in range(timestamps.size):
                data.qpos[controlled_qpos_addresses] = states[frame_index]
                data.qpos[cup_qpos_address:cup_qpos_address + 7] = cup_poses[
                    frame_index
                ]
                data.qvel[:] = 0.0
                data.time = float(timestamps[frame_index])
                mujoco.mj_forward(model, data)
                renderer.update_scene(
                    data,
                    camera="head_camera",
                    scene_option=camera_scene_option,
                )
                rgb = renderer.render()
                if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or rgb.dtype != np.uint8:
                    raise RuntimeError(
                        f"Invalid rendered RGB frame: {rgb.shape}, {rgb.dtype}"
                    )
                video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                if (frame_index + 1) % 100 == 0:
                    print(
                        f"RGB render: {frame_index + 1}/{timestamps.size} frames",
                        flush=True,
                    )
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        temporary_path.replace(output_path)
    except Exception:
        if container is not None:
            container.close()
        temporary_path.unlink(missing_ok=True)
        raise
    print(
        f"RGB video SAVED: {output_path}; frames={timestamps.size}, "
        f"resolution={IMAGE_WIDTH}x{IMAGE_HEIGHT}, fps={fps:g}"
    )
    return output_path


def render_episode_review(
    episode_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_qpos_addresses: np.ndarray,
    cup_qpos_address: int,
) -> Path:
    """Create a synchronized third-person and head-camera review video."""
    import av
    import cv2
    from fractions import Fraction

    episode_path = episode_path.expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"Review episode does not exist: {episode_path}")
    with np.load(episode_path, allow_pickle=False) as episode:
        required = {"timestamp", "observation_state", "cup_pose", "fps"}
        missing = sorted(required.difference(episode.files))
        if missing:
            raise ValueError(f"Cannot review episode; missing {missing}")
        timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
        states = np.asarray(episode["observation_state"], dtype=np.float64)
        cup_poses = np.asarray(episode["cup_pose"], dtype=np.float64)
        fps = float(episode["fps"])
        video_name = (
            str(episode["head_rgb_video"])
            if "head_rgb_video" in episode.files
            else episode_path.with_suffix(".head_rgb.mp4").name
        )
    head_video_path = episode_path.parent / video_name
    if not head_video_path.is_file():
        raise FileNotFoundError(
            f"Head RGB video does not exist: {head_video_path}"
        )
    if (
        timestamps.ndim != 1
        or timestamps.size == 0
        or states.shape != (timestamps.size, 14)
        or cup_poses.shape != (timestamps.size, 7)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(states).all()
        or not np.isfinite(cup_poses).all()
        or fps <= 0.0
    ):
        raise ValueError("Cannot review episode; invalid trajectory arrays")

    output_path = episode_path.with_suffix(".review.mp4")
    temporary_path = output_path.with_name(output_path.stem + ".tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    head_container = av.open(str(head_video_path))
    head_stream = head_container.streams.video[0]
    if (
        head_stream.width != IMAGE_WIDTH
        or head_stream.height != IMAGE_HEIGHT
        or abs(float(head_stream.average_rate) - fps) > 1e-6
    ):
        head_container.close()
        raise ValueError(
            "Head RGB video resolution/FPS does not match the episode"
        )
    head_frames = head_container.decode(video=0)
    output_container = None
    try:
        output_container = av.open(str(temporary_path), "w", format="mp4")
        output_stream = output_container.add_stream(
            "libx264", rate=Fraction(str(fps)).limit_denominator(1000)
        )
        output_stream.width = 2 * IMAGE_WIDTH
        output_stream.height = IMAGE_HEIGHT
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"crf": "18", "preset": "medium"}

        third_person_camera = mujoco.MjvCamera()
        third_person_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        third_person_camera.lookat[:] = [0.36, -0.085, 0.90]
        third_person_camera.distance = 0.85
        third_person_camera.azimuth = -90
        third_person_camera.elevation = -20
        with mujoco.Renderer(
            model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
        ) as renderer:
            for frame_index in range(timestamps.size):
                try:
                    head_frame = next(head_frames)
                except StopIteration as exc:
                    raise ValueError(
                        f"Head RGB video ended before frame {frame_index}"
                    ) from exc
                head_rgb = head_frame.to_ndarray(format="rgb24")
                data.qpos[controlled_qpos_addresses] = states[frame_index]
                data.qpos[cup_qpos_address:cup_qpos_address + 7] = cup_poses[
                    frame_index
                ]
                data.qvel[:] = 0.0
                data.time = float(timestamps[frame_index])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=third_person_camera)
                third_rgb = renderer.render()
                combined = np.concatenate((third_rgb, head_rgb), axis=1)
                cv2.line(
                    combined,
                    (IMAGE_WIDTH, 0),
                    (IMAGE_WIDTH, IMAGE_HEIGHT - 1),
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    combined, "Third-person trajectory", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    combined, "Head camera", (IMAGE_WIDTH + 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
                video_frame = av.VideoFrame.from_ndarray(combined, format="rgb24")
                for packet in output_stream.encode(video_frame):
                    output_container.mux(packet)
                if (frame_index + 1) % 100 == 0:
                    print(
                        f"Review render: {frame_index + 1}/{timestamps.size} frames",
                        flush=True,
                    )
        if next(head_frames, None) is not None:
            raise ValueError("Head RGB video has more frames than the episode")
        for packet in output_stream.encode():
            output_container.mux(packet)
        output_container.close()
        output_container = None
        temporary_path.replace(output_path)
    except Exception:
        if output_container is not None:
            output_container.close()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        head_container.close()
    print(
        f"Review video SAVED: {output_path}; frames={timestamps.size}, "
        f"resolution={2 * IMAGE_WIDTH}x{IMAGE_HEIGHT}, fps={fps:g}"
    )
    return output_path


def act_cup_clearance(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, int]:
    """Lowest point of the oriented collision cylinder relative to table top.

    Requires current forward kinematics/contact data. Center height alone is
    insufficient: a tilted cylinder can remain supported by the table.
    """
    cup = model.geom("cup_geom").id
    table = model.geom("table").id
    axis_z = float(data.geom_xmat[cup].reshape(3, 3)[2, 2])
    radius, half_height = model.geom_size[cup, :2]
    extent_z = half_height * abs(axis_z) + radius * np.sqrt(max(0.0, 1.0 - axis_z**2))
    table_rotation = data.geom_xmat[table].reshape(3, 3)
    table_top = data.geom_xpos[table, 2] + np.abs(table_rotation[2]) @ model.geom_size[table]
    clearance = float(data.geom_xpos[cup, 2] - extent_z - table_top)
    cup_body = model.body("cup").id
    contacts = sum(
        (c.geom1 == table and model.geom_bodyid[c.geom2] == cup_body)
        or (c.geom2 == table and model.geom_bodyid[c.geom1] == cup_body)
        for c in data.contact
    )
    return clearance, int(contacts)


def run_act_policy_rollout(
    policy_path: Path,
    video_path: Path,
    execution_steps: int,
    rollout_seconds: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_target: np.ndarray,
    joint_ranges: np.ndarray,
    physics_step,
    measured_state,
    save_video: bool = True,
) -> None:
    """Run one 25 Hz ACT rollout and save a third-person diagnostic video."""
    import csv
    import torch
    from fractions import Fraction

    if save_video:
        import av

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.utils import prepare_observation_for_inference

    policy_path = policy_path.expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"ACT pretrained_model directory does not exist: {policy_path}")
    video_path = video_path.expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = video_path.with_name(video_path.stem + ".tmp.mp4")
    if save_video:
        temporary_path.unlink(missing_ok=True)

    config = ACTConfig.from_pretrained(policy_path, local_files_only=True)
    if config.chunk_size < execution_steps:
        raise ValueError(
            f"ACT execution window {execution_steps} exceeds chunk size {config.chunk_size}"
        )
    config.device = "cpu"
    # The local checkpoint already contains the trained visual backbone.
    config.pretrained_backbone_weights = None
    load_start = time.perf_counter()
    policy = ACTPolicy.from_pretrained(
        policy_path,
        config=config,
        local_files_only=True,
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    print(
        f"ACT loaded on CPU in {time.perf_counter() - load_start:.3f} s; "
        f"chunk={config.chunk_size}, execute={execution_steps} steps "
        f"({execution_steps / 25.0:.3f} s) before replanning"
    )

    control_fps = 25.0
    physics_steps_per_action = round((1.0 / control_fps) / model.opt.timestep)
    if not np.isclose(
        physics_steps_per_action * model.opt.timestep,
        1.0 / control_fps,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"MuJoCo timestep {model.opt.timestep} does not divide the 25 Hz ACT period"
        )
    max_frames = round(rollout_seconds * control_fps)
    initial_cup_position = data.body("cup").xpos.copy()
    clear_seconds = 0.0
    longest_clear_seconds = 0.0
    diagnostic_rows = []
    diagnostic_data = mujoco.MjData(model)
    success = False
    clip_frames = 0
    inference_times: list[float] = []
    action_chunk: np.ndarray | None = None
    chunk_index = execution_steps

    policy_scene_option = mujoco.MjvOption()
    policy_scene_option.sitegroup[:] = 0
    video_scene_option = mujoco.MjvOption()
    video_scene_option.sitegroup[:] = 0
    third_person_camera = mujoco.MjvCamera()
    third_person_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    third_person_camera.lookat[:] = [0.36, -0.05, 0.90]
    third_person_camera.distance = 0.85
    third_person_camera.azimuth = -90
    third_person_camera.elevation = -20

    output_container = None
    rendered_frames = 0
    try:
        output_stream = None
        if save_video:
            output_container = av.open(str(temporary_path), "w", format="mp4")
            output_stream = output_container.add_stream("libx264", rate=Fraction(25, 1))
            output_stream.width = IMAGE_WIDTH
            output_stream.height = IMAGE_HEIGHT
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"crf": "18", "preset": "medium"}

        with mujoco.Renderer(
            model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
        ) as renderer:
            for frame_index in range(max_frames):
                if chunk_index >= execution_steps:
                    renderer.update_scene(
                        data,
                        camera="head_camera",
                        scene_option=policy_scene_option,
                    )
                    head_rgb = renderer.render().copy()
                    observation = {
                        "observation.state": measured_state(),
                        "observation.images.head": head_rgb,
                    }
                    prepared = prepare_observation_for_inference(
                        observation,
                        torch.device("cpu"),
                        task="Pick up the white cup",
                        robot_type="g1_dex3",
                    )
                    prepared = preprocessor(prepared)
                    inference_start = time.perf_counter()
                    with torch.inference_mode():
                        normalized_chunk = policy.predict_action_chunk(prepared)
                        predicted_chunk = postprocessor(normalized_chunk)
                    inference_times.append(time.perf_counter() - inference_start)
                    action_chunk = predicted_chunk.squeeze(0).cpu().numpy()
                    if action_chunk.shape != (config.chunk_size, 14):
                        raise RuntimeError(
                            f"ACT returned {action_chunk.shape}, expected "
                            f"({config.chunk_size}, 14)"
                        )
                    if not np.isfinite(action_chunk).all():
                        raise RuntimeError("ACT returned a non-finite action chunk")
                    chunk_index = 0

                requested_target = action_chunk[chunk_index]
                clipped_target = np.clip(
                    requested_target,
                    joint_ranges[:, 0],
                    joint_ranges[:, 1],
                )
                if not np.array_equal(requested_target, clipped_target):
                    clip_frames += 1
                joint_target[:] = clipped_target
                chunk_index += 1

                for _ in range(physics_steps_per_action):
                    physics_step()
                    # Evaluate post-step contacts on a copy: extra constraint
                    # solves must not alter the rollout's solver warm-start state.
                    mujoco.mj_copyData(diagnostic_data, model, data)
                    mujoco.mj_forward(model, diagnostic_data)
                    clearance, table_contacts = act_cup_clearance(model, diagnostic_data)
                    if clearance >= 0.010 and table_contacts == 0:
                        clear_seconds += model.opt.timestep
                    else:
                        clear_seconds = 0.0
                    longest_clear_seconds = max(longest_clear_seconds, clear_seconds)

                actual = measured_state()
                cup_z = float(data.body("cup").xpos[2])
                row = {
                    "time_s": (frame_index + 1) / control_fps,
                    "cup_center_rise_mm": 1000 * (cup_z - initial_cup_position[2]),
                    "cup_clearance_mm": 1000 * clearance,
                    "cup_table_contacts": table_contacts,
                    "clear_hold_s": clear_seconds,
                }
                for j, name in enumerate(CONTROLLED_JOINTS):
                    row[name + "_requested_rad"] = float(requested_target[j])
                    row[name + "_clipped_rad"] = float(clipped_target[j])
                    row[name + "_actual_rad"] = float(actual[j])
                for part, (points, force) in finger_cup_contacts(model, diagnostic_data).items():
                    row[part + "_contact_points"] = points
                    row[part + "_normal_force_N"] = force
                diagnostic_rows.append(row)
                if frame_index == 0 or (frame_index + 1) % 25 == 0:
                    print(f"ACT hand t={row['time_s']:.2f}s (thumb0/1/2, index0/1, middle0/1):")
                    print("  requested:", np.round(requested_target[7:].astype(float), 5).tolist())
                    print("  clipped:  ", np.round(clipped_target[7:], 5).tolist())
                    print("  actual:   ", np.round(actual[7:].astype(float), 5).tolist())
                    print(f"  cup lowest clearance={clearance * 1000:.3f} mm, "
                          f"table contacts={table_contacts}, clear hold={clear_seconds:.3f}s", flush=True)

                if save_video:
                    renderer.update_scene(
                        data,
                        camera=third_person_camera,
                        scene_option=video_scene_option,
                    )
                    third_rgb = renderer.render()
                    video_frame = av.VideoFrame.from_ndarray(third_rgb, format="rgb24")
                    for packet in output_stream.encode(video_frame):
                        output_container.mux(packet)
                rendered_frames += 1

                if clear_seconds >= 1.0 - 1e-12:
                    success = True
                    break
                if (frame_index + 1) % 100 == 0:
                    rise_mm = 1000.0 * (cup_z - initial_cup_position[2])
                    print(
                        f"ACT rollout: {frame_index + 1}/{max_frames} frames, "
                        f"cup rise={rise_mm:.2f} mm",
                        flush=True,
                    )

        if save_video:
            for packet in output_stream.encode():
                output_container.mux(packet)
            output_container.close()
            output_container = None
            temporary_path.replace(video_path)
    except Exception:
        if output_container is not None:
            output_container.close()
        if save_video:
            temporary_path.unlink(missing_ok=True)
        raise

    final_cup_position = data.body("cup").xpos.copy()
    final_rise_mm = 1000.0 * (final_cup_position[2] - initial_cup_position[2])
    mean_inference = float(np.mean(inference_times)) if inference_times else float("nan")
    print(
        f"ACT rollout {'PASS' if success else 'FAIL'}: "
        f"frames={rendered_frames}, simulated={rendered_frames / control_fps:.2f} s, "
        f"final cup rise={final_rise_mm:.2f} mm, "
        f"replans={len(inference_times)}, mean inference={mean_inference:.3f} s, "
        f"clipped action frames={clip_frames}"
    )
    print(
        "Success criterion: lowest collision-cylinder point >=10 mm above table, "
        "no cup/table contact, continuously for 1.0 s (checked every physics step)"
    )
    print(f"Longest continuous table-clear hold: {longest_clear_seconds:.4f} s")
    diagnostic_path = video_path.with_suffix(".diagnostics.csv")
    with diagnostic_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0]))
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    print("Hand per-joint ranges over rollout (rad):")
    for name in RIGHT_HAND_JOINTS:
        columns = [np.array([r[name + suffix] for r in diagnostic_rows])
                   for suffix in ("_requested_rad", "_clipped_rad", "_actual_rad")]
        requested, clipped, actual = columns
        print(f"  {name}: requested=[{requested.min():.5f}, {requested.max():.5f}], "
              f"clipped=[{clipped.min():.5f}, {clipped.max():.5f}], "
              f"actual=[{actual.min():.5f}, {actual.max():.5f}], "
              f"max tracking error={np.max(np.abs(clipped - actual)):.5f}")
    print(f"ACT diagnostic CSV SAVED: {diagnostic_path}")
    if save_video:
        print(f"ACT diagnostic video SAVED: {video_path}")


def find_recorded_cup_positions(record_dir: Path) -> list[np.ndarray]:
    """Return cup XY positions from complete, replayable episodes."""
    positions: list[np.ndarray] = []
    if not record_dir.is_dir():
        return positions
    for path in sorted(record_dir.glob("episode_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as episode:
                if not {"cup_pose", "cup_initial_position", "format_version"}.issubset(
                    episode.files
                ):
                    continue
                if int(episode["format_version"]) < 3:
                    continue
                position = np.asarray(
                    episode["cup_initial_position"], dtype=np.float64
                )
                if position.shape == (3,) and np.isfinite(position).all():
                    positions.append(position[:2].copy())
        except (OSError, ValueError):
            print(f"Warning: ignored unreadable episode while counting: {path}")
    return positions


def count_rgb_episodes(record_dir: Path) -> int:
    """Count complete format-v4 episodes whose referenced RGB video exists."""
    count = 0
    if not record_dir.is_dir():
        return count
    for path in sorted(record_dir.glob("episode_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as episode:
                if not {"format_version", "images_included", "head_rgb_video"}.issubset(
                    episode.files
                ):
                    continue
                if int(episode["format_version"]) < 4:
                    continue
                if not bool(episode["images_included"]):
                    continue
                video_name = str(episode["head_rgb_video"])
                video_path = path.parent / video_name
                if video_path.is_file() and video_path.stat().st_size > 0:
                    count += 1
        except (OSError, ValueError):
            print(f"Warning: ignored unreadable episode while counting RGB: {path}")
    return count


def load_cup_region(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        region = {
            key: float(payload[key])
            for key in ("x_min", "x_max", "y_min", "y_max")
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid cup region file {path}: {exc}") from exc
    if not (
        np.isfinite(list(region.values())).all()
        and region["x_min"] < region["x_max"]
        and region["y_min"] < region["y_max"]
    ):
        raise ValueError(f"Invalid cup region bounds in {path}: {region}")
    return region


def main() -> None:
    args = parse_args()
    if args.ik_test_axis is not None and args.orientation_test_axis is not None:
        raise ValueError(
            "Use either --ik-test-axis or --orientation-test-axis, not both"
        )
    project_dir = args.project_dir.expanduser().resolve()
    cup_region_file = (
        args.cup_region_file.expanduser().resolve()
        if args.cup_region_file is not None
        else project_dir / "cup_region.json"
    )
    saved_region = load_cup_region(cup_region_file)
    if saved_region is not None:
        args.cup_x_min = saved_region["x_min"]
        args.cup_x_max = saved_region["x_max"]
        args.cup_y_min = saved_region["y_min"]
        args.cup_y_max = saved_region["y_max"]
        print(
            f"Loaded cup region: X=[{args.cup_x_min:.3f}, {args.cup_x_max:.3f}], "
            f"Y=[{args.cup_y_min:.3f}, {args.cup_y_max:.3f}] m from "
            f"{cup_region_file}"
        )
    if args.act_cup_x is not None and not (
        args.cup_x_min <= args.act_cup_x <= args.cup_x_max
        and args.cup_y_min <= args.act_cup_y <= args.cup_y_max
    ):
        raise ValueError(
            "Fixed ACT cup position is outside the configured cup region: "
            f"({args.act_cup_x:.3f}, {args.act_cup_y:.3f})"
        )
    record_dir = (
        args.record_dir.expanduser().resolve()
        if args.record_dir is not None
        else project_dir / "demonstrations_npz"
    )
    rgb_out = (
        args.rgb_out.expanduser().resolve()
        if args.rgb_out is not None
        else project_dir / "minimal_head_rgb.png"
    )

    model = build_model(project_dir)
    data = mujoco.MjData(model)

    joint_ids = np.array([model.joint(name).id for name in CONTROLLED_JOINTS])
    qpos_addresses = model.jnt_qposadr[joint_ids]
    dof_addresses = model.jnt_dofadr[joint_ids]
    actuator_ids = np.array([model.actuator(name).id for name in CONTROLLED_JOINTS])
    cup_joint_id = model.joint("cup_freejoint").id
    cup_qpos_address = int(model.jnt_qposadr[cup_joint_id])
    cup_dof_address = int(model.jnt_dofadr[cup_joint_id])
    cup_center_xy = model.qpos0[cup_qpos_address:cup_qpos_address + 2].copy()
    region_marker_body_id = model.body("cup_region_marker").id
    region_marker_mocap_id = int(model.body_mocapid[region_marker_body_id])
    region_marker_geom_id = model.geom("cup_region_marker_geom").id
    if region_marker_mocap_id < 0:
        raise RuntimeError("cup_region_marker was not compiled as a mocap body")
    cup_random_seed = (
        int(args.cup_random_seed)
        if args.cup_random_seed is not None
        else int.from_bytes(os.urandom(8), "little") & np.iinfo(np.int64).max
    )
    cup_rng = np.random.default_rng(cup_random_seed)

    if model.nu != 14 or model.nv != 20:
        raise RuntimeError(f"Unexpected reduced model dimensions: nu={model.nu}, nv={model.nv}")
    if model.ncam != 1 or model.camera("head_camera").id < 0:
        raise RuntimeError("head_camera was not compiled correctly")
    tcp_site_id = model.site(TCP_SITE).id
    if tcp_site_id < 0:
        raise RuntimeError(f"{TCP_SITE} was not compiled correctly")
    if args.render_episode_rgb is not None:
        render_episode_head_rgb(
            args.render_episode_rgb,
            model,
            data,
            qpos_addresses,
            cup_qpos_address,
        )
        return
    if args.review_episode is not None:
        render_episode_review(
            args.review_episode,
            model,
            data,
            qpos_addresses,
            cup_qpos_address,
        )
        return
    if args.replay_episode is not None:
        replay_recorded_episode(
            args.replay_episode,
            model,
            data,
            qpos_addresses,
            cup_qpos_address,
        )
        return

    # The high-level controller maintains joint targets; PD converts them to
    # the torque commands accepted by the MJCF motor actuators.
    joint_target = np.zeros(14, dtype=np.float64)
    joint_ranges = model.jnt_range[joint_ids]
    joint_target = np.clip(joint_target, joint_ranges[:, 0], joint_ranges[:, 1])
    home_joint_target = joint_target.copy()
    thumb_close_ratios = np.asarray(args.thumb_close_ratios, dtype=np.float64)
    hand_close_rate_scale = np.concatenate((thumb_close_ratios, np.ones(4)))

    kp = np.array([80, 80, 60, 60, 15, 15, 15] + [2.0] * 7, dtype=np.float64)
    kd = np.array([8, 8, 6, 6, 1.5, 1.5, 1.5] + [0.04] * 7, dtype=np.float64)
    nominal_arm_kp = kp[:7].copy()
    nominal_arm_kd = kd[:7].copy()
    torque_limits = np.array(
        [30, 30, 25, 25, 5, 5, 5] + [0.5] * 7,
        dtype=np.float64,
    )

    def measured_state() -> np.ndarray:
        state = data.qpos[qpos_addresses].astype(np.float32).copy()
        if state.shape != (14,) or not np.isfinite(state).all():
            raise RuntimeError(f"Invalid 14D state: shape={state.shape}, state={state}")
        return state

    def physics_step() -> None:
        mujoco.mj_forward(model, data)
        error = joint_target - data.qpos[qpos_addresses]
        torque = (
            kp * error
            - kd * data.qvel[dof_addresses]
            + data.qfrc_bias[dof_addresses]
        )
        data.ctrl[actuator_ids] = np.clip(torque, -torque_limits, torque_limits)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("MuJoCo produced a non-finite state")

    mujoco.mj_resetData(model, data)
    if args.act_cup_x is not None:
        data.qpos[cup_qpos_address] = args.act_cup_x
        data.qpos[cup_qpos_address + 1] = args.act_cup_y
    data.qpos[qpos_addresses] = joint_target
    mujoco.mj_forward(model, data)

    # Let the cup settle and verify that the hold controller is numerically stable.
    for _ in range(round(0.5 / model.opt.timestep)):
        physics_step()

    if args.hand_hold_test or args.hand_lift_test or args.finger_contact_scan:
        # Diagnostic limits, not a physical grasp stability certificate.
        thumb_target = args.lift_thumb_target if args.hand_lift_test else -0.58
        middle_target = args.lift_middle_target if args.hand_lift_test else 0.35
        middle_0_target = (
            args.lift_middle_0_target
            if args.hand_lift_test and args.lift_middle_0_target is not None
            else middle_target
        )
        middle_1_target = (
            args.lift_middle_1_target
            if args.hand_lift_test and args.lift_middle_1_target is not None
            else middle_target
        )
        close_target = np.array(
            [thumb_target] * 3 + [0.35] * 2 + [middle_0_target, middle_1_target]
        )
        if np.any(close_target < joint_ranges[7:, 0]) or np.any(close_target > joint_ranges[7:, 1]):
            raise RuntimeError("Hold target exceeds joint limits")

        trajectory_renderer = None
        trajectory_frames: list[np.ndarray] = []
        capture_counter = 0
        capture_interval = max(1, round(0.04 / model.opt.timestep))
        trajectory_path = (
            args.trajectory_gif.expanduser().resolve()
            if args.trajectory_gif is not None else None
        )
        if trajectory_path is not None:
            trajectory_renderer = mujoco.Renderer(model, height=240, width=424)
            trajectory_camera = mujoco.MjvCamera()
            trajectory_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            trajectory_camera.lookat[:] = [0.36, -0.085, 0.90]
            trajectory_camera.distance = 0.62
            trajectory_camera.azimuth = -90
            trajectory_camera.elevation = -18

        def capture_trajectory(force: bool = False) -> None:
            nonlocal capture_counter
            if trajectory_renderer is None:
                return
            capture_counter += 1
            if not force and capture_counter % capture_interval:
                return
            trajectory_renderer.update_scene(data, camera=trajectory_camera)
            trajectory_frames.append(trajectory_renderer.render().copy())

        def finalize_trajectory(partial: bool = False) -> None:
            nonlocal trajectory_renderer
            if trajectory_renderer is None:
                return
            try:
                capture_trajectory(force=True)
            finally:
                trajectory_renderer.close()
                trajectory_renderer = None
            if not trajectory_frames:
                if partial:
                    print("Trajectory stopped before any GIF frames were captured")
                    return
                raise RuntimeError("Trajectory GIF requested but no frames were captured")
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            images = [Image.fromarray(frame) for frame in trajectory_frames]
            images[0].save(
                trajectory_path,
                save_all=True,
                append_images=images[1:],
                duration=40,
                loop=0,
                optimize=False,
            )
            label = "partial trajectory GIF" if partial else "trajectory GIF"
            print(
                f"Saved {label}: {trajectory_path} "
                f"({len(trajectory_frames)} frames at 25 FPS)"
            )

        def stop_with_trajectory(message: str) -> None:
            finalize_trajectory(partial=True)
            raise RuntimeError(message)

        mujoco.mj_forward(model, data)
        capture_trajectory(force=True)
        if args.pregrasp_x_mm:
            approach_start = data.site_xpos[tcp_site_id].copy()
            approach_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()
            approach_delta = np.array(
                [args.pregrasp_x_mm, 0.0, 0.0], dtype=np.float64
            ) / 1000.0
            for frame in range(100):
                requested_tcp = approach_start + approach_delta * (frame + 1) / 100
                arm, tcp, rotation, _ = solve_pose_ik(
                    model, data, tcp_site_id, qpos_addresses[:7], dof_addresses[:7],
                    joint_ranges[:7], requested_tcp, approach_rotation,
                )
                if (
                    not np.isfinite(arm).all()
                    or np.linalg.norm(tcp - requested_tcp) > 5e-4
                    or np.linalg.norm(rotation_error_world(approach_rotation, rotation))
                    > np.deg2rad(0.1)
                    or np.max(np.abs(arm - data.qpos[qpos_addresses[:7]])) > 0.10
                ):
                    stop_with_trajectory("Pregrasp approach STOP: invalid IK result")
                joint_target[:7] = arm
                for _ in range(round(0.01 / model.opt.timestep)):
                    physics_step()
                    capture_trajectory()
            mujoco.mj_forward(model, data)
            print(
                f"Pregrasp approach: requested world +X={args.pregrasp_x_mm:g} mm, "
                f"actual TCP delta={np.round(1000 * (data.site_xpos[tcp_site_id] - approach_start), 4).tolist()} mm"
            )
        if args.finger_contact_scan:
            scan_data = mujoco.MjData(model)
            scan_data.qpos[:] = data.qpos
            scan_data.qvel[:] = 0.0
            open_hand = np.clip(
                np.zeros(len(RIGHT_HAND_JOINTS)),
                joint_ranges[7:, 0],
                joint_ranges[7:, 1],
            )
            cup_body_id = model.body("cup").id
            step = 0.005

            def scan_contact_positions(part: str) -> list[np.ndarray]:
                positions = []
                for contact_id in range(scan_data.ncon):
                    contact = scan_data.contact[contact_id]
                    body1 = int(model.geom_bodyid[contact.geom1])
                    body2 = int(model.geom_bodyid[contact.geom2])
                    if body1 == cup_body_id:
                        other_body = body2
                    elif body2 == cup_body_id:
                        other_body = body1
                    else:
                        continue
                    while other_body != 0:
                        name = model.body(other_body).name or ""
                        if name.startswith(f"right_hand_{part}_"):
                            positions.append(contact.pos.copy())
                            break
                        other_body = int(model.body_parentid[other_body])
                return positions

            print(
                "Finger geometric contact scan: cup fixed, no physics steps; "
                f"TCP={np.round(data.site_xpos[tcp_site_id], 6).tolist()} m"
            )
            print(
                f"Scan resolution={step:.3f} rad; each finger's two joints move together"
            )
            for part, local_indices in (
                ("index", (3, 4)),
                ("middle", (5, 6)),
            ):
                scan_data.qpos[qpos_addresses[7:]] = open_hand
                qpos_pair = qpos_addresses[
                    [7 + local_indices[0], 7 + local_indices[1]]
                ]
                upper = float(min(
                    joint_ranges[7 + local_indices[0], 1],
                    joint_ranges[7 + local_indices[1], 1],
                ))
                previous_angle = 0.0
                previous_distance = None
                contact_bracket = None
                closest = (float("inf"), 0.0, "")
                angles = np.arange(0.0, upper + 0.5 * step, step)
                if angles[-1] < upper:
                    angles = np.append(angles, upper)
                for angle in angles:
                    scan_data.qpos[qpos_pair] = min(float(angle), upper)
                    mujoco.mj_forward(model, scan_data)
                    nearest = finger_cup_distances(model, scan_data)[part]
                    if nearest is None:
                        raise RuntimeError(f"No collision geometry found for {part}")
                    distance, geom_name = nearest
                    if distance < closest[0]:
                        closest = (distance, float(angle), geom_name)
                    if distance <= 0.0:
                        contact_bracket = (
                            previous_angle, float(angle), previous_distance, distance
                        )
                        break
                    previous_angle = float(angle)
                    previous_distance = distance

                if contact_bracket is None:
                    print(
                        f"  {part}: NO CONTACT through {upper:.6f} rad; "
                        f"closest distance={1000 * closest[0]:.4f} mm at "
                        f"{closest[1]:.6f} rad, nearest={closest[2]}"
                    )
                    continue

                low, high, coarse_low_distance, coarse_high_distance = contact_bracket
                for _ in range(24):
                    mid = 0.5 * (low + high)
                    scan_data.qpos[qpos_pair] = mid
                    mujoco.mj_forward(model, scan_data)
                    distance = finger_cup_distances(model, scan_data)[part][0]
                    if distance <= 0.0:
                        high = mid
                    else:
                        low = mid
                scan_data.qpos[qpos_pair] = low
                mujoco.mj_forward(model, scan_data)
                low_distance = finger_cup_distances(model, scan_data)[part][0]
                scan_data.qpos[qpos_pair] = high
                mujoco.mj_forward(model, scan_data)
                distance, geom_name = finger_cup_distances(model, scan_data)[part]
                geom_id = int(geom_name.rsplit("geom_", 1)[1])
                nearest_points = np.zeros(6)
                exact_distance = float(mujoco.mj_geomDistance(
                    model, scan_data, geom_id, model.geom("cup_geom").id,
                    1.0, nearest_points,
                ))
                point_text = np.round(
                    0.5 * (nearest_points[:3] + nearest_points[3:]), 6
                ).tolist()
                preload_low = min(high + 0.03, upper)
                preload_high = min(high + 0.05, upper)
                distance_jump = abs(distance - low_distance)
                print(
                    f"  {part}: first contact={high:.6f} rad per joint, "
                    f"distance before/after="
                    f"{1000 * low_distance:.6f}/{1000 * distance:.6f} mm, "
                    f"nearest={geom_name}, world contact point={point_text} m"
                )
                if distance_jump <= 0.001 and abs(exact_distance - distance) <= 1e-9:
                    print(
                        f"    candidate target with preload: "
                        f"{preload_low:.6f} to {preload_high:.6f} rad"
                    )
                else:
                    print(
                        "    WARNING: collision distance is discontinuous at this "
                        "boundary; no preload target recommended from this scan"
                    )
            print("Contact scan complete; no grasp targets were changed")
            finalize_trajectory()
            return
        initial_cup = data.body("cup").xpos.copy()
        initial_tcp = data.site_xpos[tcp_site_id].copy()
        max_displacement = max_error = max_force = 0.0
        hold_samples = all_contact_samples = 0
        print("Stop limits: cup displacement 5 mm, per-finger normal force 20 N, hand error 0.20 rad")
        capture_trajectory(force=True)

        def observe_hand(phase: str) -> tuple[dict[str, tuple[int, float]], np.ndarray, float, float]:
            nonlocal max_displacement, max_error, max_force
            mujoco.mj_forward(model, data)
            capture_trajectory()
            phase_contacts = finger_cup_contacts(model, data)
            phase_forces = np.array(
                [force for _, force in phase_contacts.values()], dtype=np.float64
            )
            displacement = float(np.linalg.norm(data.body("cup").xpos - initial_cup))
            error = float(np.max(np.abs(
                joint_target[7:] - data.qpos[qpos_addresses[7:]]
            )))
            max_displacement = max(max_displacement, displacement)
            max_error = max(max_error, error)
            max_force = max(max_force, float(np.max(phase_forces)))
            if (
                not np.isfinite(phase_forces).all()
                or displacement > 0.005
                or error > 0.20
                or np.max(phase_forces) > 20.0
            ):
                stop_with_trajectory(
                    f"Hold STOP during {phase}: "
                    f"cup displacement={1000 * displacement:.4f} mm, "
                    f"hand error={error:.6f} rad, "
                    f"finger forces={phase_forces.tolist()} N"
                )
            return phase_contacts, phase_forces, displacement, error

        if args.adaptive_close:
            group_local_indices = {
                "thumb": np.array([0, 1, 2]),
                "index": np.array([3, 4]),
                "middle": np.array([5, 6]),
            }
            close_direction = np.array([-1.0] * 3 + [1.0] * 4)
            confirm_steps = max(
                1, round(args.adaptive_contact_seconds / model.opt.timestep)
            )
            latched = {name: False for name in group_local_indices}
            contact_counts = {name: 0 for name in group_local_indices}
            loss_counts = {name: 0 for name in group_local_indices}
            joint_travel = np.where(
                close_direction < 0.0,
                -joint_ranges[7:, 0],
                joint_ranges[7:, 1],
            )
            search_duration = float(np.max(
                joint_travel
                / (args.adaptive_close_speed * hand_close_rate_scale)
            ))
            search_steps = int(np.ceil(search_duration / model.opt.timestep))
            print(
                "Adaptive close: base speed="
                f"{args.adaptive_close_speed:g} rad/s, thumb ratios="
                f"{thumb_close_ratios.tolist()}; latch threshold="
                f"{args.adaptive_contact_force:g} N for "
                f"{args.adaptive_contact_seconds:g} s"
            )
            print(
                f"After all contacts: preload={args.adaptive_preload_rad:g} rad "
                f"over {args.adaptive_preload_seconds:g} s, then hold 1 s"
            )
            for step in range(search_steps):
                for name, local_indices in group_local_indices.items():
                    if latched[name]:
                        continue
                    controlled_indices = 7 + local_indices
                    joint_target[controlled_indices] = np.clip(
                        joint_target[controlled_indices]
                        + close_direction[local_indices]
                        * hand_close_rate_scale[local_indices]
                        * args.adaptive_close_speed * model.opt.timestep,
                        joint_ranges[controlled_indices, 0],
                        joint_ranges[controlled_indices, 1],
                    )
                physics_step()
                contacts, forces, displacement, error = observe_hand("contact search")
                for name, local_indices in group_local_indices.items():
                    loaded = (
                        contacts[name][0] > 0
                        and contacts[name][1] >= args.adaptive_contact_force
                    )
                    if latched[name]:
                        loss_counts[name] = 0 if loaded else loss_counts[name] + 1
                        if loss_counts[name] >= confirm_steps:
                            latched[name] = False
                            contact_counts[name] = 0
                            loss_counts[name] = 0
                            print(
                                f"  {name} contact lost at "
                                f"{(step + 1) * model.opt.timestep:.3f} s; resuming close"
                            )
                    else:
                        contact_counts[name] = contact_counts[name] + 1 if loaded else 0
                        if contact_counts[name] >= confirm_steps:
                            latched[name] = True
                            controlled_indices = 7 + local_indices
                            joint_target[controlled_indices] = data.qpos[
                                qpos_addresses[controlled_indices]
                            ]
                            print(
                                f"  {name} latched at "
                                f"{(step + 1) * model.opt.timestep:.3f} s: "
                                f"actual={np.round(joint_target[controlled_indices], 6).tolist()} rad, "
                                f"force={contacts[name][1]:.4f} N"
                            )
                if all(latched.values()):
                    print(
                        f"All fingers latched after "
                        f"{(step + 1) * model.opt.timestep:.3f} s"
                    )
                    break
                for name, local_indices in group_local_indices.items():
                    if latched[name]:
                        continue
                    controlled_indices = 7 + local_indices
                    limits = np.where(
                        close_direction[local_indices] < 0.0,
                        joint_ranges[controlled_indices, 0],
                        joint_ranges[controlled_indices, 1],
                    )
                    if np.allclose(
                        joint_target[controlled_indices], limits, atol=1e-9
                    ):
                        stop_with_trajectory(
                            f"Adaptive close FAIL: {name} reached joint limits "
                            "without sustained cup contact"
                        )
                if (step + 1) % round(0.25 / model.opt.timestep) == 0:
                    print(
                        f"search {(step + 1) * model.opt.timestep:.2f} s: "
                        f"latched={latched}, forces={np.round(forces, 4).tolist()} N, "
                        f"cup displacement={1000 * displacement:.4f} mm"
                    )
            else:
                stop_with_trajectory(
                    "Adaptive close FAIL: contact search timed out"
                )

            preload_start = joint_target[7:].copy()
            preload_goal = np.clip(
                preload_start
                + close_direction * hand_close_rate_scale
                * args.adaptive_preload_rad,
                joint_ranges[7:, 0], joint_ranges[7:, 1],
            )
            preload_steps = max(
                1, round(args.adaptive_preload_seconds / model.opt.timestep)
            )
            for step in range(preload_steps):
                joint_target[7:] = preload_start + (
                    preload_goal - preload_start
                ) * ((step + 1) / preload_steps)
                physics_step()
                contacts, forces, displacement, error = observe_hand("preload")
                if (step + 1) % round(0.25 / model.opt.timestep) == 0:
                    print(
                        f"preload {(step + 1) * model.opt.timestep:.2f} s: "
                        f"forces={np.round(forces, 4).tolist()} N, "
                        f"cup displacement={1000 * displacement:.4f} mm"
                    )
            print("Adaptive loaded hand target (rad):", joint_target[7:].tolist())
            hold_steps = round(1.0 / model.opt.timestep)
            for step in range(hold_steps):
                physics_step()
                contacts, forces, displacement, error = observe_hand("hold")
                hold_samples += 1
                all_contact_samples += int(
                    all(n > 0 and f > 0.01 for n, f in contacts.values())
                )
                if (step + 1) % round(0.25 / model.opt.timestep) == 0:
                    print(
                        f"hold {(step + 1) * model.opt.timestep:.2f} s: "
                        f"forces={np.round(forces, 4).tolist()} N, "
                        f"cup displacement={1000 * displacement:.4f} mm, "
                        f"hand error={error:.6f} rad"
                    )
        else:
            ramp_duration = max(
                args.thumb_close_seconds, args.finger_close_seconds
            )
            print(
                f"Tabletop hold: thumb closes in {args.thumb_close_seconds:g} s, "
                f"index/middle close in {args.finger_close_seconds:g} s, "
                "then hold 1 s; arm targets fixed; no lift"
            )
            print("Hand target (rad):", close_target.tolist())
            for phase, duration in (("ramp", ramp_duration), ("hold", 1.0)):
                steps = round(duration / model.opt.timestep)
                for step in range(steps):
                    if phase == "ramp":
                        elapsed = (step + 1) * model.opt.timestep
                        thumb_progress = min(
                            elapsed / args.thumb_close_seconds, 1.0
                        )
                        finger_progress = min(
                            elapsed / args.finger_close_seconds, 1.0
                        )
                        joint_target[7:10] = close_target[:3] * thumb_progress
                        joint_target[10:14] = close_target[3:] * finger_progress
                    else:
                        joint_target[7:] = close_target
                    physics_step()
                    contacts, forces, displacement, error = observe_hand(phase)
                    if phase == "hold":
                        hold_samples += 1
                        all_contact_samples += int(
                            all(n > 0 and f > 0.01 for n, f in contacts.values())
                        )
                    if (step + 1) % round(0.25 / model.opt.timestep) == 0:
                        print(
                            f"{phase} {(step + 1) * model.opt.timestep:.2f} s: "
                            f"forces(thumb,index,middle)={np.round(forces, 4).tolist()} N, "
                            f"cup displacement={1000 * displacement:.4f} mm, "
                            f"hand error={error:.6f} rad"
                        )
        fraction = all_contact_samples / hold_samples
        print(
            f"Hold summary: three-finger loaded-contact fraction={fraction:.2%}, "
            f"peak cup displacement={1000 * max_displacement:.4f} mm, "
            f"peak per-finger force={max_force:.4f} N, peak hand error={max_error:.6f} rad, "
            f"final TCP drift={1000 * np.linalg.norm(data.site_xpos[tcp_site_id] - initial_tcp):.4f} mm"
        )
        if fraction < 0.95 or not all(n > 0 and f > 0.01 for n, f in contacts.values()):
            stop_with_trajectory(
                "Hold FAIL: sustained three-finger loaded contact not established"
            )
        print("Tabletop contact hold PASS; table support remains, lifting not tested")
        if not args.hand_lift_test:
            return

        lift_start = data.site_xpos[tcp_site_id].copy()
        # Start the comparison from the same tabletop hold under default gains.
        kp[:7] *= args.lift_arm_kp_scale
        kd[:7] *= np.sqrt(args.lift_arm_kp_scale)
        print(
            f"Lift arm gains: Kp scale={args.lift_arm_kp_scale:.1f}, "
            f"Kd scale={np.sqrt(args.lift_arm_kp_scale):.4f}; hand gains and torque limits unchanged"
        )
        lift_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()
        cup_start = data.body("cup").xpos.copy()
        palm_id = model.body("right_wrist_yaw_link").id

        def cup_in_palm() -> np.ndarray:
            return data.xmat[palm_id].reshape(3, 3).T @ (
                data.body("cup").xpos - data.xpos[palm_id]
            )

        initial_cup_in_palm = cup_in_palm().copy()
        initial_hand_actual = data.qpos[qpos_addresses[7:]].copy()
        initial_cup_tilt = float(np.rad2deg(np.arccos(np.clip(
            data.body("cup").xmat.reshape(3, 3)[2, 2], -1.0, 1.0
        ))))
        peak_relative_shift = 0.0
        peak_cup_tilt = initial_cup_tilt
        peak_finger_errors = np.zeros(7)
        print(
            "Lift relative-motion baseline: palm=right_wrist_yaw_link, "
            f"cup center in palm={np.round(1000 * initial_cup_in_palm, 4).tolist()} mm, "
            f"cup tilt from world +Z={initial_cup_tilt:.4f} deg"
        )
        cup_geom = model.geom("cup_geom").id
        table_geom = model.geom("table").id
        wrench = np.zeros(6)
        supported_samples = 0
        hold_samples = 0
        min_hold_gap = float("inf")
        min_hold_rise = float("inf")
        min_hold_contacts = 3
        slip_distance = {part: 0.0 for part in ("thumb", "index", "middle")}
        peak_tangential_speed = slip_distance.copy()
        contact_time = slip_distance.copy()
        lift_distance = args.lift_distance_mm / 1000.0
        print(
            f"Lift: TCP +{args.lift_distance_mm:g} mm over 2 s, then hold 1 s; "
            "hand targets unchanged"
        )
        for frame in range(300):
            requested_tcp = lift_start + np.array(
                [0., 0., lift_distance * min((frame + 1) / 200, 1.)]
            )
            if frame < 200:
                arm, tcp, rotation, _ = solve_pose_ik(
                    model, data, tcp_site_id, qpos_addresses[:7], dof_addresses[:7],
                    joint_ranges[:7], requested_tcp, lift_rotation,
                )
                if (not np.isfinite(arm).all()
                    or np.linalg.norm(tcp - requested_tcp) > 5e-4
                    or np.linalg.norm(rotation_error_world(lift_rotation, rotation)) > np.deg2rad(0.1)
                    or np.max(np.abs(arm - data.qpos[qpos_addresses[:7]])) > 0.10):
                    stop_with_trajectory(
                        "Lift STOP: invalid IK result or excessive joint change"
                    )
                joint_target[:7] = arm
            for _ in range(round(0.01 / model.opt.timestep)):
                physics_step()
                mujoco.mj_forward(model, data)
                capture_trajectory()
                contacts = finger_cup_contacts(model, data)
                table_force = 0.0
                table_points = 0
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    if {contact.geom1, contact.geom2} == {cup_geom, table_geom}:
                        table_points += 1
                        mujoco.mj_contactForce(model, data, contact_id, wrench)
                        table_force += max(0., float(wrench[0]))
                # Exact world-Z support extent of this cylinder over the horizontal table.
                # Avoid unstable penetration distances from the generic convex query.
                cylinder_axis_z = float(data.geom_xmat[cup_geom].reshape(3, 3)[2, 2])
                radius, half_height = model.geom_size[cup_geom, :2]
                vertical_extent = (
                    half_height * abs(cylinder_axis_z)
                    + radius * np.sqrt(max(0.0, 1.0 - cylinder_axis_z**2))
                )
                table_top = data.geom_xpos[table_geom, 2] + model.geom_size[table_geom, 2]
                gap = float(data.geom_xpos[cup_geom, 2] - vertical_extent - table_top)
                rise = float(data.body("cup").xpos[2] - cup_start[2])
                lateral = float(np.linalg.norm(data.body("cup").xpos[:2] - cup_start[:2]))
                hand_error = float(np.max(np.abs(joint_target[7:] - data.qpos[qpos_addresses[7:]])))
                relative_shift = cup_in_palm() - initial_cup_in_palm
                cup_tilt = float(np.rad2deg(np.arccos(np.clip(cylinder_axis_z, -1.0, 1.0))))
                finger_errors = joint_target[7:] - data.qpos[qpos_addresses[7:]]
                peak_relative_shift = max(peak_relative_shift, float(np.linalg.norm(relative_shift)))
                peak_cup_tilt = max(peak_cup_tilt, cup_tilt)
                peak_finger_errors = np.maximum(peak_finger_errors, np.abs(finger_errors))
                if (not np.isfinite([gap, rise, table_force, hand_error]).all()
                    or lateral > 0.005 or rise < -0.002 or hand_error > 0.20
                    or any(force > 20. for _, force in contacts.values())):
                    stop_with_trajectory(
                        "Lift STOP: abnormal cup motion, force or hand tracking"
                    )
                if frame >= 200:
                    hold_samples += 1
                    supported_samples += int(table_points > 0)
                    min_hold_gap = min(min_hold_gap, gap)
                    min_hold_rise = min(min_hold_rise, rise)
                    min_hold_contacts = min(min_hold_contacts, sum(n > 0 and f > 0.01 for n, f in contacts.values()))
            slip = finger_cup_slip_speeds(model, data)
            for part, (points, tangent_mean, tangent_max, _) in slip.items():
                if points:
                    slip_distance[part] += tangent_mean * 0.01
                    contact_time[part] += 0.01
                    peak_tangential_speed[part] = max(
                        peak_tangential_speed[part], tangent_max
                    )
            if (frame + 1) % 25 == 0:
                print(
                    f"Lift {(frame + 1) * 0.01:.2f} s: cup rise={1000 * rise:.4f} mm, "
                    f"cup-table gap={1000 * gap:.4f} mm, table force={table_force:.6f} N, "
                    f"finger forces={ [round(f, 4) for _, f in contacts.values()] } N"
                )
                print(
                    f"  Relative motion: cup-in-palm delta={np.round(1000 * relative_shift, 4).tolist()} mm, "
                    f"norm={1000 * np.linalg.norm(relative_shift):.4f} mm, cup tilt={cup_tilt:.4f} deg"
                )
                print("  Hand errors target-actual (thumb0,1,2,index0,1,middle0,1):", np.round(finger_errors, 6).tolist(), "rad")
                print(
                    "  Contact slip speed mean/max (thumb,index,middle) mm/s: mean=",
                    [round(1000 * values[1], 4) for values in slip.values()],
                    "max=", [round(1000 * values[2], 4) for values in slip.values()],
                    "normal_abs=", [round(1000 * values[3], 4) for values in slip.values()],
                )
        print(
            f"Lift summary: min hold cup rise={1000 * min_hold_rise:.4f} mm, "
            f"min hold cup-table gap={1000 * min_hold_gap:.4f} mm, "
            f"table-contact fraction={supported_samples / hold_samples:.2%}, "
            f"minimum loaded fingers={min_hold_contacts}, "
            f"actual TCP rise={1000 * (data.site_xpos[tcp_site_id, 2] - lift_start[2]):.4f} mm"
        )
        print(
            f"Relative-motion summary: final cup-in-palm shift={1000 * np.linalg.norm(relative_shift):.4f} mm, "
            f"peak shift={1000 * peak_relative_shift:.4f} mm, "
            f"cup tilt initial/final/peak={initial_cup_tilt:.4f}/{cup_tilt:.4f}/{peak_cup_tilt:.4f} deg"
        )
        print("Contact slip summary (100 Hz estimate over loaded-contact samples):")
        for part in ("thumb", "index", "middle"):
            print(
                f"  {part}: integrated tangential slip={1000 * slip_distance[part]:.4f} mm, "
                f"peak tangential speed={1000 * peak_tangential_speed[part]:.4f} mm/s, "
                f"loaded-contact time={contact_time[part]:.2f} s"
            )
        print("Final hand diagnosis (rad; change measured from lift start):")
        for i, name in enumerate(RIGHT_HAND_JOINTS):
            actual = float(data.qpos[qpos_addresses[7 + i]])
            print(
                f"  {name}: target={joint_target[7 + i]:.6f}, actual={actual:.6f}, "
                f"error={finger_errors[i]:.6f}, peak_abs_error={peak_finger_errors[i]:.6f}, "
                f"actual_change={actual - initial_hand_actual[i]:.6f}"
            )
        # Evaluate commanded joint pose separately from the contact-loaded state.
        target_data = mujoco.MjData(model)
        target_data.qpos[:] = data.qpos
        target_data.qpos[qpos_addresses[:7]] = joint_target[:7]
        mujoco.mj_forward(model, target_data)
        print(
            "Lift tracking diagnosis: "
            f"command FK residual={1000 * np.linalg.norm(target_data.site_xpos[tcp_site_id] - requested_tcp):.4f} mm, "
            f"actual position error={1000 * np.linalg.norm(data.site_xpos[tcp_site_id] - requested_tcp):.4f} mm, "
            f"actual orientation error={np.rad2deg(np.linalg.norm(rotation_error_world(lift_rotation, data.site_xmat[tcp_site_id].reshape(3, 3)))):.4f} deg"
        )
        print("Final arm tracking (torques in Nm; constraint includes contacts and joint limits):")
        for i, name in enumerate(RIGHT_ARM_JOINTS):
            actual = data.qpos[qpos_addresses[i]]
            command_torque = (
                kp[i] * (joint_target[i] - actual)
                - kd[i] * data.qvel[dof_addresses[i]]
                + data.qfrc_bias[dof_addresses[i]]
            )
            print(
                f"  {name}: error={joint_target[i] - actual:.6f} rad, "
                f"requested torque={command_torque:.4f}, ctrl={data.ctrl[actuator_ids[i]]:.4f}, "
                f"limit={torque_limits[i]:.2f}, constraint={data.qfrc_constraint[dof_addresses[i]]:.4f}"
            )
        print("Final active contact pairs (normal force > 0.01 N):")
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            if wrench[0] <= 0.01:
                continue
            names = [
                f"{model.body(int(model.geom_bodyid[g])).name}/geom_{g}"
                for g in (contact.geom1, contact.geom2)
            ]
            print(f"  {names[0]} <-> {names[1]}: normal={wrench[0]:.4f} N")
        finalize_trajectory()
        if min_hold_gap < 0.001 or min_hold_rise < 0.003 or supported_samples or min_hold_contacts < 3:
            raise RuntimeError("Lift FAIL: sustained 3-finger lift with table clearance not established")
        print(
            f"{args.lift_distance_mm:g} mm lift PASS: clear of table throughout "
            "1 s hold; no recording"
        )
        return

    if args.finger_test:
        name = "right_hand_index_0_joint"
        index = CONTROLLED_JOINTS.index(name)
        start_target = float(joint_target[index])
        requested = start_target + 0.05
        if not joint_ranges[index, 0] <= requested <= joint_ranges[index, 1]:
            raise RuntimeError("Finger test target exceeds joint range")
        mujoco.mj_forward(model, data)
        cup_id = model.body("cup").id
        initial_cup = data.xpos[cup_id].copy()
        initial_tcp = data.site_xpos[tcp_site_id].copy()
        errors = []
        print(f"Finger test: {name}, +0.05 rad and return; arm targets held fixed")
        for label, target in (("MOVE", requested), ("RETURN", start_target)):
            joint_target[index] = target
            for _ in range(round(1.0 / model.opt.timestep)):
                physics_step()
            mujoco.mj_forward(model, data)
            actual = float(data.qpos[qpos_addresses[index]])
            error = abs(target - actual)
            errors.append(error)
            print(
                f"Finger {label}: target={target:.6f} rad, actual={actual:.6f} rad, "
                f"error={error:.6f} rad, "
                f"TCP drift={1000 * np.linalg.norm(data.site_xpos[tcp_site_id] - initial_tcp):.4f} mm, "
                f"cup displacement={1000 * np.linalg.norm(data.xpos[cup_id] - initial_cup):.4f} mm"
            )
            measured_state()
        if not np.isfinite(errors).all() or max(errors) > 0.005:
            raise RuntimeError("Finger tracking FAIL: error exceeds 0.005 rad")
        print("Finger tracking PASS (limit 0.005 rad); this is not a grasp test")
        return

    if args.hand_close_scan or args.thumb_close_scan:
        direction = np.array([-1.0] * 3 + [1.0] * 4)
        mujoco.mj_forward(model, data)
        initial_cup = data.body("cup").xpos.copy()
        initial_tcp = data.site_xpos[tcp_site_id].copy()

        def scan_report(label: str) -> None:
            print(label)
            print("  hand actual (rad):", np.round(data.qpos[qpos_addresses[7:]], 6).tolist())
            print(
                f"  cup displacement={1000 * np.linalg.norm(data.body('cup').xpos - initial_cup):.4f} mm, "
                f"TCP drift={1000 * np.linalg.norm(data.site_xpos[tcp_site_id] - initial_tcp):.4f} mm"
            )
            contacts = finger_cup_contacts(model, data)
            for finger, nearest in finger_cup_distances(model, data).items():
                distance = f"{1000 * nearest[0]:.4f} mm" if nearest else "not found within 1 m"
                count, force = contacts[finger]
                print(f"  {finger}: distance={distance}, points={count}, normal_force={force:.6f} N")

        if args.thumb_close_scan:
            print("Thumb scan: prepare hand to 0.30 rad, then hold index/middle targets at +0.30 rad")
            print("Thumb-only stages: -0.35 to -0.80 rad, increments -0.05 rad, up to 1 s per stage")
        else:
            print("Hand closure scan: 0.05 rad increments, 0.50 rad cap, up to 1 s per stage")
        scan_report("Initial state")
        if any(n for n, _ in finger_cup_contacts(model, data).values()):
            print("Scan STOP: initial finger-cup contact; no closure applied")
            return
        max_stage = 16 if args.thumb_close_scan else 10
        for stage in range(1, max_stage + 1):
            amplitude = stage * 0.05
            target = direction * amplitude
            phase = "whole-hand"
            if args.thumb_close_scan:
                phase = "prepare" if stage <= 6 else "thumb-only"
                target[3:] = min(amplitude, 0.30)
            if np.any(target < joint_ranges[7:, 0]) or np.any(target > joint_ranges[7:, 1]):
                raise RuntimeError("Scan STOP: target exceeds joint range")
            joint_target[7:] = target
            print(f"Scan target: phase={phase}, hand={np.round(target, 4).tolist()} rad")
            for step in range(round(1.0 / model.opt.timestep)):
                physics_step()
                # Refresh transforms and constraints at the integrated state.
                mujoco.mj_forward(model, data)
                contacts = finger_cup_contacts(model, data)
                if any(n for n, _ in contacts.values()):
                    scan_report(
                        f"Scan STOP: first finger-cup contact, phase={phase}, target amplitude={amplitude:.2f} rad, "
                        f"stage elapsed={(step + 1) * model.opt.timestep:.4f} s"
                    )
                    print("Diagnostic ended at first contact; no further physics steps or lift")
                    return
                if np.linalg.norm(data.body("cup").xpos - initial_cup) > 0.005:
                    scan_report("Scan STOP: cup displacement exceeded 5 mm without detected finger contact")
                    raise RuntimeError("Unexpected cup motion during closure scan")
            scan_report(f"Stage {stage}: phase={phase}, target amplitude={amplitude:.2f} rad, elapsed=1 s")
        print(f"Scan COMPLETE: reached {max_stage * 0.05:.2f} rad cap without finger-cup contact; not a grasp success")
        return

    ik_report = None
    orientation_report = None
    if args.ik_test_axis is not None:
        axis_index = {"x": 0, "y": 1, "z": 2}[args.ik_test_axis]
        mujoco.mj_forward(model, data)
        initial_tcp = data.site_xpos[tcp_site_id].copy()
        target_tcp = initial_tcp.copy()
        target_tcp[axis_index] += 0.005
        initial_arm = data.qpos[qpos_addresses[:7]].copy()
        solved_arm, solved_tcp, iterations = solve_position_ik(
            model=model,
            source_data=data,
            site_id=tcp_site_id,
            arm_qpos_addresses=qpos_addresses[:7],
            arm_dof_addresses=dof_addresses[:7],
            arm_ranges=joint_ranges[:7],
            target_position=target_tcp,
        )
        solver_error = float(np.linalg.norm(target_tcp - solved_tcp))
        if solver_error > 5e-4:
            raise RuntimeError(
                f"IK failed to reach the 5 mm target: residual={solver_error:.6f} m"
            )
        joint_target[:7] = solved_arm

        # Track the solved target through the same PD interface that later
        # keyboard commands will use; do not teleport the simulated arm.
        for _ in range(round(1.0 / model.opt.timestep)):
            physics_step()
        mujoco.mj_forward(model, data)
        actual_tcp = data.site_xpos[tcp_site_id].copy()
        ik_report = {
            "axis": args.ik_test_axis,
            "initial_tcp": initial_tcp,
            "target_tcp": target_tcp,
            "solved_tcp": solved_tcp,
            "actual_tcp": actual_tcp,
            "iterations": iterations,
            "solver_error": solver_error,
            "tracking_error": float(np.linalg.norm(target_tcp - actual_tcp)),
            "max_joint_delta": float(np.max(np.abs(solved_arm - initial_arm))),
        }

    if args.orientation_test_axis is not None:
        mujoco.mj_forward(model, data)
        initial_tcp = data.site_xpos[tcp_site_id].copy()
        initial_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()
        axis_index = {"roll": 0, "pitch": 1, "yaw": 2}[
            args.orientation_test_axis
        ]
        target_rotation = (
            world_axis_rotation(axis_index, ORIENTATION_TEST_ANGLE)
            @ initial_rotation
        )
        initial_arm = data.qpos[qpos_addresses[:7]].copy()
        solved_arm, solved_tcp, solved_rotation, iterations = solve_pose_ik(
            model=model,
            source_data=data,
            site_id=tcp_site_id,
            arm_qpos_addresses=qpos_addresses[:7],
            arm_dof_addresses=dof_addresses[:7],
            arm_ranges=joint_ranges[:7],
            target_position=initial_tcp,
            target_rotation=target_rotation,
        )
        solver_position_error = float(np.linalg.norm(initial_tcp - solved_tcp))
        solver_rotation_error = float(
            np.linalg.norm(rotation_error_world(target_rotation, solved_rotation))
        )
        if solver_position_error > 5e-4 or solver_rotation_error > np.deg2rad(0.1):
            raise RuntimeError(
                f"{args.orientation_test_axis.title()} IK failed: "
                f"position residual={1000 * solver_position_error:.4f} mm, "
                f"rotation residual={np.rad2deg(solver_rotation_error):.4f} deg"
            )
        joint_target[:7] = solved_arm

        for _ in range(round(1.0 / model.opt.timestep)):
            physics_step()
        mujoco.mj_forward(model, data)
        actual_tcp = data.site_xpos[tcp_site_id].copy()
        actual_rotation = data.site_xmat[tcp_site_id].reshape(3, 3).copy()
        orientation_report = {
            "axis": args.orientation_test_axis,
            "initial_tcp": initial_tcp,
            "actual_tcp": actual_tcp,
            "iterations": iterations,
            "solver_position_error": solver_position_error,
            "solver_rotation_error": solver_rotation_error,
            "tracking_position_error": float(np.linalg.norm(initial_tcp - actual_tcp)),
            "tracking_rotation_error": float(
                np.linalg.norm(rotation_error_world(target_rotation, actual_rotation))
            ),
            "actual_rotation_change": float(
                np.linalg.norm(rotation_error_world(actual_rotation, initial_rotation))
            ),
            "max_joint_delta": float(np.max(np.abs(solved_arm - initial_arm))),
        }

    state = measured_state()
    rgb = None
    if args.headless and not args.control_regression:
        # Keep offscreen rendering in a separate process mode from the GLFW
        # viewer. Mixing both contexts caused Wayland/EGL cleanup crashes.
        with mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH) as renderer:
            renderer.update_scene(data, camera="head_camera")
            rgb = renderer.render().copy()

        if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"Invalid RGB output: shape={rgb.shape}, dtype={rgb.dtype}")
        rgb_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(rgb_out)

    cup_z = float(data.xpos[model.body("cup").id, 2])
    tcp_position = data.site(TCP_SITE).xpos.copy()
    print(
        f"Model: joints={model.njnt}, actuators={model.nu}, "
        f"cameras={model.ncam}, sites={model.nsite}, nq={model.nq}, nv={model.nv}"
    )
    print(
        f"State: shape={state.shape}, dtype={state.dtype}, "
        f"finite={bool(np.isfinite(state).all())}"
    )
    if rgb is not None:
        print(
            f"RGB: shape={rgb.shape}, dtype={rgb.dtype}, "
            f"range=[{int(rgb.min())}, {int(rgb.max())}]"
        )
    elif args.control_regression:
        print("RGB: skipped in control regression")
    elif args.act_policy is not None:
        print("RGB: rendered internally from head_camera for ACT inference")
    else:
        print("RGB: skipped in interactive mode; use --headless to render and save it")
    print(f"Cup center z after settling: {cup_z:.6f} m")
    print(f"Right-hand TCP world position: {tcp_position.tolist()}")
    if ik_report is not None:
        print(
            f"IK test: +5 mm world {ik_report['axis'].upper()}, "
            f"iterations={ik_report['iterations']}, "
            f"solver residual={1000 * ik_report['solver_error']:.4f} mm"
        )
        print("IK initial TCP:", ik_report["initial_tcp"].tolist())
        print("IK target TCP: ", ik_report["target_tcp"].tolist())
        print("IK solved TCP: ", ik_report["solved_tcp"].tolist())
        print("PD actual TCP: ", ik_report["actual_tcp"].tolist())
        print(
            f"PD tracking error={1000 * ik_report['tracking_error']:.4f} mm, "
            f"max joint delta={ik_report['max_joint_delta']:.6f} rad"
        )
    if orientation_report is not None:
        world_axis_name = {"roll": "X", "pitch": "Y", "yaw": "Z"}[
            orientation_report["axis"]
        ]
        print(
            f"{orientation_report['axis'].title()} IK test: "
            f"+5.0000 deg about world {world_axis_name}, "
            f"iterations={orientation_report['iterations']}"
        )
        print(
            "IK residual: "
            f"position={1000 * orientation_report['solver_position_error']:.4f} mm, "
            f"orientation={np.rad2deg(orientation_report['solver_rotation_error']):.4f} deg"
        )
        print(
            "PD actual: "
            f"position drift={1000 * orientation_report['tracking_position_error']:.4f} mm, "
            f"rotation change={np.rad2deg(orientation_report['actual_rotation_change']):.4f} deg, "
            f"orientation error={np.rad2deg(orientation_report['tracking_rotation_error']):.4f} deg, "
            f"max joint delta={orientation_report['max_joint_delta']:.6f} rad"
        )
    if rgb is not None:
        print(f"Saved head RGB: {rgb_out}")

    if args.act_policy is not None:
        act_video_out = (
            args.act_video_out.expanduser().resolve()
            if args.act_video_out is not None
            else project_dir / "act_rollout.mp4"
        )
        run_act_policy_rollout(
            policy_path=args.act_policy,
            video_path=act_video_out,
            execution_steps=args.act_execution_steps,
            rollout_seconds=args.act_rollout_seconds,
            model=model,
            data=data,
            joint_target=joint_target,
            joint_ranges=joint_ranges,
            physics_step=physics_step,
            measured_state=measured_state,
            save_video=not args.act_no_video,
        )
        return

    if args.headless and not args.control_regression:
        return

    mujoco.mj_forward(model, data)
    cartesian_target = data.site_xpos[tcp_site_id].copy()
    orientation_target = data.site_xmat[tcp_site_id].reshape(3, 3).copy()
    pending_tracking_report: tuple[str, float] | None = None
    interactive_hand = {
        "state": "IDLE",
        "latched": {name: False for name in ("thumb", "index", "middle")},
        "contact_counts": {name: 0 for name in ("thumb", "index", "middle")},
        "loss_counts": {name: 0 for name in ("thumb", "index", "middle")},
        "started_at": 0.0,
        "phase_started_at": 0.0,
        "next_report_at": 0.0,
        "initial_cup": data.body("cup").xpos.copy(),
        "preload_start": np.zeros(7),
        "preload_goal": np.zeros(7),
        "open_start": np.zeros(7),
    }
    interactive_arm = {
        "active": False,
        "commands": [],
        "start_time": 0.0,
        "start_target": np.zeros(7),
        "goal_target": np.zeros(7),
        "current_key": "",
        "deferred_hand": None,
    }
    recorded_cup_positions = find_recorded_cup_positions(record_dir)
    recorder = {
        "active": False,
        "saved_count": count_rgb_episodes(record_dir),
        "start_time": 0.0,
        "next_sample_time": 0.0,
        "timestamps": [],
        "states": [],
        "actions": [],
        "cup_poses": [],
        "cup_initial_position": np.zeros(3),
        "cup_offset_xy": np.zeros(2),
    }

    def reset_interactive_hand_state() -> None:
        interactive_hand["state"] = "IDLE"
        for key in ("latched", "contact_counts", "loss_counts"):
            for name in interactive_hand[key]:
                interactive_hand[key][name] = False if key == "latched" else 0

    def set_interactive_arm_gains(loaded: bool) -> None:
        kp[:7] = nominal_arm_kp * (4.0 if loaded else 1.0)
        kd[:7] = nominal_arm_kd * (2.0 if loaded else 1.0)

    movement = {
        "UP": np.array([KEYBOARD_POSITION_STEP, 0.0, 0.0]),
        "DOWN": np.array([-KEYBOARD_POSITION_STEP, 0.0, 0.0]),
        "LEFT": np.array([0.0, KEYBOARD_POSITION_STEP, 0.0]),
        "RIGHT": np.array([0.0, -KEYBOARD_POSITION_STEP, 0.0]),
        "UP_Z": np.array([0.0, 0.0, KEYBOARD_POSITION_STEP]),
        "DOWN_Z": np.array([0.0, 0.0, -KEYBOARD_POSITION_STEP]),
    }

    def apply_keyboard_command(key: str) -> bool:
        nonlocal cartesian_target, orientation_target, pending_tracking_report
        if key == "RESET":
            if recorder["active"]:
                print(
                    f"Recording discarded by reset: "
                    f"{len(recorder['timestamps'])} frames"
                )
                recorder["active"] = False
                recorder["timestamps"].clear()
                recorder["states"].clear()
                recorder["actions"].clear()
                recorder["cup_poses"].clear()
            set_interactive_arm_gains(loaded=False)
            reset_interactive_hand_state()
            interactive_arm["active"] = False
            interactive_arm["commands"].clear()
            interactive_arm["deferred_hand"] = None
            pending_tracking_report = None
            joint_target[:] = home_joint_target
            mujoco.mj_resetData(model, data)
            data.qpos[qpos_addresses] = home_joint_target
            mujoco.mj_forward(model, data)
            cartesian_target = data.site_xpos[tcp_site_id].copy()
            orientation_target = data.site_xmat[tcp_site_id].reshape(3, 3).copy()
            print("Reset scene and orientation; TCP target:", cartesian_target.tolist())
            return True

        if interactive_hand["state"] in ("SEARCHING", "PRELOAD", "OPENING"):
            print(
                f"Ignored {key}: hand state is {interactive_hand['state']}; "
                "wait for GRASP READY/HAND OPEN or press 9 to reset"
            )
            return False

        requested_target = cartesian_target.copy()
        requested_rotation = orientation_target.copy()
        if key in ("YAW_POS", "YAW_NEG"):
            angle = KEYBOARD_YAW_STEP if key == "YAW_POS" else -KEYBOARD_YAW_STEP
            requested_rotation = world_axis_rotation(2, angle) @ orientation_target
        elif key in ("ROLL_POS", "ROLL_NEG"):
            angle = KEYBOARD_ROLL_STEP if key == "ROLL_POS" else -KEYBOARD_ROLL_STEP
            requested_rotation = world_axis_rotation(0, angle) @ orientation_target
        elif key in ("PITCH_POS", "PITCH_NEG"):
            angle = KEYBOARD_PITCH_STEP if key == "PITCH_POS" else -KEYBOARD_PITCH_STEP
            requested_rotation = world_axis_rotation(1, angle) @ orientation_target
        else:
            requested_target += movement[key]
        measured_arm = data.qpos[qpos_addresses[:7]].copy()
        solved_arm, solved_tcp, solved_rotation, iterations = solve_pose_ik(
            model=model,
            source_data=data,
            site_id=tcp_site_id,
            arm_qpos_addresses=qpos_addresses[:7],
            arm_dof_addresses=dof_addresses[:7],
            arm_ranges=joint_ranges[:7],
            target_position=requested_target,
            target_rotation=requested_rotation,
        )
        residual = float(np.linalg.norm(requested_target - solved_tcp))
        rotation_residual = float(
            np.linalg.norm(rotation_error_world(requested_rotation, solved_rotation))
        )
        max_joint_delta = float(np.max(np.abs(solved_arm - measured_arm)))
        if not (
            np.isfinite(solved_arm).all()
            and np.isfinite(residual)
            and np.isfinite(rotation_residual)
        ):
            print(f"Rejected {key}: non-finite IK result")
            return False
        if residual > 5e-4 or rotation_residual > np.deg2rad(0.1):
            print(
                f"Rejected {key}: IK residual position={1000 * residual:.3f} mm, "
                f"orientation={np.rad2deg(rotation_residual):.4f} deg "
                "(limits: 0.5 mm, 0.1 deg)"
            )
            return False
        if max_joint_delta > 0.10:
            print(
                f"Rejected {key}: max joint jump={max_joint_delta:.3f} rad exceeds 0.10 rad"
            )
            return False

        joint_target[:7] = solved_arm
        cartesian_target = requested_target
        orientation_target = requested_rotation
        # Only accepted commands restart the timer. Reset cancels the report.
        pending_tracking_report = (key, float(data.time))
        print(
            f"Key {key}: target={np.round(cartesian_target, 4).tolist()}, "
            f"iterations={iterations}, residual={1000 * residual:.3f} mm, "
            f"orientation_residual={np.rad2deg(rotation_residual):.4f} deg, "
            f"max_joint_delta={max_joint_delta:.4f} rad"
        )

        return True

    def start_next_interactive_arm_command() -> None:
        nonlocal pending_tracking_report
        if interactive_arm["active"] or not interactive_arm["commands"]:
            return
        key = interactive_arm["commands"].pop(0)
        start_target = joint_target[:7].copy()
        if not apply_keyboard_command(key):
            print(f"Arm queue dropped rejected command: {key}")
            start_next_interactive_arm_command()
            return
        goal_target = joint_target[:7].copy()
        joint_target[:7] = start_target
        pending_tracking_report = None
        interactive_arm["active"] = True
        interactive_arm["start_time"] = float(data.time)
        interactive_arm["start_target"] = start_target
        interactive_arm["goal_target"] = goal_target
        interactive_arm["current_key"] = key

    def enqueue_interactive_arm_command(key: str) -> None:
        if key == "RESET":
            apply_keyboard_command(key)
            return
        if interactive_hand["state"] in ("SEARCHING", "PRELOAD", "OPENING"):
            print(
                f"Ignored {key}: hand state is {interactive_hand['state']}; "
                "wait for GRASP READY/HAND OPEN or press 9 to reset"
            )
            return
        if interactive_arm["deferred_hand"] is not None:
            print(
                f"Ignored {key}: {interactive_arm['deferred_hand']} is waiting "
                "for the arm queue"
            )
            return
        interactive_arm["commands"].append(key)
        queued = len(interactive_arm["commands"]) + int(interactive_arm["active"])
        print(
            f"Arm queue: added {key}; pending={queued}, "
            f"{args.keyboard_move_seconds:g} s/command"
        )
        start_next_interactive_arm_command()

    def update_interactive_arm() -> None:
        nonlocal pending_tracking_report
        if not interactive_arm["active"]:
            start_next_interactive_arm_command()
            return
        elapsed = float(data.time) - interactive_arm["start_time"]
        progress = min(max(elapsed / args.keyboard_move_seconds, 0.0), 1.0)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        joint_target[:7] = interactive_arm["start_target"] + smooth * (
            interactive_arm["goal_target"] - interactive_arm["start_target"]
        )
        if progress < 1.0:
            return
        finished_key = interactive_arm["current_key"]
        joint_target[:7] = interactive_arm["goal_target"]
        interactive_arm["active"] = False
        if interactive_arm["commands"]:
            start_next_interactive_arm_command()
            return
        pending_tracking_report = (f"ARM_QUEUE/{finished_key}", float(data.time))
        print("Arm queue READY")
        deferred = interactive_arm["deferred_hand"]
        interactive_arm["deferred_hand"] = None
        if deferred is not None:
            apply_interactive_hand_command(deferred)

    def queue_interactive_hand_command(key: str) -> None:
        if interactive_arm["active"] or interactive_arm["commands"]:
            if interactive_arm["deferred_hand"] is None:
                interactive_arm["deferred_hand"] = key
                print(f"Hand command {key} queued after arm movement")
            else:
                print(
                    f"Ignored {key}: hand command "
                    f"{interactive_arm['deferred_hand']} is already queued"
                )
            return
        apply_interactive_hand_command(key)

    def report_keyboard_tracking() -> tuple[float, float] | None:
        nonlocal pending_tracking_report
        if pending_tracking_report is None:
            return
        key, started_at = pending_tracking_report
        elapsed = float(data.time) - started_at
        if elapsed < 1.0 - 1e-9:
            return
        # mj_step can leave site transforms from before the integration step.
        mujoco.mj_forward(model, data)
        position_error = np.linalg.norm(
            cartesian_target - data.site_xpos[tcp_site_id]
        )
        orientation_error = np.linalg.norm(
            rotation_error_world(
                orientation_target, data.site_xmat[tcp_site_id].reshape(3, 3)
            )
        )
        print(
            f"PD actual {key} after {elapsed:.3f} s simulation: "
            f"position error={1000 * position_error:.4f} mm, "
            f"orientation error={np.rad2deg(orientation_error):.4f} deg"
        )
        pending_tracking_report = None
        return float(position_error), float(orientation_error)

    if args.control_regression:
        sequences = {
            "yaw_translation": ("YAW_POS", "UP", "DOWN", "YAW_NEG"),
            "roll_roundtrip": ("ROLL_POS", "ROLL_NEG"),
            "pitch_roundtrip": ("PITCH_POS", "PITCH_NEG"),
            "combined_rotation": (
                "YAW_POS", "ROLL_POS", "PITCH_POS",
                "PITCH_NEG", "ROLL_NEG", "YAW_NEG",
            ),
            "combined_translation": (
                "YAW_POS", "ROLL_POS", "PITCH_POS", "UP", "DOWN",
            ),
        }
        worst_position = worst_orientation = 0.0
        checked_steps = 0
        for name, commands in sequences.items():
            apply_keyboard_command("RESET")
            for _ in range(round(0.5 / model.opt.timestep)):
                physics_step()
            mujoco.mj_forward(model, data)
            cartesian_target = data.site_xpos[tcp_site_id].copy()
            orientation_target = data.site_xmat[tcp_site_id].reshape(3, 3).copy()
            print(f"Regression sequence: {name}")
            for key in commands:
                if not apply_keyboard_command(key):
                    raise RuntimeError(f"Regression FAIL: {name}/{key}: IK command rejected")
                for _ in range(round(1.0 / model.opt.timestep)):
                    physics_step()
                errors = report_keyboard_tracking()
                if errors is None:
                    raise RuntimeError(f"Regression FAIL: {name}/{key}: missing PD report")
                position_error, orientation_error = errors
                if not (
                    np.isfinite(errors).all()
                    and position_error <= 5e-4
                    and orientation_error <= np.deg2rad(0.1)
                ):
                    raise RuntimeError(
                        f"Regression FAIL: {name}/{key}: PD error exceeds "
                        "0.5 mm / 0.1 deg or is non-finite"
                    )
                measured_state()
                worst_position = max(worst_position, position_error)
                worst_orientation = max(worst_orientation, orientation_error)
                checked_steps += 1
        print(
            f"Control regression PASS: {len(sequences)} sequences, {checked_steps} steps; "
            f"max PD position error={1000 * worst_position:.4f} mm, "
            f"max PD orientation error={np.rad2deg(worst_orientation):.4f} deg"
        )
        return

    import glfw
    from mujoco import viewer as mj_viewer

    key_events: queue.SimpleQueue[str] = queue.SimpleQueue()
    selected_finger = 3  # Start with the already tested index_0 joint.
    pending_finger_report = None
    pending_hand_report = None

    def show_finger_selection() -> None:
        index = 7 + selected_finger
        print(
            f"Selected {RIGHT_HAND_JOINTS[selected_finger]}: "
            f"range={joint_ranges[index].tolist()} rad, "
            f"target={joint_target[index]:.6f} rad, "
            f"actual={data.qpos[qpos_addresses[index]]:.6f} rad"
        )

    def apply_finger_command(key: str) -> None:
        nonlocal selected_finger, pending_finger_report, pending_hand_report
        if key == "RESET":
            apply_keyboard_command(key)
            pending_finger_report = None
            pending_hand_report = None
            show_finger_selection()
            return
        if key == "FINGER_NEXT":
            selected_finger = (selected_finger + 1) % len(RIGHT_HAND_JOINTS)
            show_finger_selection()
            return
        if key in ("HAND_CLOSE", "HAND_OPEN"):
            # Absolute small diagnostic poses; repeated presses do not accumulate.
            requested_hand = (
                np.array([-0.05, -0.05, -0.05, 0.05, 0.05, 0.05, 0.05])
                if key == "HAND_CLOSE" else home_joint_target[7:].copy()
            )
            if not (
                np.all(requested_hand >= joint_ranges[7:, 0])
                and np.all(requested_hand <= joint_ranges[7:, 1])
            ):
                print(f"Rejected {key}: joint range limit")
                return
            mujoco.mj_forward(model, data)
            pending_hand_report = (
                key, requested_hand.copy(), float(data.time),
                data.site_xpos[tcp_site_id].copy(), data.body("cup").xpos.copy(),
            )
            pending_finger_report = None
            joint_target[7:] = requested_hand
            print(f"{key}: hand targets={requested_hand.tolist()} rad; arm targets held fixed")
            return
        index = 7 + selected_finger
        requested = joint_target[index] + (0.05 if key == "FINGER_POS" else -0.05)
        if not joint_ranges[index, 0] - 1e-9 <= requested <= joint_ranges[index, 1] + 1e-9:
            print(f"Rejected {RIGHT_HAND_JOINTS[selected_finger]}: joint range limit")
            return
        requested = float(np.clip(requested, *joint_ranges[index]))
        joint_target[index] = requested
        pending_hand_report = None
        pending_finger_report = (index, requested, float(data.time))
        show_finger_selection()

    def report_finger_tracking() -> None:
        nonlocal pending_finger_report
        if pending_finger_report is None:
            return
        index, target, started = pending_finger_report
        if data.time - started < 1.0 - 1e-9:
            return
        actual = float(data.qpos[qpos_addresses[index]])
        print(
            f"Finger PD {CONTROLLED_JOINTS[index]} after 1 s simulation: "
            f"target={target:.6f} rad, actual={actual:.6f} rad, "
            f"error={abs(target - actual):.6f} rad"
        )
        pending_finger_report = None

    def report_hand_tracking() -> None:
        nonlocal pending_hand_report
        if pending_hand_report is None:
            return
        key, target, started, initial_tcp, initial_cup = pending_hand_report
        if data.time - started < 1.0 - 1e-9:
            return
        mujoco.mj_forward(model, data)
        actual = data.qpos[qpos_addresses[7:]].copy()
        errors = np.abs(target - actual)
        print(f"Hand PD {key} after 1 s simulation:")
        for name, wanted, measured, error in zip(RIGHT_HAND_JOINTS, target, actual, errors):
            print(f"  {name}: target={wanted:.6f}, actual={measured:.6f}, error={error:.6f} rad")
        print(
            f"Hand summary: max error={np.max(errors):.6f} rad, "
            f"TCP drift={1000 * np.linalg.norm(data.site_xpos[tcp_site_id] - initial_tcp):.4f} mm, "
            f"cup displacement={1000 * np.linalg.norm(data.body('cup').xpos - initial_cup):.4f} mm"
        )
        contacts = finger_cup_contacts(model, data)
        print("Finger-cup contacts (snapshot after 1 s; summed normal force):")
        for finger, (count, force) in contacts.items():
            print(f"  {finger}: points={count}, normal_force={force:.6f} N")
        if not any(count for count, _ in contacts.values()):
            print("  No finger-cup contact at this instant")
        print("Finger-cup minimum distances (collision geometry; negative = overlap):")
        for finger, nearest in finger_cup_distances(model, data).items():
            if nearest is None:
                print(f"  {finger}: no eligible geometry within 1 m")
            else:
                distance, geometry = nearest
                print(f"  {finger}: distance={1000 * distance:.4f} mm, nearest={geometry}")
        pending_hand_report = None

    interactive_group_indices = {
        "thumb": np.array([0, 1, 2]),
        "index": np.array([3, 4]),
        "middle": np.array([5, 6]),
    }
    interactive_close_direction = np.array([-1.0] * 3 + [1.0] * 4)
    interactive_confirm_steps = max(
        1, round(args.adaptive_contact_seconds / model.opt.timestep)
    )

    def apply_interactive_hand_command(key: str) -> None:
        nonlocal pending_finger_report, pending_hand_report
        state = interactive_hand["state"]
        if key == "GRASP":
            if state in ("SEARCHING", "PRELOAD", "OPENING"):
                print(f"Ignored G: hand state is {state}")
                return
            if state == "READY":
                print("Hand state: GRASP READY; press H to open or 9 to reset")
                return
            if state == "ERROR":
                print("Hand state: ERROR; press H to open or 9 to reset before retrying")
                return
            mujoco.mj_forward(model, data)
            joint_target[7:] = data.qpos[qpos_addresses[7:]]
            reset_interactive_hand_state()
            interactive_hand["state"] = "SEARCHING"
            interactive_hand["started_at"] = float(data.time)
            interactive_hand["next_report_at"] = float(data.time) + 0.25
            interactive_hand["initial_cup"] = data.body("cup").xpos.copy()
            pending_finger_report = None
            pending_hand_report = None
            print(
                "Hand state: SEARCHING; "
                f"speed={args.adaptive_close_speed:g} rad/s, "
                f"thumb ratios={thumb_close_ratios.tolist()}, "
                f"contact={args.adaptive_contact_force:g} N for "
                f"{args.adaptive_contact_seconds:g} s"
            )
            return

        if key == "OPEN":
            if state == "OPENING":
                print("Hand state: OPENING")
                return
            mujoco.mj_forward(model, data)
            set_interactive_arm_gains(loaded=False)
            actual = data.qpos[qpos_addresses[7:]].copy()
            joint_target[7:] = actual
            interactive_hand["open_start"] = actual
            interactive_hand["phase_started_at"] = float(data.time)
            interactive_hand["state"] = "OPENING"
            pending_finger_report = None
            pending_hand_report = None
            print("Hand state: OPENING over 1.5 s")

    def stop_interactive_hand(message: str) -> None:
        mujoco.mj_forward(model, data)
        joint_target[7:] = data.qpos[qpos_addresses[7:]]
        interactive_hand["state"] = "ERROR"
        print(f"Hand state: ERROR; {message}; press H to open or 9 to reset")

    def update_interactive_hand() -> None:
        state = interactive_hand["state"]
        if state in ("IDLE", "READY", "ERROR"):
            return
        if state == "OPENING":
            elapsed = float(data.time) - interactive_hand["phase_started_at"]
            progress = min(elapsed / 1.5, 1.0)
            joint_target[7:] = interactive_hand["open_start"] + (
                home_joint_target[7:] - interactive_hand["open_start"]
            ) * progress
            if progress >= 1.0:
                joint_target[7:] = home_joint_target[7:]
                reset_interactive_hand_state()
                print("Hand state: HAND OPEN")
            return

        mujoco.mj_forward(model, data)
        contacts = finger_cup_contacts(model, data)
        forces = np.array([force for _, force in contacts.values()])
        displacement = float(np.linalg.norm(
            data.body("cup").xpos - interactive_hand["initial_cup"]
        ))
        hand_error = float(np.max(np.abs(
            joint_target[7:] - data.qpos[qpos_addresses[7:]]
        )))
        if (
            not np.isfinite(forces).all()
            or displacement > 0.005
            or hand_error > 0.20
            or np.max(forces) > 20.0
        ):
            stop_interactive_hand(
                f"safety stop: cup displacement={1000 * displacement:.3f} mm, "
                f"hand error={hand_error:.4f} rad, forces={np.round(forces, 4).tolist()} N"
            )
            return

        if state == "PRELOAD":
            elapsed = float(data.time) - interactive_hand["phase_started_at"]
            progress = min(elapsed / args.adaptive_preload_seconds, 1.0)
            joint_target[7:] = interactive_hand["preload_start"] + (
                interactive_hand["preload_goal"]
                - interactive_hand["preload_start"]
            ) * progress
            if progress >= 1.0:
                joint_target[7:] = interactive_hand["preload_goal"]
                if not all(n > 0 and f > 0.01 for n, f in contacts.values()):
                    stop_interactive_hand(
                        "preload finished without loaded contact on all fingers"
                    )
                    return
                interactive_hand["state"] = "READY"
                set_interactive_arm_gains(loaded=True)
                print(
                    "Hand state: GRASP READY; "
                    f"forces={np.round(forces, 4).tolist()} N; arm keys enabled; "
                    "arm gains Kp x4 / Kd x2"
                )
            return

        latched = interactive_hand["latched"]
        for name, local_indices in interactive_group_indices.items():
            loaded = (
                contacts[name][0] > 0
                and contacts[name][1] >= args.adaptive_contact_force
            )
            if latched[name]:
                interactive_hand["loss_counts"][name] = (
                    0 if loaded else interactive_hand["loss_counts"][name] + 1
                )
                if interactive_hand["loss_counts"][name] >= interactive_confirm_steps:
                    latched[name] = False
                    interactive_hand["contact_counts"][name] = 0
                    interactive_hand["loss_counts"][name] = 0
                    print(f"  {name} contact lost; resuming close")
            else:
                interactive_hand["contact_counts"][name] = (
                    interactive_hand["contact_counts"][name] + 1 if loaded else 0
                )
                if interactive_hand["contact_counts"][name] >= interactive_confirm_steps:
                    latched[name] = True
                    controlled_indices = 7 + local_indices
                    joint_target[controlled_indices] = data.qpos[
                        qpos_addresses[controlled_indices]
                    ]
                    print(
                        f"  {name} latched: "
                        f"actual={np.round(joint_target[controlled_indices], 5).tolist()} rad, "
                        f"force={contacts[name][1]:.4f} N"
                    )

        if all(latched.values()):
            preload_start = joint_target[7:].copy()
            interactive_hand["preload_start"] = preload_start
            interactive_hand["preload_goal"] = np.clip(
                preload_start
                + interactive_close_direction * hand_close_rate_scale
                * args.adaptive_preload_rad,
                joint_ranges[7:, 0], joint_ranges[7:, 1],
            )
            interactive_hand["phase_started_at"] = float(data.time)
            interactive_hand["state"] = "PRELOAD"
            print(
                "Hand state: PRELOAD; all fingers contacted, applying "
                f"{args.adaptive_preload_rad:g} rad over "
                f"{args.adaptive_preload_seconds:g} s"
            )
            return

        for name, local_indices in interactive_group_indices.items():
            if latched[name]:
                continue
            controlled_indices = 7 + local_indices
            joint_target[controlled_indices] = np.clip(
                joint_target[controlled_indices]
                + interactive_close_direction[local_indices]
                * hand_close_rate_scale[local_indices]
                * args.adaptive_close_speed * model.opt.timestep,
                joint_ranges[controlled_indices, 0],
                joint_ranges[controlled_indices, 1],
            )
            limits = np.where(
                interactive_close_direction[local_indices] < 0.0,
                joint_ranges[controlled_indices, 0],
                joint_ranges[controlled_indices, 1],
            )
            if np.allclose(joint_target[controlled_indices], limits, atol=1e-9):
                stop_interactive_hand(
                    f"{name} reached joint limits without sustained cup contact"
                )
                return

        if float(data.time) >= interactive_hand["next_report_at"]:
            print(
                f"Hand SEARCHING: latched={latched}, "
                f"forces={np.round(forces, 4).tolist()} N, "
                f"cup displacement={1000 * displacement:.3f} mm"
            )
            interactive_hand["next_report_at"] += 0.25

    def clear_episode_buffers() -> None:
        recorder["timestamps"].clear()
        recorder["states"].clear()
        recorder["actions"].clear()
        recorder["cup_poses"].clear()

    def record_episode_sample(force: bool = False) -> None:
        if not recorder["active"]:
            return
        now = float(data.time)
        if not force and now + 1e-12 < recorder["next_sample_time"]:
            return
        recorder["timestamps"].append(now - recorder["start_time"])
        recorder["states"].append(measured_state())
        recorder["actions"].append(joint_target.astype(np.float32).copy())
        recorder["cup_poses"].append(
            data.qpos[cup_qpos_address:cup_qpos_address + 7].copy()
        )
        period = 1.0 / args.record_fps
        while recorder["next_sample_time"] <= now + 1e-12:
            recorder["next_sample_time"] += period

    def apply_recording_command(key: str) -> None:
        if key == "RECORD_START":
            if recorder["active"]:
                print(
                    f"Recording already active: {len(recorder['timestamps'])} frames"
                )
                return
            if recorder["saved_count"] >= args.record_target_episodes:
                print(
                    f"Recording target already reached: "
                    f"{recorder['saved_count']}/{args.record_target_episodes} episodes"
                )
                return
            apply_keyboard_command("RESET")
            sampled_cup_xy = None
            for _ in range(10000):
                candidate = np.array([
                    cup_rng.uniform(args.cup_x_min, args.cup_x_max),
                    cup_rng.uniform(args.cup_y_min, args.cup_y_max),
                ])
                if all(
                    np.linalg.norm(candidate - previous) >= 0.002
                    for previous in recorded_cup_positions
                ):
                    sampled_cup_xy = candidate
                    break
            if sampled_cup_xy is None:
                print(
                    "Recording START rejected: could not find a cup position "
                    "at least 2 mm from saved episodes"
                )
                return
            cup_offset_xy = sampled_cup_xy - cup_center_xy
            data.qpos[cup_qpos_address:cup_qpos_address + 2] = sampled_cup_xy
            data.qvel[cup_dof_address:cup_dof_address + 6] = 0.0
            mujoco.mj_forward(model, data)
            for _ in range(round(0.5 / model.opt.timestep)):
                physics_step()
            mujoco.mj_forward(model, data)
            clear_episode_buffers()
            recorder["cup_initial_position"] = data.body("cup").xpos.copy()
            recorder["cup_offset_xy"] = cup_offset_xy.copy()
            recorder["active"] = True
            recorder["start_time"] = float(data.time)
            recorder["next_sample_time"] = float(data.time) + 1.0 / args.record_fps
            record_episode_sample(force=True)
            print(
                f"Recording START {recorder['saved_count'] + 1}/"
                f"{args.record_target_episodes}: cup="
                f"{np.round(recorder['cup_initial_position'], 6).tolist()} m, "
                f"offset_xy={np.round(1000 * cup_offset_xy, 3).tolist()} mm; "
                f"{args.record_fps:g} Hz, state=(14,), action=(14,), "
                "cup_pose=(7,), synchronized head RGB enabled"
            )
            return

        if key == "RECORD_DISCARD":
            if not recorder["active"]:
                print("No active recording to discard")
                return
            count = len(recorder["timestamps"])
            recorder["active"] = False
            clear_episode_buffers()
            print(f"Recording DISCARDED: {count} frames")
            return

        if not recorder["active"]:
            print("No active recording to save; press R first")
            return
        timestamps = np.asarray(recorder["timestamps"], dtype=np.float64)
        states = np.stack(recorder["states"]).astype(np.float32, copy=False)
        actions = np.stack(recorder["actions"]).astype(np.float32, copy=False)
        cup_poses = np.stack(recorder["cup_poses"]).astype(np.float64, copy=False)
        if (
            timestamps.ndim != 1
            or states.shape != (timestamps.size, 14)
            or actions.shape != (timestamps.size, 14)
            or cup_poses.shape != (timestamps.size, 7)
            or not np.isfinite(timestamps).all()
            or not np.isfinite(states).all()
            or not np.isfinite(actions).all()
            or not np.isfinite(cup_poses).all()
            or np.any(np.diff(timestamps) <= 0.0)
        ):
            print("Recording SAVE rejected: invalid timestamp/state/action/cup_pose arrays")
            return
        try:
            episode = compress_recorded_episode(
                timestamps, states, actions, args.record_fps
            )
        except ValueError as exc:
            print(f"Recording SAVE rejected: {exc}")
            return
        saved_timestamps = episode["timestamp"]
        sim_timestamps = episode["sim_timestamp"]
        saved_states = episode["observation_state"]
        saved_actions = episode["action"]
        saved_cup_poses = cup_poses[episode["original_frame_index"]]
        record_dir.mkdir(parents=True, exist_ok=True)
        wall_stamp = time.strftime("%Y%m%d_%H%M%S")
        milliseconds = (time.time_ns() // 1_000_000) % 1000
        output_path = record_dir / f"episode_{wall_stamp}_{milliseconds:03d}.npz"
        video_path = output_path.with_suffix(".head_rgb.mp4")
        np.savez_compressed(
            output_path,
            timestamp=saved_timestamps,
            sim_timestamp=sim_timestamps,
            frame_index=episode["frame_index"],
            original_frame_index=episode["original_frame_index"],
            observation_state=saved_states,
            action=saved_actions,
            cup_pose=saved_cup_poses,
            cup_initial_position=np.asarray(
                recorder["cup_initial_position"], dtype=np.float64
            ),
            cup_offset_xy=np.asarray(recorder["cup_offset_xy"], dtype=np.float64),
            cup_random_bounds_xy=np.asarray(
                [args.cup_x_min, args.cup_x_max, args.cup_y_min, args.cup_y_max],
                dtype=np.float64,
            ),
            cup_random_seed=np.asarray(cup_random_seed, dtype=np.int64),
            thumb_close_ratios=thumb_close_ratios.astype(np.float32),
            fps=np.asarray(args.record_fps, dtype=np.float32),
            joint_names=np.asarray(CONTROLLED_JOINTS),
            task=np.asarray("pick_up_white_cup"),
            images_included=np.asarray(True),
            head_rgb_video=np.asarray(video_path.name),
            head_rgb_shape=np.asarray(
                [saved_timestamps.size, IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                dtype=np.int64,
            ),
            format_version=np.asarray(4, dtype=np.int32),
            original_frame_count=np.asarray(
                episode["original_frame_count"], dtype=np.int64
            ),
            trim_start_frame=np.asarray(episode["trim_start_frame"], dtype=np.int64),
            idle_threshold_seconds=np.asarray(0.6, dtype=np.float32),
            idle_keep_seconds=np.asarray(0.2, dtype=np.float32),
            stationary_velocity_threshold_rad_s=np.asarray(0.01, dtype=np.float32),
        )
        print(
            f"Rendering synchronized head RGB: {saved_timestamps.size} frames; "
            "viewer will pause until encoding finishes"
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--project-dir", str(project_dir),
                    "--render-episode-rgb", str(output_path),
                ],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            output_path.unlink(missing_ok=True)
            video_path.unlink(missing_ok=True)
            print(f"Recording SAVE rejected: RGB rendering failed: {exc}")
            return
        if not video_path.is_file() or video_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            video_path.unlink(missing_ok=True)
            print("Recording SAVE rejected: RGB renderer produced no video")
            return
        recorded_cup_positions.append(
            np.asarray(recorder["cup_initial_position"][:2], dtype=np.float64)
        )
        recorder["saved_count"] += 1
        recorder["active"] = False
        clear_episode_buffers()
        if saved_timestamps.size > 1:
            intervals = np.diff(saved_timestamps)
            cadence = (
                f"dt mean/min/max={intervals.mean():.6f}/"
                f"{intervals.min():.6f}/{intervals.max():.6f} s"
            )
        else:
            cadence = "single-frame episode"
        print(
            f"Recording SAVED: {output_path}; raw_frames={timestamps.size}, "
            f"saved_frames={saved_timestamps.size}, "
            f"duration={saved_timestamps[-1]:.3f} s, {cadence}; "
            f"progress={recorder['saved_count']}/{args.record_target_episodes}"
        )
        if recorder["saved_count"] == args.record_target_episodes:
            print(
                f"Recording target reached: "
                f"{args.record_target_episodes} saved episodes"
            )

    region_selection = {"corner_1": None, "corner_2": None}

    def show_cup_region(x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, 0.832])
        half_size = np.array([(x_max - x_min) / 2, (y_max - y_min) / 2, 0.001])
        model.geom_size[region_marker_geom_id] = half_size
        data.mocap_pos[region_marker_mocap_id] = center
        data.mocap_quat[region_marker_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(model, data)

    def apply_cup_region_command(key: str) -> None:
        if key in ("REGION_X_POS", "REGION_X_NEG", "REGION_Y_POS", "REGION_Y_NEG"):
            delta = {
                "REGION_X_POS": np.array([0.010, 0.0]),
                "REGION_X_NEG": np.array([-0.010, 0.0]),
                "REGION_Y_POS": np.array([0.0, 0.010]),
                "REGION_Y_NEG": np.array([0.0, -0.010]),
            }[key]
            current = data.qpos[cup_qpos_address:cup_qpos_address + 2].copy()
            # Keep the complete 72 mm diameter cup on the tabletop.
            safe_min = np.array([0.216, -0.799])
            safe_max = np.array([1.244, 0.629])
            data.qpos[cup_qpos_address:cup_qpos_address + 2] = np.clip(
                current + delta, safe_min, safe_max
            )
            data.qpos[cup_qpos_address + 2] = 0.880
            data.qpos[cup_qpos_address + 3:cup_qpos_address + 7] = (
                1.0, 0.0, 0.0, 0.0
            )
            data.qvel[cup_dof_address:cup_dof_address + 6] = 0.0
            mujoco.mj_forward(model, data)
            print(
                "Region cursor cup XY: "
                f"{np.round(data.qpos[cup_qpos_address:cup_qpos_address + 2], 3).tolist()} m"
            )
            return
        if key == "REGION_CORNER_1":
            region_selection["corner_1"] = data.qpos[
                cup_qpos_address:cup_qpos_address + 2
            ].copy()
            print(
                "Cup region corner 1: "
                f"{np.round(region_selection['corner_1'], 3).tolist()} m; "
                "move to the opposite corner and press V"
            )
            return
        if key == "REGION_CORNER_2":
            if region_selection["corner_1"] is None:
                print("Cup region corner 1 is not set; press C first")
                return
            corner_2 = data.qpos[cup_qpos_address:cup_qpos_address + 2].copy()
            corner_1 = region_selection["corner_1"]
            x_min, y_min = np.minimum(corner_1, corner_2)
            x_max, y_max = np.maximum(corner_1, corner_2)
            if x_max - x_min < 0.010 or y_max - y_min < 0.010:
                print("Cup region rejected: width and height must each be at least 10 mm")
                return
            region_selection["corner_2"] = corner_2
            payload = {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
                "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            cup_region_file.parent.mkdir(parents=True, exist_ok=True)
            cup_region_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            show_cup_region(x_min, x_max, y_min, y_max)
            print(
                f"Cup region SAVED: X=[{x_min:.3f}, {x_max:.3f}], "
                f"Y=[{y_min:.3f}, {y_max:.3f}] m -> {cup_region_file}"
            )
            return
        if key == "REGION_RESET":
            region_selection["corner_1"] = None
            region_selection["corner_2"] = None
            data.qpos[cup_qpos_address:cup_qpos_address + 2] = cup_center_xy
            data.qpos[cup_qpos_address + 2] = 0.880
            data.qpos[cup_qpos_address + 3:cup_qpos_address + 7] = (
                1.0, 0.0, 0.0, 0.0
            )
            data.qvel[cup_dof_address:cup_dof_address + 6] = 0.0
            mujoco.mj_forward(model, data)
            print("Cup region selection reset; cursor returned to scene center")

    key_map = {
        glfw.KEY_UP: "UP",
        glfw.KEY_DOWN: "DOWN",
        glfw.KEY_LEFT: "LEFT",
        glfw.KEY_RIGHT: "RIGHT",
        glfw.KEY_8: "UP_Z",
        glfw.KEY_7: "DOWN_Z",
        glfw.KEY_6: "YAW_POS",
        glfw.KEY_5: "YAW_NEG",
        glfw.KEY_4: "ROLL_POS",
        glfw.KEY_3: "ROLL_NEG",
        glfw.KEY_2: "PITCH_POS",
        glfw.KEY_1: "PITCH_NEG",
        glfw.KEY_G: "GRASP",
        glfw.KEY_H: "OPEN",
        glfw.KEY_R: "RECORD_START",
        glfw.KEY_E: "RECORD_SAVE",
        glfw.KEY_X: "RECORD_DISCARD",
        glfw.KEY_9: "RESET",
    }

    if args.select_cup_region:
        key_map = {
            glfw.KEY_UP: "REGION_X_POS",
            glfw.KEY_DOWN: "REGION_X_NEG",
            glfw.KEY_LEFT: "REGION_Y_POS",
            glfw.KEY_RIGHT: "REGION_Y_NEG",
            glfw.KEY_C: "REGION_CORNER_1",
            glfw.KEY_V: "REGION_CORNER_2",
            glfw.KEY_9: "REGION_RESET",
        }
    elif args.finger_debug:
        key_map = {
            glfw.KEY_0: "FINGER_NEXT",
            glfw.KEY_8: "FINGER_POS",
            glfw.KEY_7: "FINGER_NEG",
            glfw.KEY_6: "HAND_CLOSE",
            glfw.KEY_5: "HAND_OPEN",
            glfw.KEY_9: "RESET",
        }

    def key_callback(keycode: int) -> None:
        command = key_map.get(keycode)
        if command is not None:
            key_events.put(command)

    with mj_viewer.launch_passive(
        model,
        data,
        show_left_ui=True,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        viewer.cam.lookat[:] = [0.33, -0.08, 0.90]
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -20
        if args.select_cup_region:
            if saved_region is not None:
                show_cup_region(
                    saved_region["x_min"], saved_region["x_max"],
                    saved_region["y_min"], saved_region["y_max"],
                )
            print("Cup region selection (10 mm per press):")
            print("  Up/Down: cup +X/-X")
            print("  Left/Right: cup +Y/-Y")
            print("  C: set first corner; V: set opposite corner and save")
            print(f"  Region file: {cup_region_file}")
            print("  9: clear selected corners and return cup to scene center")
        elif args.finger_debug:
            print("Finger diagnostics: 0 next joint; 8/7 +0.05/-0.05 rad; 9 reset")
            print("  6: whole-hand small close (absolute +/-0.05 rad); 5: hand zero pose")
            print("  Wait for Hand PD before the next movement; repeated 6 does not close further")
            print("Arm targets held fixed. Wait for Finger PD before the next movement.")
            show_finger_selection()
        else:
            print("Keyboard XYZ control (world frame, 5 mm per press):")
            print("  Up/Down: +X/-X forward/back")
            print("  Left/Right: +Y/-Y left/right")
            print("  8/7: +Z/-Z up/down")
            print("  6/5: yaw +5/-5 deg about world Z, keeping TCP target position")
            print("  4/3: roll +5/-5 deg about world X, keeping TCP target position")
            print("  2/1: pitch +2/-2 deg about world Y, keeping TCP target position")
            print("  XYZ movement keeps the target orientation")
            print(
                f"  Arm keys use a smooth FIFO queue: "
                f"{args.keyboard_move_seconds:g} s per command"
            )
            print("  G: adaptive grasp; H: open hand over 1.5 s")
            print("  Arm movement is locked during SEARCHING/PRELOAD/OPENING")
            print(
                f"  R: reset, randomize cup in X=[{args.cup_x_min:.3f}, "
                f"{args.cup_x_max:.3f}], Y=[{args.cup_y_min:.3f}, "
                f"{args.cup_y_max:.3f}] m, and start recording"
            )
            print("  E: save episode; X: discard episode")
            print(
                f"  Recording: {args.record_fps:g} Hz state/action/cup pose + "
                f"head RGB MP4 -> {record_dir}"
            )
            print(
                f"  Session target: {args.record_target_episodes} saved episodes; "
                f"found={recorder['saved_count']}, random seed={cup_random_seed}"
            )
            print("  9: reset robot and cup")
            print("  PD error prints after 1 s simulation; each accepted key restarts the timer")
        print("Viewer running; close the window to exit.")
        while viewer.is_running():
            loop_start = time.monotonic()
            with viewer.lock():
                while not key_events.empty():
                    command = key_events.get()
                    if args.select_cup_region:
                        apply_cup_region_command(command)
                    elif args.finger_debug:
                        apply_finger_command(command)
                    elif command in (
                        "RECORD_START", "RECORD_SAVE", "RECORD_DISCARD"
                    ):
                        apply_recording_command(command)
                    elif command in ("GRASP", "OPEN"):
                        queue_interactive_hand_command(command)
                    else:
                        enqueue_interactive_arm_command(command)
                for _ in range(50):
                    if args.select_cup_region:
                        mujoco.mj_forward(model, data)
                    else:
                        if not args.finger_debug:
                            update_interactive_arm()
                        physics_step()
                    if not args.finger_debug and not args.select_cup_region:
                        update_interactive_hand()
                        record_episode_sample()
                report_keyboard_tracking()
                report_finger_tracking()
                report_hand_tracking()
            if not viewer.is_running():
                break
            viewer.sync()
            simulated_interval = 50 * model.opt.timestep
            time.sleep(max(0.0, simulated_interval - (time.monotonic() - loop_start)))

        # On this Wayland/GLFW combination, Python's normal atexit cleanup can
        # try to release an EGL context after the native viewer already closed
        # it. All interactive work is complete here, so flush logs and let the
        # operating system release the GUI resources without a second teardown.
        if recorder["active"]:
            print(
                f"Recording discarded because viewer closed before E: "
                f"{len(recorder['timestamps'])} frames"
            )
        print("Viewer closed; exiting cleanly.")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
