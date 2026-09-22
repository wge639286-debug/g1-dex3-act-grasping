"""Compose the G1 + Dex3 scene, controllers, data tools, and CLI modes.

Domain logic lives in focused package modules.  This module owns mode selection
and the remaining legacy interactive-session orchestration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import mujoco
import numpy as np
from PIL import Image

from g1_dex3_act_grasping.cli import parse_args
from g1_dex3_act_grasping.constants import (
    CONTROLLED_JOINTS, IMAGE_HEIGHT, IMAGE_WIDTH, KEYBOARD_PITCH_STEP,
    KEYBOARD_POSITION_STEP, KEYBOARD_ROLL_STEP, KEYBOARD_YAW_STEP,
    ORIENTATION_TEST_ANGLE, RIGHT_ARM_JOINTS, RIGHT_HAND_JOINTS, TCP_SITE,
)
from g1_dex3_act_grasping.control.dex3_controller import (
    finger_cup_contacts, finger_cup_distances, finger_cup_slip_speeds,
)
from g1_dex3_act_grasping.control.ik import (
    rotation_error_world, solve_pose_ik, solve_position_ik, world_axis_rotation,
)
from g1_dex3_act_grasping.data.recorder import (
    compress_recorded_episode, count_rgb_episodes, find_recorded_cup_positions,
    load_cup_region,
)
from g1_dex3_act_grasping.data.replay import (
    render_episode_head_rgb, render_episode_review, replay_recorded_episode,
)
from g1_dex3_act_grasping.envs.g1_cup_env import build_model
from g1_dex3_act_grasping.policies.act_policy import run_act_policy_rollout


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
                    str(project_dir / "g1_cup_minimal.py"),
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
