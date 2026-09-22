"""LeRobot ACT loading and closed-loop MuJoCo rollout."""

from __future__ import annotations

from pathlib import Path
import time

import mujoco
import numpy as np

from g1_dex3_act_grasping.constants import (
    CONTROLLED_JOINTS, IMAGE_HEIGHT, IMAGE_WIDTH, RIGHT_HAND_JOINTS,
)
from g1_dex3_act_grasping.control.dex3_controller import finger_cup_contacts
from g1_dex3_act_grasping.evaluation.success_metrics import act_cup_clearance


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
