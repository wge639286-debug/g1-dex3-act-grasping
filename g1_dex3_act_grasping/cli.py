"""Command-line interface for data collection, diagnostics, and ACT rollout."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
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
