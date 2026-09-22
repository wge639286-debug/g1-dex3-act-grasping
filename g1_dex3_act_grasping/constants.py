"""Shared robot, camera, and teleoperation constants."""

from __future__ import annotations

import numpy as np

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
