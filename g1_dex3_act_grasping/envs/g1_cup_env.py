"""Construction of the fixed-base G1 + Dex3 white-cup MuJoCo scene."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from g1_dex3_act_grasping.constants import (
    CONTROLLED_JOINTS, IMAGE_HEIGHT, IMAGE_WIDTH, TCP_SITE,
)


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
