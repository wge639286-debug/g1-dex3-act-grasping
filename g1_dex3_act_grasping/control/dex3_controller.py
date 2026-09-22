"""Dex3 contact, distance, and slip diagnostics."""

from __future__ import annotations

import mujoco
import numpy as np


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
