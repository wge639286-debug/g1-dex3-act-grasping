"""Physics-based success measurements for cup lifting."""

from __future__ import annotations

import mujoco
import numpy as np


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
