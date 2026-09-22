#!/usr/bin/env python3
"""Backward-compatible CLI entry point for the G1 + Dex3 cup project.

Implementation lives in the ``g1_dex3_act_grasping`` package.  Existing
commands continue to use this filename so recorded instructions remain valid.
"""

from g1_dex3_act_grasping.application import main


if __name__ == "__main__":
    main()
