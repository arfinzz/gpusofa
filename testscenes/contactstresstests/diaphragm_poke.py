"""Hard poke of a diaphragm: a thin flat sheet of muscle-like tissue, 8 x 8 cm and 4.5 mm
thick, clamped on its rim, sags under its weight and is then pushed 20 mm down (from its
undeformed plane) by a rigid 16 mm ball-tip tool at 20 mm/s; it wraps around the tool,
which makes a large contact patch. Set-up, settings and outputs: hard_poke_common.py.
Run: bash scripts/run_hard_poke_wsl.sh diaphragm
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import hard_poke_common as hp  # noqa: E402


def createScene(root):
    return hp.build_scene(root, "diaphragm")
