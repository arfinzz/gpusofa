"""Hard poke of a cuboid: a block of liver-like tissue, 6 x 6 cm and 3 cm tall, standing on
a table, pushed 12 mm deep by a rigid 16 mm ball-tip tool at 20 mm/s: the whole tip and
the start of the shaft sink in, which makes a large contact patch. Set-up, settings and
outputs: hard_poke_common.py.
Run: bash scripts/run_hard_poke_wsl.sh cuboid
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import hard_poke_common as hp  # noqa: E402


def createScene(root):
    return hp.build_scene(root, "cuboid")
