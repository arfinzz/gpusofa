"""Material check: one block, every material, CPU (SOFA) against GPU.

A 2 cm cube of tissue, base fixed, is loaded by a strong, tilted gravity (shear
and compression at once, so the strain is not lined up with the axes) and
swings to rest over 0.6 s. The same block and material run on SOFA's CPU
components (SOFA_VALIDATION_SIDE=cpu) and on GpuTissueSolver (gpu).

Checked:
  * stage by stage (gpu with SOFA_VALIDATION_COMPARE=1): GpuTissueSolver runs a
    hidden copy of SOFA's CPU components on the GPU's state every step and writes
    the differences of forces, system matrix, velocity change and free motion;
  * whole run: the watched vertices' paths, CPU run against GPU run.

SOFA_VALIDATION_MATERIAL picks the material (validation_common.MATERIALS):
neohookean, stable_neohookean, stvk, mooney_rivlin, ogden, ogden_maxwell
(the realistic default), sls_ogden_sofa (SofaViscoElastic, as it runs).
SOFA_VALIDATION_DIVISIONS sets the cells per edge (default 6).

Output (log dir): material_check_<material>_<side>.csv (+ _gpu_tissue_compare.csv).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

import validation_common as vc  # noqa: E402

SIZE = 0.02
DIVISIONS = int(os.environ.get("SOFA_VALIDATION_DIVISIONS", "6"))
# Tilted, strong gravity: about 20% shear and 15% compression at rest for the 1 kPa materials.
GRAVITY = [vc.env_float("SOFA_VALIDATION_GX", 12.0), vc.env_float("SOFA_VALIDATION_GY", -9.81),
           vc.env_float("SOFA_VALIDATION_GZ", 5.0)]
DT = 0.01
STEPS = 60


def createScene(root):
    label = vc.run_label("material_check")
    root.name = "MaterialCheck"
    root.dt = DT
    root.gravity = GRAVITY
    vc.add_plugins(root)
    root.addObject("DefaultAnimationLoop")
    root.addObject("VisualStyle", displayFlags="showBehaviorModels showForceFields")

    half = SIZE / 2
    mesh = vc.BoxMesh(vc.uniform(-half, half, DIVISIONS), vc.uniform(-SIZE, 0.0, DIVISIONS), vc.uniform(-half, half, DIVISIONS))
    bottom = mesh.face(1, 0)
    tissue, _ = vc.add_tissue(root, mesh, vc.material(), fixed=bottom, label=label)

    watch = {"top_center": mesh.nearest([0.0, 0.0, 0.0]),
             "top_corner_px_pz": mesh.nearest([half, 0.0, half]),
             "top_corner_mx_mz": mesh.nearest([-half, 0.0, -half]),
             "mid_side_px": mesh.nearest([half, -half, 0.0])}
    root.addObject(vc.PositionLogger(name="logger", root=root, dofs=tissue.dofs, vertices=list(watch.values()),
                                     names=list(watch.keys()), path=os.path.join(vc.log_dir(), label + ".csv"),
                                     solver=tissue.odeSolver if vc.SIDE == "gpu" and vc.MEASURE_TIMES else None))
    print(f"MaterialCheck: side={vc.SIDE} material={vc.MATERIAL} nodes={len(mesh.positions)} "
          f"tetrahedra={len(mesh.tetrahedra)} steps={STEPS}", flush=True)
    return root
