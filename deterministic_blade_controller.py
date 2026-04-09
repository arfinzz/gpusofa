import atexit
import csv
import os
import time

import Sofa.Core


class DeterministicBladeForceController(Sofa.Core.Controller):
    """
    Deterministic force schedule for repeatable benchmarking.

    This keeps the same rigid-body / contact structure as your current scene,
    but removes human keyboard input. The schedule is fixed in simulation steps,
    so if dt stays the same, every run applies the same forces in the same order.

    Phases:
      0 .. settle_steps-1                  : no extra force
      settle .. settle+descend-1           : push blade downward (-Y)
      then                                  : keep pressing downward while sweeping +X
      final phase                           : lift blade (+Y) and stop
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_field = kwargs.get("force_field")
        self.settle_steps = int(kwargs.get("settle_steps", 100))
        self.descend_steps = int(kwargs.get("descend_steps", 350))
        self.sweep_steps = int(kwargs.get("sweep_steps", 700))
        self.lift_steps = int(kwargs.get("lift_steps", 200))
        self.down_force = float(kwargs.get("down_force", 25.0))
        self.sweep_force = float(kwargs.get("sweep_force", 10.0))
        self.lift_force = float(kwargs.get("lift_force", 18.0))
        self.current_step = 0

    def _zero_force(self):
        if self.force_field is not None:
            self.force_field.forces.value = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    def onAnimateBeginEvent(self, event):
        if self.force_field is None:
            return

        f = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        s0 = self.settle_steps
        s1 = s0 + self.descend_steps
        s2 = s1 + self.sweep_steps
        s3 = s2 + self.lift_steps

        if self.current_step < s0:
            # settle, no extra force
            pass
        elif self.current_step < s1:
            # push down into the tissue
            f[1] = -self.down_force
        elif self.current_step < s2:
            # continue pressing down while sweeping in +X
            f[1] = -0.45 * self.down_force
            f[0] = self.sweep_force
        elif self.current_step < s3:
            # lift out
            f[1] = self.lift_force
        else:
            # stop all forces after the scripted motion completes
            pass

        self.force_field.forces.value = [f]
        self.current_step += 1


class KinematicBladePathController(Sofa.Core.Controller):
    """
    Exact path controller for maximum repeatability.

    This overwrites the rigid blade pose each animation step. It is excellent for
    benchmark repeatability, but it is less physically faithful than the force
    controller because the blade motion is prescribed rather than emerging from
    rigid-body dynamics.

    Use this when you want the blade to follow exactly the same path in CPU and GPU
    runs, independent of small floating-point differences.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rigid_dofs = kwargs.get("rigid_dofs")
        self.start_pos = list(kwargs.get("start_pos", [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        self.settle_steps = int(kwargs.get("settle_steps", 100))
        self.descend_steps = int(kwargs.get("descend_steps", 350))
        self.sweep_steps = int(kwargs.get("sweep_steps", 700))
        self.lift_steps = int(kwargs.get("lift_steps", 200))
        self.total_down = float(kwargs.get("total_down", 2.6))
        self.total_sweep_x = float(kwargs.get("total_sweep_x", 2.5))
        self.total_lift = float(kwargs.get("total_lift", 2.0))
        self.current_step = 0

    @staticmethod
    def _lerp(a, b, t):
        return a + (b - a) * t

    def onAnimateBeginEvent(self, event):
        if self.rigid_dofs is None:
            return

        s0 = self.settle_steps
        s1 = s0 + self.descend_steps
        s2 = s1 + self.sweep_steps
        s3 = s2 + self.lift_steps

        x0, y0, z0, qx, qy, qz, qw = self.start_pos
        x, y, z = x0, y0, z0

        if self.current_step < s0:
            pass
        elif self.current_step < s1:
            t = (self.current_step - s0) / max(1, self.descend_steps)
            y = self._lerp(y0, y0 - self.total_down, t)
        elif self.current_step < s2:
            t = (self.current_step - s1) / max(1, self.sweep_steps)
            y = y0 - self.total_down
            x = self._lerp(x0, x0 + self.total_sweep_x, t)
        elif self.current_step < s3:
            t = (self.current_step - s2) / max(1, self.lift_steps)
            x = x0 + self.total_sweep_x
            y = self._lerp(y0 - self.total_down, y0 - self.total_down + self.total_lift, t)
        else:
            x = x0 + self.total_sweep_x
            y = y0 - self.total_down + self.total_lift

        self.rigid_dofs.position.value = [[x, y, z, qx, qy, qz, qw]]
        self.current_step += 1


class MultiKinematicBladePathController(Sofa.Core.Controller):
    """
    Applies the same deterministic path shape to multiple rigid blades.

    Each blade keeps its own starting pose and optional sweep direction, which
    makes it easy to create dense, repeatable collision-heavy benchmarks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rigid_dofs_list = list(kwargs.get("rigid_dofs_list", []))
        self.start_positions = [list(p) for p in kwargs.get("start_positions", [])]
        self.sweep_signs = list(kwargs.get("sweep_signs", []))
        self.settle_steps = int(kwargs.get("settle_steps", 100))
        self.descend_steps = int(kwargs.get("descend_steps", 350))
        self.sweep_steps = int(kwargs.get("sweep_steps", 700))
        self.lift_steps = int(kwargs.get("lift_steps", 200))
        self.total_down = float(kwargs.get("total_down", 2.6))
        self.total_sweep_x = float(kwargs.get("total_sweep_x", 2.5))
        self.total_lift = float(kwargs.get("total_lift", 2.0))
        self.current_step = 0

        if not self.sweep_signs:
            self.sweep_signs = [1.0 for _ in self.start_positions]

    @staticmethod
    def _lerp(a, b, t):
        return a + (b - a) * t

    def _compute_xy(self, start_pos, sweep_sign):
        s0 = self.settle_steps
        s1 = s0 + self.descend_steps
        s2 = s1 + self.sweep_steps
        s3 = s2 + self.lift_steps

        x0, y0, z0, qx, qy, qz, qw = start_pos
        x, y, z = x0, y0, z0

        if self.current_step < s0:
            pass
        elif self.current_step < s1:
            t = (self.current_step - s0) / max(1, self.descend_steps)
            y = self._lerp(y0, y0 - self.total_down, t)
        elif self.current_step < s2:
            t = (self.current_step - s1) / max(1, self.sweep_steps)
            y = y0 - self.total_down
            x = self._lerp(x0, x0 + sweep_sign * self.total_sweep_x, t)
        elif self.current_step < s3:
            t = (self.current_step - s2) / max(1, self.lift_steps)
            x = x0 + sweep_sign * self.total_sweep_x
            y = self._lerp(y0 - self.total_down, y0 - self.total_down + self.total_lift, t)
        else:
            x = x0 + sweep_sign * self.total_sweep_x
            y = y0 - self.total_down + self.total_lift

        return [x, y, z, qx, qy, qz, qw]

    def onAnimateBeginEvent(self, event):
        for index, rigid_dofs in enumerate(self.rigid_dofs_list):
            if rigid_dofs is None or index >= len(self.start_positions):
                continue

            sweep_sign = self.sweep_signs[index] if index < len(self.sweep_signs) else 1.0
            pose = self._compute_xy(self.start_positions[index], sweep_sign)
            rigid_dofs.position.value = [pose]

        self.current_step += 1


class BenchmarkTimingController(Sofa.Core.Controller):
    """
    Lightweight per-step timing logger for batch benchmark runs.

    It records step durations from animation begin/end callbacks and writes:
      - a CSV with one row per step
      - a small summary text file updated periodically and at process exit
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = str(kwargs.get("label", "benchmark"))
        self.warmup_steps = int(kwargs.get("warmup_steps", 50))
        self.flush_interval = int(kwargs.get("flush_interval", 50))
        self.log_interval = int(kwargs.get("log_interval", 200))
        self.output_dir = os.path.abspath(str(kwargs.get("output_dir", os.path.join(os.getcwd(), "benchmark_logs"))))
        self.current_step = 0
        self.step_start_time = None
        self.measured_count = 0
        self.measured_total = 0.0
        self.measured_min = None
        self.measured_max = None
        self.pending_rows = []

        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, f"{self.label}_timings.csv")
        self.summary_path = os.path.join(self.output_dir, f"{self.label}_summary.txt")

        self._ensure_csv_header()
        atexit.register(self._finalize)

    def _ensure_csv_header(self):
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return

        with open(self.csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["step", "duration_seconds", "included_in_stats"])

    def _flush_rows(self):
        if not self.pending_rows:
            return

        with open(self.csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(self.pending_rows)
        self.pending_rows.clear()

    def _write_summary(self):
        if self.measured_count <= 0:
            summary = [
                f"label={self.label}",
                "measured_steps=0",
                f"warmup_steps={self.warmup_steps}",
            ]
        else:
            avg = self.measured_total / self.measured_count
            fps = (1.0 / avg) if avg > 0.0 else 0.0
            summary = [
                f"label={self.label}",
                f"measured_steps={self.measured_count}",
                f"warmup_steps={self.warmup_steps}",
                f"avg_step_seconds={avg:.9f}",
                f"min_step_seconds={self.measured_min:.9f}",
                f"max_step_seconds={self.measured_max:.9f}",
                f"avg_fps={fps:.6f}",
                f"csv_path={self.csv_path}",
            ]

        with open(self.summary_path, "w", encoding="utf-8") as summary_file:
            summary_file.write("\n".join(summary) + "\n")

    def _finalize(self):
        self._flush_rows()
        self._write_summary()

    def onAnimateBeginEvent(self, event):
        self.step_start_time = time.perf_counter()

    def onAnimateEndEvent(self, event):
        if self.step_start_time is None:
            return

        duration = time.perf_counter() - self.step_start_time
        included = self.current_step >= self.warmup_steps
        self.pending_rows.append([self.current_step, f"{duration:.9f}", int(included)])

        if included:
            self.measured_count += 1
            self.measured_total += duration
            self.measured_min = duration if self.measured_min is None else min(self.measured_min, duration)
            self.measured_max = duration if self.measured_max is None else max(self.measured_max, duration)

            if self.log_interval > 0 and self.measured_count % self.log_interval == 0:
                avg = self.measured_total / self.measured_count
                fps = (1.0 / avg) if avg > 0.0 else 0.0
                print(
                    f"[BenchmarkTimingController] {self.label}: "
                    f"measured_steps={self.measured_count}, avg_step={avg:.6f}s, avg_fps={fps:.3f}"
                )

        if self.flush_interval > 0 and len(self.pending_rows) >= self.flush_interval:
            self._flush_rows()
            self._write_summary()

        self.current_step += 1
        self.step_start_time = None
