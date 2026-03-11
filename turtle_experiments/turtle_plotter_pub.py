#!/usr/bin/env python3
"""
turtle_plotter.py
=================

Publication-grade plotting utility for the Turtlesim control experiments.

Key goals
---------
1) Produce figures that fit within LaTeX margins when included without manual scaling.
2) Guarantee export resolution >= 300 dpi for raster formats.
3) Optionally export vector PDF for best print quality.

How this script avoids "figure exceeds margins" in LaTeX
-------------------------------------------------------
LaTeX uses the natural size of bitmap figures when no width/scale is specified.
If the PNG has a huge pixel count and/or missing/odd DPI metadata, the natural size
can become larger than \\textwidth, causing overflow.

This script therefore:
- Fixes the physical figure size in inches to a width compatible with the paper text block.
- Saves with an explicit DPI (default 300) so the natural size is consistent.
- Uses bbox_inches="tight" to minimize surrounding whitespace.

Recommended LaTeX inclusion (robust)
------------------------------------
Even with correctly sized images, best practice is still:
  \\includegraphics[width=\\linewidth]{figures/flowchart.pdf}

But if you omit width, the exported figures should remain within typical margins.

Usage examples
--------------
Single-column layout (default, ~160 mm textwidth):
  python3 turtle_plotter.py

Two-column layout (half width):
  python3 turtle_plotter.py --layout two-column

Explicit textwidth (mm) if your template differs:
  python3 turtle_plotter.py --textwidth-mm 170

Export only PDF (vector):
  python3 turtle_plotter.py --formats pdf

Export both PDF and PNG at 300 dpi:
  python3 turtle_plotter.py --formats pdf png --dpi 300
"""

from __future__ import annotations

import argparse
import json
import math
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ----------------------------- plotting style ----------------------------- #

def set_pub_style() -> None:
    """
    Matplotlib style tuned for journal figures.
    The font sizes are chosen to remain readable when the figure is 160 mm wide.
    """
    rcParams.update({
        # Layout
        "figure.constrained_layout.use": True,
        "axes.grid": True,
        "grid.alpha": 0.25,

        # Fonts
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,

        # Lines
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.8,
    })


def legend_right_of_axes(ax: plt.Axes, fig: Optional[plt.Figure] = None, right_margin: float = 0.78) -> None:
    """Place legend to the right of the axes, outside the plotting region."""
    if fig is not None:
        # constrained_layout conflicts with manual subplots_adjust
        fig.set_constrained_layout(False)
        fig.subplots_adjust(right=right_margin)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0.0)


# --------------------------- configuration model -------------------------- #

@dataclass(frozen=True)
class FigureSpec:
    width_in: float
    height_in: float


@dataclass(frozen=True)
class ExportConfig:
    output_dir: Path
    formats: Tuple[str, ...]  # ("pdf", "png")
    dpi: int
    bbox_inches: str = "tight"
    pad_inches: float = 0.02


@dataclass(frozen=True)
class LayoutConfig:
    """
    Defines figure width for single-column vs two-column LaTeX layouts.
    - single-column: figures targeted to ~\\textwidth
    - two-column: figures targeted to ~0.48\\textwidth
    """
    textwidth_mm: float
    layout: str  # "single-column" or "two-column"

    @property
    def textwidth_in(self) -> float:
        return self.textwidth_mm / 25.4

    @property
    def figure_width_in(self) -> float:
        if self.layout == "two-column":
            return 0.48 * self.textwidth_in
        return self.textwidth_in


# ------------------------------ data loading ------------------------------ #

def _first_existing(patterns: Iterable[str]) -> str:
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No file matches patterns: {list(patterns)}")


def load_experiment_data(experiment_name: str) -> Dict[str, np.ndarray]:
    """
    Load one experiment JSON file and normalize array lengths.
    """
    json_file = _first_existing([
        f"turtle_experiments/{experiment_name}.json",
        f"{experiment_name}.json",
    ])

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data["metadata"]
    d = data["data"]

    # Required series
    time = np.asarray(d["time"], dtype=float)
    x = np.asarray(d["pos_x"], dtype=float)
    y = np.asarray(d["pos_y"], dtype=float)
    theta_rad = np.asarray(d["pos_theta"], dtype=float)
    theta_deg = np.degrees(theta_rad)

    error_x = np.asarray(d["error_x"], dtype=float)
    error_y = np.asarray(d["error_y"], dtype=float)
    if "error_distance" in d:
        error_dist = np.asarray(d["error_distance"], dtype=float)
    else:
        error_dist = np.sqrt(error_x**2 + error_y**2)

    v_lin = np.asarray(d["vel_linear_x"], dtype=float)
    w_ang = np.asarray(d["vel_angular_z"], dtype=float)

    vxG = np.asarray(d.get("vel_x_global", np.zeros_like(time)), dtype=float)
    vyG = np.asarray(d.get("vel_y_global", np.zeros_like(time)), dtype=float)

    # Normalize lengths (truncate to minimum)
    series = {
        "time": time,
        "x": x,
        "y": y,
        "theta_deg": theta_deg,
        "theta_rad": theta_rad,
        "error_x": error_x,
        "error_y": error_y,
        "error_distance": error_dist,
        "linear_vel": v_lin,
        "angular_vel": w_ang,
        "vel_x_global": vxG,
        "vel_y_global": vyG,
    }
    lengths = {k: len(v) for k, v in series.items()}
    min_len = min(lengths.values())

    if len(set(lengths.values())) > 1:
        print(f"Warning: arrays with different sizes in '{experiment_name}'. Truncating to {min_len} samples.")
        series = {k: v[:min_len] for k, v in series.items()}

    # Attach metadata
    out: Dict[str, np.ndarray] = dict(series)
    out["target_x"] = float(metadata["target_position"]["x"])
    out["target_y"] = float(metadata["target_position"]["y"])
    out["Kp_angular"] = float(metadata["controller_gains"]["Kp_angular"])
    out["Kp_linear"] = float(metadata["controller_gains"]["Kp_linear"])
    out["total_time"] = float(series["time"][-1]) if min_len else 0.0
    out["final_error"] = float(series["error_distance"][-1]) if min_len else 0.0
    out["sample_count"] = int(min_len)
    out["prefix"] = experiment_name
    out["experiment_name"] = metadata.get("experiment_name", experiment_name)

    return out


# ------------------------------ resampling -------------------------------- #

def resample_with_constant_extrapolation(
    t_old: np.ndarray, y_old: np.ndarray, t_new: np.ndarray
) -> np.ndarray:
    """
    Linear interpolation; outside the original time range, uses the first/last value (constant extrapolation).
    This avoids SciPy as a dependency and keeps the "align to 15s" behavior stable.
    """
    if len(t_old) == 0:
        return np.zeros_like(t_new)
    if len(t_old) == 1:
        return np.full_like(t_new, y_old[0])

    # Ensure increasing time (robustness)
    order = np.argsort(t_old)
    t_old_s = t_old[order]
    y_old_s = y_old[order]

    y_new = np.interp(t_new, t_old_s, y_old_s)

    # Constant extrapolation on both ends
    y_new = np.where(t_new < t_old_s[0], y_old_s[0], y_new)
    y_new = np.where(t_new > t_old_s[-1], y_old_s[-1], y_new)

    return y_new


def align_experiments(
    experiments: List[Dict[str, np.ndarray]],
    target_time: float = 15.0,
    samples: int = 1500,
) -> List[Dict[str, np.ndarray]]:
    """
    Resample all experiments onto a common time grid [0, target_time] with constant extrapolation.
    """
    new_time = np.linspace(0.0, float(target_time), int(samples))

    keys_to_resample = [
        "x", "y", "theta_deg", "theta_rad", "error_distance", "linear_vel", "angular_vel",
        "error_x", "error_y", "vel_x_global", "vel_y_global"
    ]

    aligned: List[Dict[str, np.ndarray]] = []
    for exp in experiments:
        t_old = exp["time"]
        new_exp = dict(exp)
        new_exp["original_time"] = float(exp.get("total_time", t_old[-1] if len(t_old) else 0.0))
        new_exp["time"] = new_time
        new_exp["total_time"] = float(target_time)

        for k in keys_to_resample:
            new_exp[k] = resample_with_constant_extrapolation(t_old, exp[k], new_time)

        aligned.append(new_exp)

        # Console report
        mode = "EXTRAPOLATED" if new_exp["original_time"] < target_time else "TRUNCATED"
        print(f"  {exp['prefix']}: {new_exp['original_time']:.2f}s -> {target_time:.2f}s ({mode}), "
              f"{exp['sample_count']} -> {samples} samples")

    return aligned


# ------------------------------ exporting --------------------------------- #

def save_figure(fig: plt.Figure, name: str, export: ExportConfig) -> List[Path]:
    export.output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for fmt in export.formats:
        fmt_l = fmt.lower().strip(".")
        out = export.output_dir / f"{name}.{fmt_l}"

        if fmt_l in ("png", "jpg", "jpeg", "tif", "tiff"):
            fig.savefig(
                out,
                dpi=export.dpi,
                bbox_inches=export.bbox_inches,
                pad_inches=export.pad_inches,
            )
        else:
            # For vector formats, dpi is irrelevant but harmless.
            fig.savefig(
                out,
                bbox_inches=export.bbox_inches,
                pad_inches=export.pad_inches,
            )
        written.append(out)

    return written


# ------------------------------ figure specs ------------------------------ #

def spec_from_width(width_in: float, aspect: float) -> FigureSpec:
    """aspect = height/width"""
    return FigureSpec(width_in=float(width_in), height_in=float(width_in * aspect))


def default_specs(layout: LayoutConfig) -> Dict[str, FigureSpec]:
    """Standardized specs so every plot is consistent in LaTeX."""
    w = layout.figure_width_in
    return {
        # Widest figures (good for flowcharts and trajectories)
        "wide": spec_from_width(w, aspect=0.70),
        "tall": spec_from_width(w, aspect=0.95),

        # Typical time-series
        "timeseries": spec_from_width(w, aspect=0.58),

        # Compact
        "compact": spec_from_width(w, aspect=0.45),
    }


# ------------------------------ plot makers -------------------------------- #

def _sorted_by_kp(experiments: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    return sorted(experiments, key=lambda e: float(e["Kp_linear"]))


def plot_xy_trajectories(experiments: List[Dict[str, np.ndarray]], spec: FigureSpec) -> plt.Figure:
    exps = _sorted_by_kp(experiments)
    fig, ax = plt.subplots(figsize=(spec.width_in, spec.height_in))
    ax.set_aspect("equal", adjustable="box")

    for exp in exps:
        ax.plot(exp["x"], exp["y"], label=f"$K_p={exp['Kp_linear']:.0f}$")
        ax.plot(exp["x"][-1], exp["y"][-1], marker="o", markersize=4)

    tx, ty = exps[0]["target_x"], exps[0]["target_y"]
    ax.plot(tx, ty, marker="x", markersize=7, mew=1.6, label=f"Target ({tx:.1f}, {ty:.1f})")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Planar trajectories (aligned time window)")
    legend_right_of_axes(ax, fig)
    ax.grid(True, alpha=0.25)
    return fig


def plot_error_convergence(
    experiments: List[Dict[str, np.ndarray]],
    spec: FigureSpec,
    max_time: float,
    legend_to_right: bool = False
) -> plt.Figure:
    exps = _sorted_by_kp(experiments)
    fig, ax = plt.subplots(figsize=(spec.width_in, spec.height_in))

    for exp in exps:
        ax.semilogy(exp["time"], exp["error_distance"], label=f"$K_p={exp['Kp_linear']:.0f}$")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Error convergence (log scale)")
    ax.set_xlim(0, max_time)

    if legend_to_right:
        legend_right_of_axes(ax, fig)
    else:
        ax.legend(loc="best", frameon=False)

    ax.grid(True, alpha=0.25, which="both")
    return fig


def plot_timeseries(
    experiments: List[Dict[str, np.ndarray]],
    spec: FigureSpec,
    max_time: float,
    y_key: str,
    y_label: str,
    title: str,
    hline: Optional[float] = None,
    hline_label: Optional[str] = None,
    legend_to_right: bool = False
) -> plt.Figure:
    exps = _sorted_by_kp(experiments)
    fig, ax = plt.subplots(figsize=(spec.width_in, spec.height_in))

    for exp in exps:
        ax.plot(exp["time"], exp[y_key], label=f"$K_p={exp['Kp_linear']:.0f}$")

    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1.0, alpha=0.7, label=hline_label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xlim(0, max_time)

    if legend_to_right:
        legend_right_of_axes(ax, fig)
    else:
        ax.legend(loc="best", frameon=False)

    ax.grid(True, alpha=0.25)
    return fig


def plot_velocity_magnitude(
    experiments: List[Dict[str, np.ndarray]],
    spec: FigureSpec,
    max_time: float,
    legend_to_right: bool = False
) -> plt.Figure:
    exps = _sorted_by_kp(experiments)
    fig, ax = plt.subplots(figsize=(spec.width_in, spec.height_in))

    for exp in exps:
        mag = np.sqrt(exp["vel_x_global"]**2 + exp["vel_y_global"]**2)
        ax.plot(exp["time"], mag, label=f"$K_p={exp['Kp_linear']:.0f}$")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Global velocity magnitude (m/s)")
    ax.set_title("Global velocity magnitude")
    ax.set_xlim(0, max_time)

    if legend_to_right:
        legend_right_of_axes(ax, fig)
    else:
        ax.legend(loc="best", frameon=False)

    ax.grid(True, alpha=0.25)
    return fig


def plot_flowchart_like_summary(experiments: List[Dict[str, np.ndarray]], spec: FigureSpec, max_time: float) -> plt.Figure:
    """
    Compact 2x2 panel comparing experiments (trajectories, error, linear vel, angular vel).
    """
    exps = _sorted_by_kp(experiments)
    tx, ty = exps[0]["target_x"], exps[0]["target_y"]

    fig, axes = plt.subplots(2, 2, figsize=(spec.width_in, spec.height_in))

    # (1) XY trajectory
    ax = axes[0, 0]
    ax.set_aspect("equal", adjustable="box")
    for exp in exps:
        ax.plot(exp["x"], exp["y"], label=f"$K_p={exp['Kp_linear']:.0f}$")
    ax.plot(tx, ty, marker="x", markersize=7, mew=1.6)
    ax.set_title("Trajectories")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

    # (2) Error
    ax = axes[0, 1]
    for exp in exps:
        ax.semilogy(exp["time"], exp["error_distance"], label=f"$K_p={exp['Kp_linear']:.0f}$")
    ax.set_title("Error (log)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.set_xlim(0, max_time)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best", frameon=False)

    # (3) Linear velocity
    ax = axes[1, 0]
    for exp in exps:
        ax.plot(exp["time"], exp["linear_vel"], label=f"$K_p={exp['Kp_linear']:.0f}$")
    ax.set_title("Linear velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("m/s")
    ax.set_xlim(0, max_time)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

    # (4) Angular velocity
    ax = axes[1, 1]
    for exp in exps:
        ax.plot(exp["time"], exp["angular_vel"], label=f"$K_\\theta={exp['Kp_angular']:.0f}$")
    ax.set_title("Angular velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("rad/s")
    ax.set_xlim(0, max_time)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

    fig.suptitle("Experiment comparison (compact panel)", fontsize=10.5)
    return fig


# ------------------------------ reporting --------------------------------- #

def print_summary(experiments: List[Dict[str, np.ndarray]]) -> None:
    print("\n" + "=" * 78)
    print("PERFORMANCE SUMMARY (aligned)")
    print("=" * 78)
    header = f"{'Experiment':<12} {'Kp_lin':>7} {'Kp_ang':>7} {'Orig(s)':>9} {'N':>7} {'Final err':>10}"
    print(header)
    print("-" * 78)

    for exp in _sorted_by_kp(experiments):
        print(
            f"{exp['prefix']:<12} "
            f"{exp['Kp_linear']:>7.0f} "
            f"{exp['Kp_angular']:>7.0f} "
            f"{exp.get('original_time', exp['total_time']):>9.2f} "
            f"{exp['sample_count']:>7d} "
            f"{exp['final_error']:>10.4f}"
        )
    print("=" * 78 + "\n")


# ------------------------------ metrics + LaTeX table ------------------------------ #

def wrap_to_pi(rad: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (rad + np.pi) % (2.0 * np.pi) - np.pi


def compute_metrics_table_latex(aligned: List[Dict[str, np.ndarray]]) -> str:
    """
    Compute metrics for angular and linear motion and return a LaTeX table string.

    Metrics:
      - Settling time (2%): t_s = 4*tau, with tau = 1/Kp (first-order proxy)
      - Steady-state absolute error:
          Angular: |wrap(theta(t_end) - theta_d)| in deg, where theta_d is angle from initial position to target.
          Linear:  || [x(t_end)-x_d, y(t_end)-y_d] || in m
      - Overshoot (%):
          Angular: peak absolute angular error relative to initial error magnitude.
          Linear : peak radial error relative to initial radial error magnitude.

    Note:
      If your controller uses intermediate sub-goals (e.g., (x_d, y_0) then (x_d, y_d)),
      you may prefer to compute angular metrics with respect to the sub-goal for the rotation phase.
    """
    rows_ang = []
    rows_lin = []

    for exp in _sorted_by_kp(aligned):
        t = exp["time"]
        x = exp["x"]
        y = exp["y"]
        theta_rad = exp["theta_rad"]  # aligned/resampled

        xd = float(exp["target_x"])
        yd = float(exp["target_y"])

        # Desired orientation: angle from initial position to final target
        x0 = float(x[0])
        y0 = float(y[0])
        theta_d = math.atan2(yd - y0, xd - x0)

        # Angular error time series
        e_theta = wrap_to_pi(theta_rad - theta_d)
        e_theta_abs_deg = np.degrees(np.abs(e_theta))

        ess_theta_deg = float(e_theta_abs_deg[-1])

        step_theta = float(e_theta_abs_deg[0])
        peak_theta = float(np.max(e_theta_abs_deg))
        os_theta_pct = 0.0
        if step_theta > 1e-12:
            os_theta_pct = max(0.0, 100.0 * (peak_theta - step_theta) / step_theta)

        Kp_ang = float(exp["Kp_angular"])
        tau_ang = 1.0 / Kp_ang if Kp_ang > 0 else float("inf")
        ts_ang = 4.0 * tau_ang

        rows_ang.append((exp["prefix"], Kp_ang, tau_ang, ts_ang, ess_theta_deg, os_theta_pct))

        # Linear error (Euclidean)
        ex = x - xd
        ey = y - yd
        rho = np.sqrt(ex**2 + ey**2)

        ess_lin = float(rho[-1])
        rho0 = float(rho[0])
        rho_peak = float(np.max(rho))
        os_lin_pct = 0.0
        if rho0 > 1e-12:
            os_lin_pct = max(0.0, 100.0 * (rho_peak - rho0) / rho0)

        Kp_lin = float(exp["Kp_linear"])
        tau_lin = 1.0 / Kp_lin if Kp_lin > 0 else float("inf")
        ts_lin = 4.0 * tau_lin

        rows_lin.append((exp["prefix"], Kp_lin, tau_lin, ts_lin, ess_lin, os_lin_pct))

    # Build LaTeX table (booktabs style)
    lines: List[str] = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Settling time ($t_s=4\tau$ for 2\%), steady-state absolute error, and overshoot for angular and linear motions (aligned window).}")
    lines.append(r"\label{tab:metrics_motion}")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{6}{c}{\textbf{Angular motion}}\\")
    lines.append(r"\midrule")
    lines.append(r"Experiment & $K_p$ & $\tau$ (s) & $t_s$ (s) & $|e_\theta(\infty)|$ (deg) & OS (\%)\\")
    lines.append(r"\midrule")
    for (name, kp, tau, ts, ess, os_) in rows_ang:
        lines.append(f"{name} & {kp:.2f} & {tau:.4f} & {ts:.4f} & {ess:.4f} & {os_:.2f}\\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{c}{\textbf{Linear motion}}\\")
    lines.append(r"\midrule")
    lines.append(r"Experiment & $K_p$ & $\tau$ (s) & $t_s$ (s) & $\|\mathbf{e}(\infty)\|$ (m) & OS (\%)\\")
    lines.append(r"\midrule")
    for (name, kp, tau, ts, ess, os_) in rows_lin:
        lines.append(f"{name} & {kp:.2f} & {tau:.4f} & {ts:.4f} & {ess:.6f} & {os_:.2f}\\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# --------------------------------- main ----------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate publication-grade plots for Turtlesim experiments.")
    p.add_argument("--experiments", nargs="+", default=["kp1_210", "kp2_210", "kp4_210", "kp10_210"],
                   help="Experiment name list (JSON files).")
    p.add_argument("--target-time", type=float, default=15.0, help="Common time horizon (s).")
    p.add_argument("--samples", type=int, default=1500, help="Samples for the aligned time grid.")
    p.add_argument("--output-dir", type=str, default="figures_out", help="Output folder for saved figures.")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Export formats: pdf png svg ...")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster export (png/jpg/tiff). Minimum 300.")
    p.add_argument("--layout", choices=["single-column", "two-column"], default="single-column",
                   help="LaTeX column layout: affects default figure width.")
    p.add_argument("--textwidth-mm", type=float, default=160.0,
                   help="Approximate LaTeX \\textwidth in mm (single-column). Default 160 mm.")
    p.add_argument("--print-latex-table", action="store_true",
                   help="If set, print the LaTeX metrics table to stdout.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dpi = int(args.dpi)
    if dpi < 300:
        print(f"Warning: dpi={dpi} < 300. For publication, 300+ is recommended. Setting dpi=300.")
        dpi = 300

    set_pub_style()

    layout = LayoutConfig(textwidth_mm=float(args.textwidth_mm), layout=str(args.layout))
    specs = default_specs(layout)

    export = ExportConfig(
        output_dir=Path(args.output_dir),
        formats=tuple(args.formats),
        dpi=dpi,
    )

    # 1) Load data
    print("=" * 70)
    print("TURTLESIM EXPERIMENT PLOTTER (publication sizing)")
    print("=" * 70)
    experiments: List[Dict[str, np.ndarray]] = []
    for name in args.experiments:
        try:
            exp = load_experiment_data(name)
            experiments.append(exp)
            print(f"  ✓ Loaded {name}: Kp=({exp['Kp_linear']:.0f},{exp['Kp_angular']:.0f}), "
                  f"{exp['total_time']:.2f}s, N={exp['sample_count']}")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")

    if len(experiments) < 1:
        print("ERROR: no experiments loaded.")
        return 2

    # 2) Align time
    print("\nAligning data:")
    aligned = align_experiments(experiments, target_time=float(args.target_time), samples=int(args.samples))
    print_summary(aligned)

    max_time = float(args.target_time)

    # 3) Generate and export figures
    written_all: List[Path] = []

    figs = [
        ("xy_trajectories", plot_xy_trajectories(aligned, specs["wide"])),
        ("error_convergence", plot_error_convergence(aligned, specs["timeseries"], max_time, legend_to_right=False)),
        ("x_position", plot_timeseries(aligned, specs["timeseries"], max_time, "x", "X position (m)",
                                       "X position vs time", hline=float(aligned[0]["target_x"]),
                                       hline_label="Target X")),
        ("y_position", plot_timeseries(aligned, specs["timeseries"], max_time, "y", "Y position (m)",
                                       "Y position vs time", hline=float(aligned[0]["target_y"]),
                                       hline_label="Target Y")),
        ("linear_velocity", plot_timeseries(aligned, specs["timeseries"], max_time, "linear_vel", "Linear velocity (m/s)",
                                            "Linear velocity vs time", legend_to_right=True)),
        ("angular_velocity", plot_timeseries(aligned, specs["timeseries"], max_time, "angular_vel", "Angular velocity (rad/s)",
                                             "Angular velocity vs time", legend_to_right=True)),
        ("orientation", plot_timeseries(aligned, specs["timeseries"], max_time, "theta_deg", "Orientation (deg)",
                                        "Orientation vs time")),
        ("x_error", plot_timeseries(aligned, specs["timeseries"], max_time, "error_x", "X error (m)",
                                    "X error vs time", hline=0.0, hline_label="0")),
        ("y_error", plot_timeseries(aligned, specs["timeseries"], max_time, "error_y", "Y error (m)",
                                    "Y error vs time", hline=0.0, hline_label="0")),
        ("vel_x_global", plot_timeseries(aligned, specs["timeseries"], max_time, "vel_x_global", "Global $v_x$ (m/s)",
                                         "Global $v_x$ vs time", hline=0.0, hline_label="0", legend_to_right=True)),
        # Requested: legend outside (right)
        ("vel_y_global", plot_timeseries(aligned, specs["timeseries"], max_time, "vel_y_global", "Global $v_y$ (m/s)",
                                         "Global $v_y$ vs time", hline=0.0, hline_label="0", legend_to_right=True)),
        # Requested: legend outside (right)
        ("global_velocity_magnitude", plot_velocity_magnitude(aligned, specs["timeseries"], max_time, legend_to_right=True)),
        # Compact multi-panel figure
        ("comparison_panel_compact", plot_flowchart_like_summary(aligned, specs["tall"], max_time)),
    ]

    for name, fig in figs:
        written = save_figure(fig, name, export)
        written_all.extend(written)
        plt.close(fig)

    print("Saved figures:")
    for p in written_all:
        print(f"  - {p}")

    print("\nNotes:")
    print(f"  • Layout: {layout.layout}, assumed textwidth: {layout.textwidth_mm:.1f} mm "
          f"({layout.textwidth_in:.2f} in), figure width: {layout.figure_width_in:.2f} in.")
    print("  • For LaTeX, prefer vector PDF: \\includegraphics[width=\\linewidth]{...}.")
    print("  • If your template text block is wider/narrower, pass --textwidth-mm accordingly.")

    # 4) Print LaTeX table (optional)
    if args.print_latex_table:
        latex_table = compute_metrics_table_latex(aligned)
        print("\nLaTeX table (copy/paste):\n")
        print(latex_table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())