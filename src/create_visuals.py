#!/usr/bin/env python3
"""Create report-ready plots, natural mechanism drawings, and animations."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    ANIMATION_DIR,
    ARTIFACT_DIR,
    DATA_DIR,
    FAMILIES,
    FIGURE_DIR,
    MASTER_SEED,
    PROTOCOLS,
)


FAMILY_DISPLAY = {
    "fixed_coupling": "Fixed coupling",
    "revolute_hinge": "Revolute hinge",
    "prismatic_guide": "Prismatic guide",
    "ratchet_wheel": "Ratchet wheel",
    "gear_pair": "Gear pair",
}
FAMILY_COLORS = {
    "fixed_coupling": "#515B66",
    "revolute_hinge": "#2A7F8E",
    "prismatic_guide": "#D9902F",
    "ratchet_wheel": "#C65353",
    "gear_pair": "#4F8A48",
}
REP_COLORS = {"absolute": "#6E7785", "relational": "#2A7F8E"}
BACKGROUND = "#F7F8F5"
INK = "#22272B"
SECONDARY = "#D9902F"
INPUT = "#C53E4A"
RESPONSE = "#169B39"


# -----------------------------------------------------------------------------
# Shared plot styling and artifact helpers
# -----------------------------------------------------------------------------

def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#AAB0B5",
            "axes.linewidth": 0.8,
            "axes.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
            "grid.color": "#DDE1E3",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": BACKGROUND,
        }
    )


def read_csv(name):
    with (ARTIFACT_DIR / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, stem):
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Quantitative result figures
# -----------------------------------------------------------------------------

def plot_active_probe_curve():
    summary = json.loads((ARTIFACT_DIR / "analysis_summary.json").read_text())
    rows = summary["classical_active_probe_curve"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    mapping = (("absolute", "Absolute summaries", "#6E7785"), ("relational_combined", "Relational summaries", "#2A7F8E"))
    for representation, label, color in mapping:
        selected = [
            row for row in rows if row["representation"] == representation and row["model"] == "logistic"
        ]
        selected.sort(key=lambda row: row["active_probe_count"])
        x = np.array([row["active_probe_count"] for row in selected])
        mean = np.array([row["mean_accuracy_across_probe_subsets"] for row in selected])
        low = np.array([row["minimum_accuracy"] for row in selected])
        high = np.array([row["maximum_accuracy"] for row in selected])
        ax.plot(x, mean, marker="o", linewidth=2.2, color=color, label=label)
        ax.fill_between(x, low, high, color=color, alpha=0.13)
    ax.axhline(0.2, color=INK, linestyle="--", linewidth=1, label="Chance")
    ax.set(
        title="Active probes make hidden constraints identifiable",
        xlabel="Number of active probes observed",
        ylabel="OOD family accuracy",
        xticks=range(5),
        ylim=(0.12, 1.04),
    )
    ax.grid(axis="y")
    ax.legend(loc="lower right")
    save_figure(fig, "active_probe_accuracy")


def plot_neural_performance():
    latent_rows = read_csv("neural_latent_metrics.csv")
    prediction_rows = read_csv("neural_prediction_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for representation in ("absolute", "relational"):
        latent_means = []
        latent_stds = []
        for budget in range(5):
            values = [
                float(row["accuracy"])
                for row in latent_rows
                if row["representation"] == representation
                and int(row["context_active_probes"]) == budget
            ]
            latent_means.append(np.mean(values))
            latent_stds.append(np.std(values))
        axes[0].errorbar(
            range(5),
            latent_means,
            yerr=latent_stds,
            marker="o",
            capsize=3,
            linewidth=2,
            color=REP_COLORS[representation],
            label=representation.title(),
        )

        prediction_means = []
        prediction_stds = []
        for budget in range(4):
            values = [
                float(row["normalized_rmse"])
                for row in prediction_rows
                if row["representation"] == representation
                and row["split"] == "ood_test"
                and int(row["context_active_probes"]) == budget
            ]
            prediction_means.append(np.mean(values))
            prediction_stds.append(np.std(values))
        axes[1].errorbar(
            range(4),
            prediction_means,
            yerr=prediction_stds,
            marker="o",
            capsize=3,
            linewidth=2,
            color=REP_COLORS[representation],
            label=representation.title(),
        )
    axes[0].axhline(0.2, color=INK, linestyle="--", linewidth=1)
    axes[0].set(
        title="Linear probe on learned latent",
        xlabel="Active probes encoded",
        ylabel="OOD family accuracy",
        xticks=range(5),
        ylim=(0.12, 1.04),
    )
    axes[1].set(
        title="Predicting a held-out probe response",
        xlabel="Other active probes encoded",
        ylabel="Normalized RMSE (lower is better)",
        xticks=range(4),
    )
    for ax in axes:
        ax.grid(axis="y")
        ax.legend()
    fig.suptitle("Predictive encoder-decoder under geometric distribution shift", fontsize=14, y=1.03)
    fig.tight_layout()
    save_figure(fig, "neural_performance")


def plot_ood_slices():
    rows = read_csv("neural_ood_slices.csv")
    regimes = ("pose", "scale", "mass", "shape", "combined")
    x = np.arange(len(regimes))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for offset, representation in enumerate(("absolute", "relational")):
        selected = [row for row in rows if row["representation"] == representation]
        accuracy = []
        accuracy_std = []
        rmse = []
        rmse_std = []
        for regime in regimes:
            regime_rows = [row for row in selected if row["ood_regime"] == regime]
            accuracy.append(np.mean([float(row["latent_family_accuracy"]) for row in regime_rows]))
            accuracy_std.append(np.std([float(row["latent_family_accuracy"]) for row in regime_rows]))
            rmse.append(np.mean([float(row["held_out_probe_normalized_rmse"]) for row in regime_rows]))
            rmse_std.append(np.std([float(row["held_out_probe_normalized_rmse"]) for row in regime_rows]))
        position = x + (offset - 0.5) * width
        axes[0].bar(
            position,
            accuracy,
            width,
            yerr=accuracy_std,
            capsize=3,
            color=REP_COLORS[representation],
            label=representation.title(),
        )
        axes[1].bar(
            position,
            rmse,
            width,
            yerr=rmse_std,
            capsize=3,
            color=REP_COLORS[representation],
            label=representation.title(),
        )
    axes[0].axhline(0.2, color=INK, linestyle="--", linewidth=1)
    axes[0].set(title="Latent identification", ylabel="Accuracy", ylim=(0, 1.06))
    axes[1].set(title="Held-out response prediction", ylabel="Normalized RMSE")
    for ax in axes:
        ax.set_xticks(x, [name.title() for name in regimes], rotation=20)
        ax.grid(axis="y")
        ax.legend()
    fig.suptitle("What changes under each OOD shift", fontsize=14, y=1.03)
    fig.tight_layout()
    save_figure(fig, "ood_slices")


def plot_latent_pca():
    stored = np.load(ARTIFACT_DIR / "learned_latents.npz")
    labels = stored["labels"]
    splits = stored["split_codes"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, representation in zip(axes, ("absolute", "relational")):
        latent = stored[f"{representation}_seed{MASTER_SEED}"]
        projection = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(latent))
        for family_index, family in enumerate(FAMILIES):
            development = (labels == family_index) & (splits != 2)
            test = (labels == family_index) & (splits == 2)
            ax.scatter(
                projection[development, 0],
                projection[development, 1],
                s=14,
                alpha=0.28,
                color=FAMILY_COLORS[family],
            )
            ax.scatter(
                projection[test, 0],
                projection[test, 1],
                s=35,
                marker="x",
                linewidth=1.4,
                color=FAMILY_COLORS[family],
                label=FAMILY_DISPLAY[family] if representation == "relational" else None,
            )
        ax.set_title(f"{representation.title()} latent")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(alpha=0.5)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Learned mechanism codes: development points and OOD crosses", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "learned_latent_pca")


def plot_response_signatures():
    stored = np.load(DATA_DIR / "active_constraint_space_v2.npz")
    names = stored["relational_feature_names"].tolist()
    index = {name: names.index(name) for name in names}
    labels = stored["family_labels"]
    protocols = stored["protocol_labels"]
    measurements = (
        ("push_x translation", 1, ("primary_relative_velocity_x", "primary_relative_velocity_y")),
        ("push_x rotation", 1, ("primary_relative_angular_velocity",)),
        ("CCW rotation", 3, ("primary_relative_angular_velocity",)),
        ("CW rotation", 4, ("primary_relative_angular_velocity",)),
        ("follower rotation", 3, ("secondary_relative_angular_velocity",)),
        ("linear reaction", 2, ("linear_reaction_normalized",)),
        ("angular reaction", 4, ("angular_reaction_normalized",)),
    )
    matrix = np.zeros((len(FAMILIES), len(measurements)))
    for family_index in range(len(FAMILIES)):
        for column, (_, protocol, channels) in enumerate(measurements):
            mask = (labels == family_index) & (protocols == protocol)
            values = stored["relational"][mask][:, :, [index[name] for name in channels]]
            if len(channels) == 2:
                activity = np.linalg.norm(values, axis=2)
            else:
                activity = np.abs(values[:, :, 0])
            matrix[family_index, column] = np.median(np.mean(activity, axis=1))
    normalized = matrix / np.maximum(matrix.max(axis=0, keepdims=True), 1e-8)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    image = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(measurements)), [name for name, _, _ in measurements], rotation=28, ha="right")
    ax.set_yticks(range(len(FAMILIES)), [FAMILY_DISPLAY[name] for name in FAMILIES])
    ax.set_title("Normalized intervention-response signatures")
    for row in range(len(FAMILIES)):
        for column in range(len(measurements)):
            ax.text(
                column,
                row,
                f"{normalized[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if normalized[row, column] < 0.45 else "#111111",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Column-normalized response")
    fig.tight_layout()
    save_figure(fig, "response_signatures")


# -----------------------------------------------------------------------------
# Mechanism rendering primitives
# -----------------------------------------------------------------------------

def draw_rotated_rectangle(ax, center, width, height, angle, **kwargs):
    rectangle = patches.FancyBboxPatch(
        (center[0] - width / 2.0, center[1] - height / 2.0),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={0.08 * min(width, height)}",
        **kwargs,
    )
    rectangle.set_transform(
        transforms.Affine2D().rotate_deg_around(center[0], center[1], math.degrees(angle))
        + ax.transData
    )
    ax.add_patch(rectangle)
    return rectangle


def gear_polygon(center, radius, teeth, angle):
    points = []
    teeth = max(int(teeth), 8)
    for index in range(teeth * 2):
        theta = angle + index * math.pi / teeth
        local_radius = radius if index % 2 == 0 else radius * 0.84
        points.append((center[0] + local_radius * math.cos(theta), center[1] + local_radius * math.sin(theta)))
    return points


def draw_input(ax, raw_frame, raw_index, scale):
    force = np.array((raw_frame[raw_index["input_force_x"]], raw_frame[raw_index["input_force_y"]]))
    torque = float(raw_frame[raw_index["input_torque"]])
    point = np.array((raw_frame[raw_index["input_point_x"]], raw_frame[raw_index["input_point_y"]]))
    norm = np.linalg.norm(force)
    if norm > 1e-7:
        direction = force / norm
        start = point - direction * 0.85 * scale
        ax.annotate(
            "",
            xy=point,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": INPUT, "lw": 2.0},
            zorder=12,
        )
    if abs(torque) > 1e-7:
        sign = 1.0 if torque > 0 else -1.0
        start_angle = 35 if sign > 0 else 325
        end_angle = 325 if sign > 0 else 35
        arc = patches.Arc(
            point,
            1.25 * scale,
            1.25 * scale,
            theta1=min(start_angle, end_angle),
            theta2=max(start_angle, end_angle),
            color=INPUT,
            lw=2.0,
            zorder=12,
        )
        ax.add_patch(arc)
        theta = math.radians(end_angle)
        endpoint = point + 0.625 * scale * np.array((math.cos(theta), math.sin(theta)))
        tangent = sign * np.array((-math.sin(theta), math.cos(theta)))
        ax.annotate(
            "",
            xy=endpoint,
            xytext=endpoint - 0.22 * scale * tangent,
            arrowprops={"arrowstyle": "-|>", "color": INPUT, "lw": 2.0},
            zorder=13,
        )


def draw_mechanism(ax, family, pose, metadata, raw_frame=None, raw_index=None, trail=None):
    base, primary, secondary = pose
    base_center, base_angle = base[:2], base[2]
    primary_center, primary_angle = primary[:2], primary[2]
    secondary_center, secondary_angle = secondary[:2], secondary[2]
    scale = metadata["scale"]

    draw_rotated_rectangle(
        ax,
        base_center,
        metadata["base_width"],
        metadata["base_height"],
        base_angle,
        facecolor="#DDE1DE",
        edgecolor="#60676C",
        linewidth=1.3,
        zorder=1,
    )
    if trail is not None and len(trail):
        ax.plot(trail[:, 0], trail[:, 1], color=FAMILY_COLORS[family], alpha=0.35, lw=1.2, zorder=2)

    if family == "fixed_coupling":
        ax.plot(
            [primary_center[0], secondary_center[0]],
            [primary_center[1], secondary_center[1]],
            color="#747D83",
            lw=7,
            solid_capstyle="round",
            zorder=3,
        )
        for center, color in ((primary_center, FAMILY_COLORS[family]), (secondary_center, SECONDARY)):
            ax.add_patch(patches.Circle(center, 0.38 * scale, facecolor=color, edgecolor=INK, lw=1.4, zorder=5))
            for theta in np.linspace(0, 2 * math.pi, 5)[:-1]:
                bolt = center + 0.23 * scale * np.array((math.cos(theta), math.sin(theta)))
                ax.add_patch(patches.Circle(bolt, 0.045 * scale, facecolor=BACKGROUND, edgecolor=INK, lw=0.7, zorder=6))
    elif family == "revolute_hinge":
        draw_rotated_rectangle(
            ax,
            primary_center,
            metadata["primary_width"],
            max(metadata["primary_height"], 0.30 * scale),
            primary_angle,
            facecolor=FAMILY_COLORS[family],
            edgecolor=INK,
            linewidth=1.3,
            zorder=4,
        )
        ax.add_patch(patches.Circle(primary_center, 0.20 * scale, facecolor="#E5B25D", edgecolor=INK, lw=1.2, zorder=7))
        ax.add_patch(patches.Circle(primary_center, 0.07 * scale, facecolor=INK, zorder=8))
        ax.add_patch(patches.Circle(secondary_center, 0.20 * scale, facecolor=SECONDARY, edgecolor=INK, lw=1.0, zorder=4))
    elif family == "prismatic_guide":
        axis = np.array((math.cos(base_angle), math.sin(base_angle)))
        normal = np.array((-axis[1], axis[0]))
        half = 0.55 * metadata["travel"]
        for offset in (-0.34 * scale, 0.34 * scale):
            start = base_center - half * axis + offset * normal
            end = base_center + half * axis + offset * normal
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#60676C", lw=3.0, zorder=4)
        for sign in (-1, 1):
            center = base_center + sign * half * axis
            ax.plot(
                [center[0] - 0.42 * scale * normal[0], center[0] + 0.42 * scale * normal[0]],
                [center[1] - 0.42 * scale * normal[1], center[1] + 0.42 * scale * normal[1]],
                color=INK,
                lw=3,
                zorder=5,
            )
        draw_rotated_rectangle(
            ax,
            primary_center,
            0.65 * metadata["primary_width"],
            0.85 * metadata["primary_height"],
            base_angle,
            facecolor=FAMILY_COLORS[family],
            edgecolor=INK,
            linewidth=1.3,
            zorder=6,
        )
    elif family == "ratchet_wheel":
        radius = 0.48 * metadata["primary_width"]
        ax.add_patch(
            patches.Polygon(
                gear_polygon(primary_center, radius, metadata["teeth_primary"], primary_angle),
                closed=True,
                facecolor=FAMILY_COLORS[family],
                edgecolor=INK,
                lw=1.2,
                zorder=4,
            )
        )
        ax.add_patch(patches.Circle(primary_center, 0.18 * scale, facecolor="#E5B25D", edgecolor=INK, lw=1.0, zorder=6))
        direction = primary_center - secondary_center
        direction /= max(np.linalg.norm(direction), 1e-8)
        pawl_tip = secondary_center + direction * 0.60 * scale
        normal = np.array((-direction[1], direction[0]))
        pawl_points = [
            secondary_center + 0.16 * scale * normal,
            secondary_center - 0.16 * scale * normal,
            pawl_tip,
        ]
        ax.add_patch(patches.Polygon(pawl_points, closed=True, facecolor=SECONDARY, edgecolor=INK, lw=1.2, zorder=7))
        ax.add_patch(patches.Circle(secondary_center, 0.10 * scale, facecolor=INK, zorder=8))
    elif family == "gear_pair":
        for center, angle, width, teeth, color in (
            (primary_center, primary_angle, metadata["primary_width"], metadata["teeth_primary"], FAMILY_COLORS[family]),
            (secondary_center, secondary_angle, metadata["secondary_width"], metadata["teeth_secondary"], SECONDARY),
        ):
            ax.add_patch(
                patches.Polygon(
                    gear_polygon(center, 0.48 * width, teeth, angle),
                    closed=True,
                    facecolor=color,
                    edgecolor=INK,
                    lw=1.2,
                    zorder=5,
                )
            )
            ax.add_patch(patches.Circle(center, 0.13 * scale, facecolor="#F1D08A", edgecolor=INK, lw=1.0, zorder=7))

    if raw_frame is not None:
        draw_input(ax, raw_frame, raw_index, scale)
    ax.set_aspect("equal")
    ax.set_facecolor(BACKGROUND)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# -----------------------------------------------------------------------------
# Dataset loading and static explanatory figures
# -----------------------------------------------------------------------------

def load_metadata():
    records = []
    with (DATA_DIR / "trajectory_metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def bounds_for(poses, margin=0.8):
    x = poses[..., 0]
    y = poses[..., 1]
    span = max(float(np.ptp(x)), float(np.ptp(y)), 2.0)
    center = (float((x.min() + x.max()) / 2.0), float((y.min() + y.max()) / 2.0))
    radius = 0.65 * span + margin
    return (center[0] - radius, center[0] + radius, center[1] - radius, center[1] + radius)


def create_mechanism_atlas(stored, metadata, raw_index):
    protocols = (0, 3, 4)
    column_names = ("Rest", "Counter-clockwise load", "Clockwise load")
    assembly_index = 12
    frame = 52
    fig, axes = plt.subplots(len(FAMILIES), len(protocols), figsize=(10.5, 14))
    for family_index, family in enumerate(FAMILIES):
        trajectory_base = (family_index * 80 + assembly_index) * len(PROTOCOLS)
        selected_poses = stored["poses"][trajectory_base + np.array(protocols)]
        limits = bounds_for(selected_poses)
        for column, protocol in enumerate(protocols):
            trajectory = trajectory_base + protocol
            ax = axes[family_index, column]
            draw_mechanism(
                ax,
                family,
                stored["poses"][trajectory, frame],
                metadata[trajectory],
                stored["raw"][trajectory, frame],
                raw_index,
                stored["poses"][trajectory, : frame + 1, 1, :2],
            )
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
            if family_index == 0:
                ax.set_title(column_names[column])
            if column == 0:
                ax.text(
                    -0.08,
                    0.5,
                    FAMILY_DISPLAY[family],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=11,
                    fontweight="bold",
                    color=FAMILY_COLORS[family],
                )
    fig.suptitle("Five naturalized assemblies in the matched experimental rig", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0.04, 0, 1, 0.98))
    save_figure(fig, "mechanism_atlas")


def create_relational_graph():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    positions = {"base": (0.18, 0.52), "primary": (0.52, 0.72), "secondary": (0.80, 0.36)}
    colors = {"base": "#60676C", "primary": "#2A7F8E", "secondary": SECONDARY}
    for name, position in positions.items():
        ax.add_patch(patches.Circle(position, 0.085, facecolor=colors[name], edgecolor=INK, lw=1.5))
        ax.text(position[0], position[1], name.title(), color="white", ha="center", va="center", fontweight="bold")
    edges = (
        ("base", "primary", "relative pose\nvelocity + reaction"),
        ("base", "secondary", "relative pose\ngeometry"),
        ("primary", "secondary", "distance + angle\ntransmission response"),
    )
    for source, target, label in edges:
        start = np.array(positions[source])
        end = np.array(positions[target])
        midpoint = 0.5 * (start + end)
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "<->", "color": "#5F686E", "lw": 1.6})
        ax.text(midpoint[0], midpoint[1] + 0.035, label, ha="center", va="bottom", fontsize=9, color=INK)
    ax.annotate(
        "active force / torque",
        xy=(0.52, 0.81),
        xytext=(0.52, 0.97),
        ha="center",
        color=INPUT,
        arrowprops={"arrowstyle": "-|>", "color": INPUT, "lw": 2},
    )
    ax.text(0.5, 0.10, "Node attributes: normalized shape, dimensions, mass and inertia", ha="center", color=INK)
    ax.set_title("Relational measurement graph", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    save_figure(fig, "relational_measurement_graph")


# -----------------------------------------------------------------------------
# Family animations and complete visualization driver
# -----------------------------------------------------------------------------

def circular_axes_arrow(ax, center, radius, clockwise, color, width=2.4):
    """Draw one continuous curved arrow in axes coordinates."""
    start_angle, end_angle = ((145, 25) if clockwise else (25, 145))
    start = np.asarray(center) + radius * np.array(
        (math.cos(math.radians(start_angle)), math.sin(math.radians(start_angle)))
    )
    end = np.asarray(center) + radius * np.array(
        (math.cos(math.radians(end_angle)), math.sin(math.radians(end_angle)))
    )
    arrow = patches.FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        connectionstyle=f"arc3,rad={-0.42 if clockwise else 0.42}",
        arrowstyle="-|>",
        mutation_scale=12,
        color=color,
        linewidth=width,
        shrinkA=0,
        shrinkB=0,
        zorder=31,
    )
    ax.add_patch(arrow)


def circular_data_arrow(ax, center, radius, clockwise, color, width=2.2):
    """Draw one continuous response arrow around a rotating component."""
    start_angle, end_angle = ((145, 25) if clockwise else (25, 145))
    start = np.asarray(center) + radius * np.array(
        (math.cos(math.radians(start_angle)), math.sin(math.radians(start_angle)))
    )
    end = np.asarray(center) + radius * np.array(
        (math.cos(math.radians(end_angle)), math.sin(math.radians(end_angle)))
    )
    arrow = patches.FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={-0.42 if clockwise else 0.42}",
        arrowstyle="-|>",
        mutation_scale=12,
        color=color,
        linewidth=width,
        shrinkA=0,
        shrinkB=0,
        zorder=31,
    )
    ax.add_patch(arrow)


def add_visual_cues(ax, family, protocol, pose, initial_pose):
    """Add unclipped red probe cues and green response-direction cues."""
    if protocol == "push_x":
        ax.annotate(
            "",
            xy=(0.30, 0.86),
            xytext=(0.15, 0.76),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": INPUT, "lw": 2.4},
            zorder=31,
        )
    elif protocol == "push_y":
        ax.annotate(
            "",
            xy=(0.20, 0.86),
            xytext=(0.20, 0.67),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": INPUT, "lw": 2.4},
            zorder=31,
        )
    elif protocol == "torque_ccw":
        circular_axes_arrow(ax, (0.22, 0.72), 0.115, False, INPUT, 2.4)
    elif protocol == "torque_cw":
        circular_axes_arrow(ax, (0.22, 0.72), 0.115, True, INPUT, 2.4)

    angle_delta = float(pose[1, 2] - initial_pose[1, 2])
    displacement = np.asarray(pose[1, :2]) - np.asarray(initial_pose[1, :2])
    span = max(
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
    )
    if family == "revolute_hinge" and abs(angle_delta) > 0.015:
        circular_data_arrow(
            ax,
            pose[1, :2],
            0.12 * span,
            angle_delta < 0,
            RESPONSE,
        )
    elif family == "prismatic_guide" and np.linalg.norm(displacement) > 0.02:
        base_angle = float(pose[0, 2])
        axis = np.array((math.cos(base_angle), math.sin(base_angle)))
        sign = 1.0 if float(np.dot(displacement, axis)) >= 0 else -1.0
        center = np.asarray(pose[1, :2])
        ax.annotate(
            "",
            xy=center + sign * 0.40 * axis,
            xytext=center - sign * 0.40 * axis,
            arrowprops={"arrowstyle": "-|>", "color": RESPONSE, "lw": 2.4},
            zorder=30,
        )
    elif family == "ratchet_wheel" and abs(angle_delta) > 0.015:
        circular_data_arrow(
            ax,
            pose[1, :2],
            0.14 * span,
            angle_delta < 0,
            RESPONSE,
        )
    elif family == "gear_pair" and abs(angle_delta) > 0.015:
        circular_data_arrow(
            ax,
            pose[1, :2],
            0.15 * span,
            angle_delta < 0,
            RESPONSE,
        )
        secondary_delta = float(pose[2, 2] - initial_pose[2, 2])
        if abs(secondary_delta) > 0.015:
            circular_data_arrow(
                ax,
                pose[2, :2],
                0.105 * span,
                secondary_delta < 0,
                RESPONSE,
            )

def create_family_animation(stored, metadata, raw_index, family_index):
    family = FAMILIES[family_index]
    assembly_index = 18
    active_protocols = (1, 2, 3, 4)
    base = (family_index * 80 + assembly_index) * len(PROTOCOLS)
    trajectories = [base + protocol for protocol in active_protocols]
    selected_poses = stored["poses"][trajectories]
    limits = bounds_for(selected_poses, margin=0.6 * metadata[base]["scale"])
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.2))
    axes = axes.ravel()
    frame_indices = np.arange(0, stored["poses"].shape[1], 2)

    def update(animation_frame):
        frame = int(frame_indices[animation_frame])
        for ax, trajectory, protocol in zip(axes, trajectories, active_protocols):
            ax.clear()
            draw_mechanism(
                ax,
                family,
                stored["poses"][trajectory, frame],
                metadata[trajectory],
                None,
                None,
                stored["poses"][trajectory, : frame + 1, 1, :2],
            )
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
            add_visual_cues(
                ax,
                family,
                PROTOCOLS[protocol],
                stored["poses"][trajectory, frame],
                stored["poses"][trajectory, 0],
            )
            ax.set_title(PROTOCOLS[protocol].replace("_", " ").title(), fontsize=10)
        fig.suptitle(f"{FAMILY_DISPLAY[family]}: responses from one matched assembly", fontsize=14)
        return axes

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=50, blit=False)
    animation.save(
        ANIMATION_DIR / f"{family}_active_probes.gif",
        writer=PillowWriter(fps=20),
        dpi=110,
    )
    plt.close(fig)


def main():
    style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    plot_active_probe_curve()
    plot_neural_performance()
    plot_ood_slices()
    plot_latent_pca()
    plot_response_signatures()
    create_relational_graph()

    stored = np.load(DATA_DIR / "active_constraint_space_v2.npz")
    metadata = load_metadata()
    raw_names = stored["raw_feature_names"].tolist()
    raw_index = {name: raw_names.index(name) for name in raw_names}
    create_mechanism_atlas(stored, metadata, raw_index)
    for family_index in range(len(FAMILIES)):
        create_family_animation(stored, metadata, raw_index, family_index)
        print(f"Rendered {FAMILIES[family_index]}")
    print(f"Created figures in {FIGURE_DIR} and animations in {ANIMATION_DIR}")


if __name__ == "__main__":
    main()
