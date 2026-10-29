#!/usr/bin/env python3
"""Create figures designed specifically for the MSc report layout."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DATA_DIR, FAMILIES, PROTOCOLS  # noqa: E402
from create_visuals import (  # noqa: E402
    BACKGROUND,
    FAMILY_COLORS,
    FAMILY_DISPLAY,
    INK,
    INPUT,
    add_visual_cues,
    bounds_for,
    draw_mechanism,
    load_metadata,
    style,
)


REPORT_FIGURE_DIR = Path(__file__).resolve().parent / "figures"


# -----------------------------------------------------------------------------
# Figure export and mechanism motion cues
# -----------------------------------------------------------------------------

def save(fig, stem):
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_FIGURE_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(REPORT_FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_motion_cue(ax, family, pose, scale, protocol):
    primary = np.asarray(pose[1, :2], dtype=float)
    secondary = np.asarray(pose[2, :2], dtype=float)
    if family == "fixed_coupling":
        return
    elif family == "prismatic_guide" and protocol == "push_x":
        ax.annotate(
            "",
            xy=primary + np.array((0.72 * scale, 0.0)),
            xytext=primary - np.array((0.05 * scale, 0.0)),
            arrowprops={"arrowstyle": "-|>", "color": FAMILY_COLORS[family], "lw": 1.8},
            zorder=14,
        )
    elif family == "prismatic_guide":
        return
    elif family == "ratchet_wheel" and protocol == "torque_cw":
        return
    elif family in {"revolute_hinge", "ratchet_wheel"}:
        sign = 1 if protocol == "torque_ccw" else -1
        theta1, theta2 = (25, 320) if sign > 0 else (40, 335)
        arc = patches.Arc(
            primary,
            1.35 * scale,
            1.35 * scale,
            theta1=min(theta1, theta2),
            theta2=max(theta1, theta2),
            color=FAMILY_COLORS[family],
            lw=1.8,
            zorder=14,
        )
        ax.add_patch(arc)
    elif family == "gear_pair":
        for center, sign, radius in (
            (primary, 1 if protocol == "torque_ccw" else -1, 0.72),
            (secondary, -1 if protocol == "torque_ccw" else 1, 0.50),
        ):
            start = center + np.array((-radius * scale, radius * 0.20 * scale))
            end = center + np.array((radius * scale, radius * 0.20 * scale))
            if sign < 0:
                start, end = end, start
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": FAMILY_COLORS[family],
                    "lw": 1.6,
                    "connectionstyle": "arc3,rad=0.65",
                },
                zorder=14,
            )


def create_family_storyboard(stored, metadata, raw_index):
    protocol_display = {
        "push_x": "Push X",
        "push_y": "Push Y",
        "torque_ccw": "Torque CCW",
        "torque_cw": "Torque CW",
    }
    protocol_map = {
        "fixed_coupling": (1, 3),
        "revolute_hinge": (3, 4),
        "prismatic_guide": (1, 2),
        "ratchet_wheel": (3, 4),
        "gear_pair": (3, 4),
    }
    assembly_index = 18
    frame = 52
    fig, axes = plt.subplots(len(FAMILIES), 3, figsize=(10.7, 11.8))

    for family_index, family in enumerate(FAMILIES):
        first, second = protocol_map[family]
        protocols = (0, first, second)
        trajectory_base = (family_index * 80 + assembly_index) * len(PROTOCOLS)
        trajectories = [trajectory_base + protocol for protocol in protocols]
        limits = bounds_for(stored["poses"][trajectories], margin=0.5 * metadata[trajectory_base]["scale"])

        for column, (trajectory, protocol) in enumerate(zip(trajectories, protocols)):
            ax = axes[family_index, column]
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
            if protocol:
                add_visual_cues(
                    ax,
                    family,
                    PROTOCOLS[protocol],
                    stored["poses"][trajectory, frame],
                    stored["poses"][trajectory, 0],
                )
            ax.set_title(
                "Rest" if protocol == 0 else protocol_display[PROTOCOLS[protocol]],
                fontsize=9,
                pad=2,
            )
            if column == 0:
                ax.text(
                    -0.10,
                    0.5,
                    FAMILY_DISPLAY[family],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                    color=FAMILY_COLORS[family],
                )

    fig.suptitle(
        "Static storyboards of the five hidden constraint families",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.50,
        0.006,
        "Red arrows show the applied probe; green arrows show the resulting motion.",
        ha="center",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0.04, 0.02, 1, 0.98), h_pad=1.0)
    save(fig, "family_probe_storyboard")


# -----------------------------------------------------------------------------
# Encoder-decoder architecture diagram
# -----------------------------------------------------------------------------

def rounded_box(ax, xy, width, height, text, face, edge=INK, fontsize=9, weight="normal"):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight=weight,
    )
    return box


def arrow(ax, start, end, color="#5F686E", style="-|>", dashed=False):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "lw": 1.5,
            "linestyle": "--" if dashed else "-",
        },
    )


def create_encoder_decoder_diagram():
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(
        ax,
        (0.03, 0.60),
        0.16,
        0.19,
        "Observed trials\npassive + probes\n$40 \\,\\times\\, D$",
        "#E8ECEC",
        weight="bold",
    )
    rounded_box(
        ax,
        (0.03, 0.27),
        0.16,
        0.17,
        "Assembly geometry\nabsolute or\nrelational",
        "#E8ECEC",
        weight="bold",
    )
    rounded_box(ax, (0.25, 0.60), 0.13, 0.19, "Shared GRU\ntrial encoder\n48 units", "#D8EBEF")
    rounded_box(ax, (0.25, 0.27), 0.13, 0.17, "Geometry MLP\n$D_g\\!\\to\\!48\\!\\to\\!24$", "#D8EBEF")
    rounded_box(ax, (0.44, 0.60), 0.13, 0.19, "Masked mean\nacross observed\nprotocols", "#E1E5E8")
    rounded_box(
        ax,
        (0.44, 0.27),
        0.13,
        0.17,
        "Geometry code\n24 values",
        "#E1E5E8",
    )
    rounded_box(
        ax,
        (0.62, 0.50),
        0.13,
        0.22,
        "Bottleneck\nMLP + tanh\n\n$\\mathbf{z}\\in\\mathbb{R}^{16}$",
        "#F3D9DA",
        edge=INPUT,
        fontsize=10,
        weight="bold",
    )
    rounded_box(
        ax,
        (0.62, 0.17),
        0.13,
        0.18,
        "Held-out query\nprobe + action\n+ time",
        "#FBE9D2",
    )
    rounded_box(
        ax,
        (0.81, 0.43),
        0.15,
        0.25,
        "Probe-conditioned\nMLP decoder\n\nPredicted relative\nresponse $40\\!\\times\\!16$",
        "#DCEAD9",
        edge=FAMILY_COLORS["gear_pair"],
        weight="bold",
    )

    arrow(ax, (0.19, 0.695), (0.25, 0.695))
    arrow(ax, (0.19, 0.355), (0.25, 0.355))
    arrow(ax, (0.38, 0.695), (0.44, 0.695))
    arrow(ax, (0.38, 0.355), (0.44, 0.355))
    arrow(ax, (0.57, 0.695), (0.62, 0.64))
    arrow(ax, (0.57, 0.355), (0.62, 0.56))
    arrow(ax, (0.75, 0.61), (0.81, 0.58))
    arrow(ax, (0.75, 0.26), (0.81, 0.49))
    arrow(ax, (0.57, 0.355), (0.81, 0.51), color="#879098")

    rounded_box(
        ax,
        (0.62, 0.82),
        0.17,
        0.11,
        "Frozen linear probe\n(family labels)",
        "#F5F5F3",
        edge="#8A8F93",
        fontsize=8.5,
    )
    arrow(ax, (0.685, 0.72), (0.685, 0.82), color="#8A8F93", dashed=True)
    ax.text(
        0.805,
        0.875,
        "evaluation only",
        fontsize=8,
        color="#6B7277",
        va="center",
    )

    ax.text(
        0.50,
        0.05,
        "Training objective: predict an unobserved intervention response. "
        "Mechanism-family labels never train the encoder-decoder.",
        ha="center",
        fontsize=9.5,
        color=INK,
    )
    ax.set_title("Predictive mechanism encoder-decoder", fontsize=15, pad=8)
    save(fig, "encoder_decoder_architecture")


# -----------------------------------------------------------------------------
# Relational measurement graph and command-line entry point
# -----------------------------------------------------------------------------

def create_relational_graph():
    fig, ax = plt.subplots(figsize=(9.6, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "Base": ((0.15, 0.48), "#60676C"),
        "Primary": ((0.50, 0.72), "#2A7F8E"),
        "Secondary": ((0.82, 0.34), "#D9902F"),
    }
    for label, (position, color) in nodes.items():
        circle = patches.Circle(position, 0.078, facecolor=color, edgecolor=INK, lw=1.4)
        ax.add_patch(circle)
        ax.text(*position, label, color="white", ha="center", va="center", fontweight="bold")

    arrow(ax, (0.225, 0.51), (0.423, 0.67), style="<->")
    ax.text(0.27, 0.66, "relative pose\nvelocity\nacceleration", ha="center", fontsize=8.8)

    arrow(ax, (0.218, 0.45), (0.745, 0.35), style="<->")
    ax.text(0.48, 0.32, "local anchors and geometry", ha="center", fontsize=8.8)

    arrow(ax, (0.565, 0.67), (0.763, 0.40), style="<->")
    ax.text(0.72, 0.58, "pair distance\nrelative angle\nfollower motion", ha="center", fontsize=8.8)

    ax.annotate(
        "force / torque probe",
        xy=(0.50, 0.81),
        xytext=(0.50, 0.96),
        ha="center",
        fontsize=9.5,
        color=INPUT,
        arrowprops={"arrowstyle": "-|>", "color": INPUT, "lw": 1.8},
    )
    ax.text(
        0.50,
        0.12,
        "Node attributes: scale-normalised dimensions, mass ratios, inertia ratios and shape code",
        ha="center",
        fontsize=9,
        color=INK,
    )
    ax.text(
        0.50,
        0.06,
        "Edge attributes evolve over time; constraint reactions, energy, power and work are also recorded.",
        ha="center",
        fontsize=8.8,
        color="#555D62",
    )
    ax.set_title("Relational mechanical representation", fontsize=15, pad=8)
    save(fig, "relational_representation_graph")


def main():
    style()
    stored = np.load(DATA_DIR / "active_constraint_space_v2.npz")
    metadata = load_metadata()
    raw_names = stored["raw_feature_names"].tolist()
    raw_index = {name: raw_names.index(name) for name in raw_names}

    create_family_storyboard(stored, metadata, raw_index)
    create_encoder_decoder_diagram()
    create_relational_graph()
    print(f"Created report figures in {REPORT_FIGURE_DIR}")


if __name__ == "__main__":
    main()
