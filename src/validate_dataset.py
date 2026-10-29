#!/usr/bin/env python3
"""Validate dataset integrity and basic physical behavior."""

from __future__ import annotations

import json

import numpy as np

from config import (
    ABSOLUTE_GEOMETRY_FEATURES,
    ARTIFACT_DIR,
    ASSEMBLIES_PER_FAMILY,
    DATA_DIR,
    FAMILIES,
    FRAMES,
    PROTOCOLS,
    RAW_FRAME_FEATURES,
    RELATIONAL_FRAME_FEATURES,
    RELATIONAL_GEOMETRY_FEATURES,
)


# -----------------------------------------------------------------------------
# Validation helper
# -----------------------------------------------------------------------------

def require(condition, message):
    if not condition:
        raise AssertionError(message)


# -----------------------------------------------------------------------------
# Structural, leakage, and physical-behaviour checks
# -----------------------------------------------------------------------------

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stored = np.load(DATA_DIR / "active_constraint_space_v2.npz")
    trajectories = len(FAMILIES) * ASSEMBLIES_PER_FAMILY * len(PROTOCOLS)
    expected = {
        "raw": (trajectories, FRAMES, len(RAW_FRAME_FEATURES)),
        "relational": (trajectories, FRAMES, len(RELATIONAL_FRAME_FEATURES)),
        "absolute_geometry": (trajectories, len(ABSOLUTE_GEOMETRY_FEATURES)),
        "relational_geometry": (trajectories, len(RELATIONAL_GEOMETRY_FEATURES)),
        "poses": (trajectories, FRAMES, 3, 7),
    }
    for key, shape in expected.items():
        require(stored[key].shape == shape, f"{key}: expected {shape}, got {stored[key].shape}")
        require(np.isfinite(stored[key]).all(), f"{key} contains non-finite values")

    family_labels = stored["family_labels"]
    protocol_labels = stored["protocol_labels"]
    assembly_ids = stored["assembly_ids"]
    split_codes = stored["split_codes"]
    require(set(np.unique(split_codes)) == {0, 1, 2}, "missing split")
    require(
        np.all(np.bincount(family_labels, minlength=len(FAMILIES)) == ASSEMBLIES_PER_FAMILY * len(PROTOCOLS)),
        "family imbalance",
    )
    require(
        np.all(np.bincount(protocol_labels, minlength=len(PROTOCOLS)) == len(FAMILIES) * ASSEMBLIES_PER_FAMILY),
        "protocol imbalance",
    )

    # Every assembly must have five trials, identical static geometry, and one split.
    for assembly_id in np.unique(assembly_ids):
        indices = np.flatnonzero(assembly_ids == assembly_id)
        require(len(indices) == len(PROTOCOLS), f"assembly {assembly_id} trial count")
        require(set(protocol_labels[indices]) == set(range(len(PROTOCOLS))), f"assembly {assembly_id} protocols")
        require(len(set(split_codes[indices].tolist())) == 1, f"assembly {assembly_id} split leakage")
        require(
            np.allclose(stored["absolute_geometry"][indices], stored["absolute_geometry"][indices[0]]),
            f"assembly {assembly_id} absolute geometry differs across trials",
        )
        require(
            np.allclose(stored["relational_geometry"][indices], stored["relational_geometry"][indices[0]]),
            f"assembly {assembly_id} relational geometry differs across trials",
        )

    raw_names = stored["raw_feature_names"].tolist()
    relational_names = stored["relational_feature_names"].tolist()
    prohibited = ("family", "constraint", "allowed", "blocked", "mechanism")
    require(
        not any(word in name for word in prohibited for name in raw_names + relational_names),
        "a label-like answer feature leaked into model inputs",
    )
    raw_index = {name: index for index, name in enumerate(raw_names)}
    action_indices = [raw_index[name] for name in ("input_force_x", "input_force_y", "input_torque")]
    passive = protocol_labels == 0
    active = protocol_labels != 0
    require(np.max(np.abs(stored["raw"][passive][:, :, action_indices])) == 0.0, "passive trial has input")
    require(np.min(np.max(np.abs(stored["raw"][active][:, :, action_indices]), axis=(1, 2))) > 0.0, "active trial lacks input")

    rel_index = {name: index for index, name in enumerate(relational_names)}
    angular_channel = rel_index["primary_relative_angular_velocity"]
    secondary_angular_channel = rel_index["secondary_relative_angular_velocity"]
    position_channels = [rel_index["primary_relative_x"], rel_index["primary_relative_y"]]

    family_protocol_summary = {}
    for family_index, family in enumerate(FAMILIES):
        family_protocol_summary[family] = {}
        for protocol_index, protocol in enumerate(PROTOCOLS):
            mask = (family_labels == family_index) & (protocol_labels == protocol_index)
            sequence = stored["relational"][mask]
            displacement = np.linalg.norm(sequence[:, -1, position_channels] - sequence[:, 0, position_channels], axis=1)
            angular_activity = np.mean(np.abs(sequence[:, :, angular_channel]), axis=1)
            secondary_activity = np.mean(np.abs(sequence[:, :, secondary_angular_channel]), axis=1)
            family_protocol_summary[family][protocol] = {
                "median_primary_displacement": float(np.median(displacement)),
                "median_primary_angular_speed": float(np.median(angular_activity)),
                "median_secondary_angular_speed": float(np.median(secondary_activity)),
            }

    require(
        family_protocol_summary["fixed_coupling"]["torque_ccw"]["median_primary_angular_speed"] < 0.03,
        "fixed coupling rotates too freely",
    )
    require(
        family_protocol_summary["revolute_hinge"]["torque_ccw"]["median_primary_angular_speed"] > 0.10,
        "hinge does not rotate",
    )
    require(
        family_protocol_summary["prismatic_guide"]["push_x"]["median_primary_displacement"]
        > family_protocol_summary["prismatic_guide"]["push_y"]["median_primary_displacement"] + 0.05,
        "slider does not prefer its guide axis",
    )
    ratchet_ccw = family_protocol_summary["ratchet_wheel"]["torque_ccw"]["median_primary_angular_speed"]
    ratchet_cw = family_protocol_summary["ratchet_wheel"]["torque_cw"]["median_primary_angular_speed"]
    require(abs(ratchet_ccw - ratchet_cw) > 0.10, "ratchet lacks directional asymmetry")
    require(
        family_protocol_summary["gear_pair"]["torque_ccw"]["median_secondary_angular_speed"] > 0.10,
        "gear follower does not rotate",
    )

    report = {
        "status": "passed",
        "trajectories": trajectories,
        "assemblies": len(np.unique(assembly_ids)),
        "total_frames": trajectories * FRAMES,
        "shapes": {key: list(value) for key, value in expected.items()},
        "family_protocol_summary": family_protocol_summary,
    }
    (ARTIFACT_DIR / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
