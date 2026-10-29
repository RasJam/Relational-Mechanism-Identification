#!/usr/bin/env python3
"""Create assembly-level feature matrices for classical baselines."""

from __future__ import annotations

import json

import numpy as np

from config import ARTIFACT_DIR, DATA_DIR, FRAMES, PROTOCOLS


STAT_NAMES = (
    "mean",
    "std",
    "minimum",
    "maximum",
    "start",
    "end",
    "delta",
    "mean_absolute",
    "total_variation",
)


# -----------------------------------------------------------------------------
# Temporal summary construction
# -----------------------------------------------------------------------------

def summarize_trials(sequence, channel_names):
    # sequence: assemblies x protocols x frames x channels
    statistics = [
        sequence.mean(axis=2),
        sequence.std(axis=2),
        sequence.min(axis=2),
        sequence.max(axis=2),
        sequence[:, :, 0, :],
        sequence[:, :, -1, :],
        sequence[:, :, -1, :] - sequence[:, :, 0, :],
        np.mean(np.abs(sequence), axis=2),
        np.sum(np.abs(np.diff(sequence, axis=2)), axis=2),
    ]
    sample_frames = np.linspace(0, FRAMES - 1, 8, dtype=int)
    sampled = sequence[:, :, sample_frames, :].transpose(0, 1, 3, 2)
    matrix = np.concatenate(statistics + [sampled.reshape(*sampled.shape[:2], -1)], axis=2)
    names = [f"{stat}__{name}" for stat in STAT_NAMES for name in channel_names]
    names.extend(f"t{frame:03d}__{name}" for name in channel_names for frame in sample_frames)
    return matrix.astype(np.float32), names


# -----------------------------------------------------------------------------
# Assembly grouping and feature-file generation
# -----------------------------------------------------------------------------

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stored = np.load(DATA_DIR / "active_constraint_space_v2.npz")
    assembly_ids = stored["assembly_ids"]
    unique_assemblies = np.unique(assembly_ids)
    assembly_count = len(unique_assemblies)
    protocol_count = len(PROTOCOLS)

    # Generation order is family -> assembly -> protocol, but build the grouping
    # explicitly so the feature file remains correct if generation order changes.
    grouped_indices = np.empty((assembly_count, protocol_count), dtype=np.int32)
    labels = np.empty(assembly_count, dtype=np.int8)
    splits = np.empty(assembly_count, dtype=np.int8)
    local_indices = np.empty(assembly_count, dtype=np.int16)
    for row, assembly_id in enumerate(unique_assemblies):
        indices = np.flatnonzero(assembly_ids == assembly_id)
        protocol_labels = stored["protocol_labels"][indices]
        grouped_indices[row, protocol_labels] = indices
        labels[row] = stored["family_labels"][indices[0]]
        splits[row] = stored["split_codes"][indices[0]]
        local_indices[row] = int(assembly_id % 80)

    raw_grouped = stored["raw"][grouped_indices]
    relational_grouped = stored["relational"][grouped_indices]
    raw_summary, raw_summary_names = summarize_trials(
        raw_grouped, stored["raw_feature_names"].tolist()
    )
    relational_summary, relational_summary_names = summarize_trials(
        relational_grouped, stored["relational_feature_names"].tolist()
    )

    feature_path = ARTIFACT_DIR / "assembly_features.npz"
    np.savez_compressed(
        feature_path,
        raw_summary=raw_summary,
        relational_summary=relational_summary,
        absolute_geometry=stored["absolute_geometry"][grouped_indices[:, 0]],
        relational_geometry=stored["relational_geometry"][grouped_indices[:, 0]],
        labels=labels,
        split_codes=splits,
        local_assembly_indices=local_indices,
        global_assembly_ids=unique_assemblies,
    )
    names = {
        "raw_summary": raw_summary_names,
        "relational_summary": relational_summary_names,
        "absolute_geometry": stored["absolute_geometry_feature_names"].tolist(),
        "relational_geometry": stored["relational_geometry_feature_names"].tolist(),
    }
    (ARTIFACT_DIR / "assembly_feature_names.json").write_text(
        json.dumps(names, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Built features for {assembly_count} assemblies: "
        f"raw {raw_summary.shape}, relational {relational_summary.shape}"
    )


if __name__ == "__main__":
    main()
