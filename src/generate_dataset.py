#!/usr/bin/env python3
"""Generate paired passive and active trials for five constraint families."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from config import (
    ABSOLUTE_GEOMETRY_FEATURES,
    ASSEMBLIES_PER_FAMILY,
    DATA_DIR,
    DT,
    FAMILIES,
    FAMILY_LABELS,
    FRAMES,
    MASTER_SEED,
    PROTOCOLS,
    PROTOCOL_LABELS,
    RAW_FRAME_FEATURES,
    RELATIONAL_FRAME_FEATURES,
    RELATIONAL_GEOMETRY_FEATURES,
    TRAIN_ASSEMBLIES,
    VALIDATION_ASSEMBLIES,
)
from simulator import sample_config, simulate


# -----------------------------------------------------------------------------
# File integrity helper
# -----------------------------------------------------------------------------

def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# Dataset generation and serialization
# -----------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    assembly_count = len(FAMILIES) * ASSEMBLIES_PER_FAMILY
    trajectory_count = assembly_count * len(PROTOCOLS)

    raw = np.empty((trajectory_count, FRAMES, len(RAW_FRAME_FEATURES)), dtype=np.float32)
    relational = np.empty(
        (trajectory_count, FRAMES, len(RELATIONAL_FRAME_FEATURES)), dtype=np.float32
    )
    absolute_geometry = np.empty(
        (trajectory_count, len(ABSOLUTE_GEOMETRY_FEATURES)), dtype=np.float32
    )
    relational_geometry = np.empty(
        (trajectory_count, len(RELATIONAL_GEOMETRY_FEATURES)), dtype=np.float32
    )
    poses = np.empty((trajectory_count, FRAMES, 3, 7), dtype=np.float32)
    family_labels = np.empty(trajectory_count, dtype=np.int8)
    protocol_labels = np.empty(trajectory_count, dtype=np.int8)
    assembly_ids = np.empty(trajectory_count, dtype=np.int16)
    split_codes = np.empty(trajectory_count, dtype=np.int8)
    metadata = []
    split_to_code = {"train": 0, "validation": 1, "ood_test": 2}

    started = time.perf_counter()
    trajectory_index = 0
    for family in FAMILIES:
        family_index = FAMILY_LABELS[family]
        for assembly_index in range(ASSEMBLIES_PER_FAMILY):
            global_assembly_id = family_index * ASSEMBLIES_PER_FAMILY + assembly_index
            for protocol in PROTOCOLS:
                config = sample_config(family, assembly_index, protocol)
                result = simulate(config)
                raw[trajectory_index] = result["raw"]
                relational[trajectory_index] = result["relational"]
                absolute_geometry[trajectory_index] = result["absolute_geometry"]
                relational_geometry[trajectory_index] = result["relational_geometry"]
                poses[trajectory_index] = result["poses"]
                family_labels[trajectory_index] = family_index
                protocol_labels[trajectory_index] = PROTOCOL_LABELS[protocol]
                assembly_ids[trajectory_index] = global_assembly_id
                split_codes[trajectory_index] = split_to_code[config.split]
                metadata.append(
                    {
                        "trajectory_id": f"acs2-{family}-{assembly_index:03d}-{protocol}",
                        "trajectory_index": trajectory_index,
                        "global_assembly_id": global_assembly_id,
                        "family_label": family_index,
                        "protocol_label": PROTOCOL_LABELS[protocol],
                        **result["metadata"],
                    }
                )
                trajectory_index += 1
                if trajectory_index % 100 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"Generated {trajectory_index}/{trajectory_count} trajectories in {elapsed:.1f}s")

    dataset_path = DATA_DIR / "active_constraint_space_v2.npz"
    np.savez_compressed(
        dataset_path,
        raw=raw,
        relational=relational,
        absolute_geometry=absolute_geometry,
        relational_geometry=relational_geometry,
        poses=poses,
        family_labels=family_labels,
        protocol_labels=protocol_labels,
        assembly_ids=assembly_ids,
        split_codes=split_codes,
        raw_feature_names=np.array(RAW_FRAME_FEATURES),
        relational_feature_names=np.array(RELATIONAL_FRAME_FEATURES),
        absolute_geometry_feature_names=np.array(ABSOLUTE_GEOMETRY_FEATURES),
        relational_geometry_feature_names=np.array(RELATIONAL_GEOMETRY_FEATURES),
        family_names=np.array(FAMILIES),
        protocol_names=np.array(PROTOCOLS),
    )

    metadata_path = DATA_DIR / "trajectory_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in metadata:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    manifest = {
        "dataset": "Active Constraint Space v2",
        "version": "acs-2.0",
        "description": (
            "Paired passive and active trials for naturalized 2D kinematic assemblies, "
            "with absolute and relational state, action, reaction, energy, and geometry measurements."
        ),
        "seed": MASTER_SEED,
        "families": list(FAMILIES),
        "protocols": list(PROTOCOLS),
        "assemblies_per_family": ASSEMBLIES_PER_FAMILY,
        "trials_per_assembly": len(PROTOCOLS),
        "assemblies": assembly_count,
        "trajectories": trajectory_count,
        "frames_per_trajectory": FRAMES,
        "total_frames": trajectory_count * FRAMES,
        "dt_seconds": DT,
        "splits_per_family": {
            "train": TRAIN_ASSEMBLIES,
            "validation": VALIDATION_ASSEMBLIES,
            "ood_test": ASSEMBLIES_PER_FAMILY - TRAIN_ASSEMBLIES - VALIDATION_ASSEMBLIES,
        },
        "ood_regimes": ["pose", "scale", "mass", "shape", "combined"],
        "measurements": {
            "raw_frame_features": len(RAW_FRAME_FEATURES),
            "relational_frame_features": len(RELATIONAL_FRAME_FEATURES),
            "absolute_geometry_features": len(ABSOLUTE_GEOMETRY_FEATURES),
            "relational_geometry_features": len(RELATIONAL_GEOMETRY_FEATURES),
        },
        "experimental_unit": (
            "One assembly is cloned into passive, push-x, push-y, clockwise-torque, "
            "and counter-clockwise-torque trials from an identical initial state."
        ),
        "label_exclusion": (
            "Family labels, constraint identifiers, allowed-motion flags, and blocked-motion flags "
            "are absent from all model input arrays."
        ),
        "files": {},
    }
    for path in (dataset_path, metadata_path):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    print(
        f"Completed {assembly_count} assemblies, {trajectory_count} trajectories, "
        f"and {trajectory_count * FRAMES} frames in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
