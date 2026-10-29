#!/usr/bin/env python3
"""Evaluate absolute and relational classical baselines under active probing."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACT_DIR, FAMILIES, MASTER_SEED, MODEL_DIR, PROTOCOLS


ACTIVE_PROTOCOLS = tuple(range(1, len(PROTOCOLS)))
OOD_REGIMES = ("pose", "scale", "mass", "shape", "combined")


# -----------------------------------------------------------------------------
# Probe subsets and representation matrices
# -----------------------------------------------------------------------------

def protocol_sets():
    sets = []
    for count in range(len(ACTIVE_PROTOCOLS) + 1):
        for active in itertools.combinations(ACTIVE_PROTOCOLS, count):
            sets.append((0,) + active)
    return sets


def matrix_for(stored, representation, selected_protocols):
    if representation == "geometry_only":
        return stored["relational_geometry"]
    if representation == "absolute":
        sequence = stored["raw_summary"][:, selected_protocols].reshape(len(stored["labels"]), -1)
        return np.concatenate([stored["absolute_geometry"], sequence], axis=1)
    sequence = stored["relational_summary"][:, selected_protocols].reshape(len(stored["labels"]), -1)
    if representation == "relational_mechanics":
        return sequence
    if representation == "relational_combined":
        return np.concatenate([stored["relational_geometry"], sequence], axis=1)
    raise ValueError(representation)


# -----------------------------------------------------------------------------
# Metrics and model definitions
# -----------------------------------------------------------------------------

def exact_interval(labels, predictions):
    result = binomtest(int(np.sum(labels == predictions)), len(labels))
    interval = result.proportion_ci(confidence_level=0.95, method="exact")
    return float(interval.low), float(interval.high)


def metric_row(representation, model_name, protocols, split, labels, predictions, seed):
    low, high = exact_interval(labels, predictions)
    return {
        "representation": representation,
        "model": model_name,
        "seed": seed,
        "protocols": "+".join(PROTOCOLS[index] for index in protocols),
        "active_probe_count": len(protocols) - 1,
        "split": split,
        "samples": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "accuracy_ci_low": low,
        "accuracy_ci_high": high,
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_models(seed):
    return {
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.5,
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


# -----------------------------------------------------------------------------
# Training, OOD evaluation, and artifact export
# -----------------------------------------------------------------------------

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stored = np.load(ARTIFACT_DIR / "assembly_features.npz")
    labels = stored["labels"]
    splits = stored["split_codes"]
    local_indices = stored["local_assembly_indices"]
    train_mask = splits == 0
    validation_mask = splits == 1
    development_mask = splits != 2
    test_mask = splits == 2

    rows = []
    predictions_rows = []
    confusions = {}
    selected_sets = protocol_sets()
    representations = ("absolute", "relational_mechanics", "relational_combined")

    # One complete factorial run over every active-probe subset.
    for representation in representations:
        for protocols in selected_sets:
            matrix = matrix_for(stored, representation, protocols)
            for model_name, model in make_models(MASTER_SEED).items():
                model.fit(matrix[train_mask], labels[train_mask])
                validation_predictions = model.predict(matrix[validation_mask])
                rows.append(
                    metric_row(
                        representation,
                        model_name,
                        protocols,
                        "validation",
                        labels[validation_mask],
                        validation_predictions,
                        MASTER_SEED,
                    )
                )
                model.fit(matrix[development_mask], labels[development_mask])
                test_predictions = model.predict(matrix[test_mask])
                rows.append(
                    metric_row(
                        representation,
                        model_name,
                        protocols,
                        "ood_test",
                        labels[test_mask],
                        test_predictions,
                        MASTER_SEED,
                    )
                )
                if len(protocols) in {1, len(PROTOCOLS)}:
                    key = f"{representation}__{model_name}__{len(protocols)-1}_probes"
                    confusions[key] = confusion_matrix(
                        labels[test_mask], test_predictions, labels=np.arange(len(FAMILIES))
                    )
                    for local, assembly_row in enumerate(np.flatnonzero(test_mask)):
                        regime = OOD_REGIMES[(int(local_indices[assembly_row]) - 60) // 4]
                        predictions_rows.append(
                            {
                                "assembly_row": int(assembly_row),
                                "family": FAMILIES[int(labels[assembly_row])],
                                "representation": representation,
                                "model": model_name,
                                "protocols": "+".join(PROTOCOLS[index] for index in protocols),
                                "ood_regime": regime,
                                "prediction": FAMILIES[int(test_predictions[local])],
                                "correct": int(test_predictions[local] == labels[assembly_row]),
                            }
                        )
                    joblib.dump(model, MODEL_DIR / f"baseline-{key}.joblib")

    # Geometry-only control: no trajectory or action information.
    geometry_matrix = matrix_for(stored, "geometry_only", (0,))
    for model_name, model in make_models(MASTER_SEED).items():
        model.fit(geometry_matrix[train_mask], labels[train_mask])
        for split_name, mask in (("validation", validation_mask), ("ood_test", test_mask)):
            if split_name == "ood_test":
                model.fit(geometry_matrix[development_mask], labels[development_mask])
            predictions = model.predict(geometry_matrix[mask])
            rows.append(
                metric_row(
                    "geometry_only",
                    model_name,
                    (0,),
                    split_name,
                    labels[mask],
                    predictions,
                    MASTER_SEED,
                )
            )

    # Repeat the two headline conditions to expose model-seed variation.
    repeated_rows = []
    for seed_offset in range(5):
        seed = MASTER_SEED + seed_offset
        for representation in representations:
            for protocols in ((0,), tuple(range(len(PROTOCOLS)))):
                matrix = matrix_for(stored, representation, protocols)
                for model_name, model in make_models(seed).items():
                    model.fit(matrix[development_mask], labels[development_mask])
                    predictions = model.predict(matrix[test_mask])
                    repeated_rows.append(
                        metric_row(
                            representation,
                            model_name,
                            protocols,
                            "ood_test",
                            labels[test_mask],
                            predictions,
                            seed,
                        )
                    )

    write_csv(ARTIFACT_DIR / "baseline_metrics.csv", rows)
    write_csv(ARTIFACT_DIR / "baseline_seed_repeats.csv", repeated_rows)
    write_csv(ARTIFACT_DIR / "baseline_predictions.csv", predictions_rows)
    np.savez_compressed(ARTIFACT_DIR / "baseline_confusions.npz", **confusions)

    headline = {}
    for representation in representations:
        for probe_count in (0, 4):
            matching = [
                row
                for row in repeated_rows
                if row["representation"] == representation
                and row["model"] == "random_forest"
                and row["active_probe_count"] == probe_count
            ]
            values = np.array([row["accuracy"] for row in matching])
            headline[f"{representation}_{probe_count}_probes"] = {
                "mean_accuracy": float(values.mean()),
                "std_accuracy": float(values.std(ddof=1)),
                "seeds": len(values),
            }
    (ARTIFACT_DIR / "baseline_headline.json").write_text(
        json.dumps(headline, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
