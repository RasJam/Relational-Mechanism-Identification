#!/usr/bin/env python3
"""Aggregate baseline and neural results into report-ready summaries."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from config import ARTIFACT_DIR


# -----------------------------------------------------------------------------
# Artifact loading
# -----------------------------------------------------------------------------

def read_csv(name):
    with (ARTIFACT_DIR / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# -----------------------------------------------------------------------------
# Statistical aggregation and report-ready summaries
# -----------------------------------------------------------------------------

def main():
    baseline_metrics = read_csv("baseline_metrics.csv")
    baseline_predictions = read_csv("baseline_predictions.csv")
    neural_prediction = read_csv("neural_prediction_metrics.csv")
    neural_latent = read_csv("neural_latent_metrics.csv")
    neural_distances = read_csv("neural_latent_distances.csv")
    neural_slices = read_csv("neural_ood_slices.csv")

    active_curve = []
    for representation in ("absolute", "relational_mechanics", "relational_combined"):
        for model in ("logistic", "random_forest"):
            for budget in range(5):
                selected = [
                    float(row["accuracy"])
                    for row in baseline_metrics
                    if row["split"] == "ood_test"
                    and row["representation"] == representation
                    and row["model"] == model
                    and int(row["active_probe_count"]) == budget
                ]
                active_curve.append(
                    {
                        "representation": representation,
                        "model": model,
                        "active_probe_count": budget,
                        "mean_accuracy_across_probe_subsets": float(np.mean(selected)),
                        "minimum_accuracy": float(np.min(selected)),
                        "maximum_accuracy": float(np.max(selected)),
                        "probe_subsets": len(selected),
                    }
                )

    paired_tests = []
    grouped_predictions = defaultdict(dict)
    for row in baseline_predictions:
        key = (row["assembly_row"], row["representation"], row["model"])
        probe_count = 0 if row["protocols"] == "passive" else 4
        grouped_predictions[key][probe_count] = int(row["correct"])
    for representation in ("absolute", "relational_mechanics", "relational_combined"):
        for model in ("logistic", "random_forest"):
            pairs = [
                values
                for (assembly, rep, model_name), values in grouped_predictions.items()
                if rep == representation and model_name == model and set(values) == {0, 4}
            ]
            passive_only = sum(values[0] == 1 and values[4] == 0 for values in pairs)
            active_only = sum(values[0] == 0 and values[4] == 1 for values in pairs)
            discordant = passive_only + active_only
            p_value = (
                float(binomtest(min(passive_only, active_only), discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            paired_tests.append(
                {
                    "representation": representation,
                    "model": model,
                    "passive_correct_active_wrong": passive_only,
                    "passive_wrong_active_correct": active_only,
                    "discordant_pairs": discordant,
                    "exact_mcnemar_p": p_value,
                }
            )

    neural_summary = {}
    for representation in ("absolute", "relational"):
        neural_summary[representation] = {}
        for budget in range(5):
            selected = [
                float(row["accuracy"])
                for row in neural_latent
                if row["representation"] == representation
                and int(row["context_active_probes"]) == budget
            ]
            neural_summary[representation][f"latent_accuracy_{budget}_probes"] = {
                "mean": float(np.mean(selected)),
                "std": float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0,
                "runs": len(selected),
            }
        selected_prediction = [
            float(row["normalized_rmse"])
            for row in neural_prediction
            if row["representation"] == representation
            and row["split"] == "ood_test"
            and int(row["context_active_probes"]) == 3
        ]
        selected_distance = [
            float(row["separation_ratio"])
            for row in neural_distances
            if row["representation"] == representation
        ]
        neural_summary[representation]["held_out_probe_rmse"] = {
            "mean": float(np.mean(selected_prediction)),
            "std": float(np.std(selected_prediction, ddof=1)),
        }
        neural_summary[representation]["latent_separation_ratio"] = {
            "mean": float(np.mean(selected_distance)),
            "std": float(np.std(selected_distance, ddof=1)),
        }

    ood_summary = defaultdict(dict)
    for representation in ("absolute", "relational"):
        for regime in ("pose", "scale", "mass", "shape", "combined"):
            selected = [
                row
                for row in neural_slices
                if row["representation"] == representation and row["ood_regime"] == regime
            ]
            ood_summary[representation][regime] = {
                "prediction_rmse_mean": float(
                    np.mean([float(row["held_out_probe_normalized_rmse"]) for row in selected])
                ),
                "latent_accuracy_mean": float(
                    np.mean([float(row["latent_family_accuracy"]) for row in selected])
                ),
            }

    output = {
        "classical_active_probe_curve": active_curve,
        "paired_active_probe_tests": paired_tests,
        "neural_summary": neural_summary,
        "neural_ood_slices": dict(ood_summary),
        "interpretation_guardrails": [
            "The five classes are controlled 2D kinematic constraint families, not universal mechanisms.",
            "Passive trials are intentionally non-identifiable because matched rigs begin at the same rest state.",
            "Family labels are not used to train the encoder-decoder; they are used only for linear-probe evaluation.",
            "The relational advantage is measured under simulator distribution shifts and is not evidence of real-world transfer.",
        ],
    }
    (ARTIFACT_DIR / "analysis_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(neural_summary, indent=2))


if __name__ == "__main__":
    main()
