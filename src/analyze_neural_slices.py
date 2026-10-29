#!/usr/bin/env python3
"""Break neural prediction and latent identification down by OOD regime."""

from __future__ import annotations

import csv

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACT_DIR, MASTER_SEED, MODEL_DIR, PROTOCOLS
from train_encoder_decoder import (
    ACTIVE_PROTOCOLS,
    MechanismEncoderDecoder,
    SEEDS,
    as_tensors,
    extract_latents,
    prepare_data,
    time_features,
)


OOD_REGIMES = ("pose", "scale", "mass", "shape", "combined")


# -----------------------------------------------------------------------------
# Checkpoint and CSV helpers
# -----------------------------------------------------------------------------

def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(representation, seed, data):
    checkpoint = torch.load(
        MODEL_DIR / f"{representation}_encoder_decoder_seed{seed}.pt",
        map_location="cpu",
        weights_only=False,
    )
    tensors = as_tensors(data)
    model = MechanismEncoderDecoder(
        sequence_dim=tensors["sequence"].shape[-1],
        geometry_dim=tensors["geometry"].shape[-1],
        action_dim=tensors["actions"].shape[-1],
        response_dim=tensors["responses"].shape[-1],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# -----------------------------------------------------------------------------
# OOD-regime latent and response evaluation
# -----------------------------------------------------------------------------

def main():
    torch.set_num_threads(min(6, torch.get_num_threads()))
    rows = []
    for representation in ("absolute", "relational"):
        data, _ = prepare_data(representation)
        tensors = as_tensors(data)
        times = time_features(tensors["sequence"].shape[2])
        development = data["splits"] != 2
        test = data["splits"] == 2
        for seed in SEEDS:
            model = load_model(representation, seed, data)
            all_latent = extract_latents(model, data, ACTIVE_PROTOCOLS)
            classifier = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("classifier", LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")),
                ]
            )
            classifier.fit(all_latent[development], data["labels"][development])
            latent_predictions = classifier.predict(all_latent)

            for regime_index, regime in enumerate(OOD_REGIMES):
                lower = 60 + regime_index * 4
                upper = lower + 4
                mask_numpy = test & (data["local_indices"] >= lower) & (data["local_indices"] < upper)
                assembly = torch.from_numpy(np.flatnonzero(mask_numpy))
                prediction_errors = []
                with torch.no_grad():
                    for target in ACTIVE_PROTOCOLS:
                        context_mask = torch.ones((len(assembly), len(PROTOCOLS)), dtype=torch.bool)
                        context_mask[:, target] = False
                        target_tensor = torch.full((len(assembly),), target, dtype=torch.long)
                        prediction, _ = model(
                            tensors["sequence"][assembly],
                            context_mask,
                            tensors["geometry"][assembly],
                            tensors["actions"][assembly, target],
                            target_tensor,
                            times,
                        )
                        target_values = tensors["responses"][assembly, target]
                        prediction_errors.append((prediction.numpy() - target_values.numpy()).reshape(-1))
                error = np.concatenate(prediction_errors)
                rows.append(
                    {
                        "representation": representation,
                        "seed": seed,
                        "ood_regime": regime,
                        "assemblies": int(mask_numpy.sum()),
                        "held_out_probe_normalized_rmse": float(np.sqrt(np.mean(error**2))),
                        "latent_family_accuracy": accuracy_score(
                            data["labels"][mask_numpy], latent_predictions[mask_numpy]
                        ),
                    }
                )
    write_csv(ARTIFACT_DIR / "neural_ood_slices.csv", rows)
    for representation in ("absolute", "relational"):
        print(representation)
        for regime in OOD_REGIMES:
            selected = [
                row
                for row in rows
                if row["representation"] == representation and row["ood_regime"] == regime
            ]
            print(
                f"  {regime}: RMSE={np.mean([row['held_out_probe_normalized_rmse'] for row in selected]):.3f}, "
                f"accuracy={np.mean([row['latent_family_accuracy'] for row in selected]):.3f}"
            )


if __name__ == "__main__":
    main()
