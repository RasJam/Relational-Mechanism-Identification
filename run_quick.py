#!/usr/bin/env python3
"""Run a compact end-to-end verification of the submitted implementation.

The quick run creates new trajectories in memory, evaluates passive and active
classification, and performs a short encoder-decoder optimization. It verifies
the complete software path without reproducing the full thesis experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from build_features import summarize_trials  # noqa: E402
from config import (  # noqa: E402
    FAMILIES,
    FRAMES,
    MASTER_SEED,
    PROTOCOLS,
    RAW_FRAME_FEATURES,
    RELATIONAL_ACTION_FEATURES,
    RELATIONAL_FRAME_FEATURES,
    RELATIONAL_GEOMETRY_FEATURES,
    RESPONSE_FEATURES,
)
from simulator import sample_config, simulate  # noqa: E402
from train_encoder_decoder import MechanismEncoderDecoder, time_features  # noqa: E402


ASSEMBLIES_PER_FAMILY = 12
TRAIN_ASSEMBLIES_PER_FAMILY = 8
FRAME_STRIDE = 3
NEURAL_STEPS = 20


def standardize(values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Standardize an array while protecting constant channels from division by zero."""
    mean = values.mean(axis=axes, keepdims=True)
    standard_deviation = values.std(axis=axes, keepdims=True)
    standard_deviation = np.where(standard_deviation < 1.0e-6, 1.0, standard_deviation)
    return ((values - mean) / standard_deviation).astype(np.float32)


def classifier() -> Pipeline:
    """Create the same standardized logistic model used by the main baseline."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.5,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=MASTER_SEED,
                ),
            ),
        ]
    )


def generate_quick_dataset() -> dict[str, np.ndarray]:
    """Simulate a small balanced set of matched assemblies for all five families."""
    assembly_count = len(FAMILIES) * ASSEMBLIES_PER_FAMILY
    protocol_count = len(PROTOCOLS)
    raw = np.empty(
        (assembly_count, protocol_count, FRAMES, len(RAW_FRAME_FEATURES)),
        dtype=np.float32,
    )
    relational = np.empty(
        (assembly_count, protocol_count, FRAMES, len(RELATIONAL_FRAME_FEATURES)),
        dtype=np.float32,
    )
    geometry = np.empty(
        (assembly_count, len(RELATIONAL_GEOMETRY_FEATURES)),
        dtype=np.float32,
    )
    labels = np.empty(assembly_count, dtype=np.int64)
    local_indices = np.empty(assembly_count, dtype=np.int64)

    row = 0
    for family_index, family in enumerate(FAMILIES):
        for assembly_index in range(ASSEMBLIES_PER_FAMILY):
            for protocol_index, protocol in enumerate(PROTOCOLS):
                output = simulate(sample_config(family, assembly_index, protocol))
                raw[row, protocol_index] = output["raw"]
                relational[row, protocol_index] = output["relational"]
                if protocol_index == 0:
                    geometry[row] = output["relational_geometry"]
            labels[row] = family_index
            local_indices[row] = assembly_index
            row += 1

    return {
        "raw": raw,
        "relational": relational,
        "geometry": geometry,
        "labels": labels,
        "local_indices": local_indices,
    }


def evaluate_classical(dataset: dict[str, np.ndarray]) -> dict[str, float]:
    """Compare passive-only and all-probe relational classification."""
    relational_summary, _ = summarize_trials(
        dataset["relational"],
        RELATIONAL_FRAME_FEATURES,
    )
    labels = dataset["labels"]
    training = dataset["local_indices"] < TRAIN_ASSEMBLIES_PER_FAMILY
    testing = ~training

    passive = np.concatenate(
        [
            dataset["geometry"],
            relational_summary[:, 0].reshape(len(labels), -1),
        ],
        axis=1,
    )
    all_probes = np.concatenate(
        [
            dataset["geometry"],
            relational_summary.reshape(len(labels), -1),
        ],
        axis=1,
    )

    results = {}
    for name, matrix in (("passive_accuracy", passive), ("all_probe_accuracy", all_probes)):
        model = classifier()
        model.fit(matrix[training], labels[training])
        predictions = model.predict(matrix[testing])
        results[name] = float(accuracy_score(labels[testing], predictions))
    return results


def evaluate_neural_step(dataset: dict[str, np.ndarray]) -> dict[str, object]:
    """Train the submitted encoder-decoder briefly and verify its tensor interface."""
    torch.manual_seed(MASTER_SEED)
    relational_names = list(RELATIONAL_FRAME_FEATURES)
    response_indices = [relational_names.index(name) for name in RESPONSE_FEATURES]
    action_indices = [
        relational_names.index(name) for name in RELATIONAL_ACTION_FEATURES
    ]
    frames = np.arange(0, FRAMES, FRAME_STRIDE)

    sequence = dataset["relational"][:, :, frames]
    geometry = dataset["geometry"]
    actions = sequence[:, :, :, action_indices]
    responses = sequence[:, :, :, response_indices]

    sequence = standardize(sequence, (0, 1, 2))
    geometry = standardize(geometry, (0,))
    actions = standardize(actions, (0, 1, 2))
    responses = standardize(responses, (0, 1, 2))

    batch_size = 16
    context = torch.from_numpy(sequence[:batch_size])
    geometry_tensor = torch.from_numpy(geometry[:batch_size])
    query_action = torch.from_numpy(actions[:batch_size, 1])
    target_response = torch.from_numpy(responses[:batch_size, 1])
    query_protocol = torch.full((batch_size,), 1, dtype=torch.long)
    context_mask = torch.zeros((batch_size, len(PROTOCOLS)), dtype=torch.bool)
    context_mask[:, 0] = True
    context_mask[:, 3] = True
    times = time_features(len(frames))

    model = MechanismEncoderDecoder(
        sequence_dim=context.shape[-1],
        geometry_dim=geometry_tensor.shape[-1],
        action_dim=query_action.shape[-1],
        response_dim=target_response.shape[-1],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
    loss_function = torch.nn.MSELoss()

    losses = []
    latent = None
    for _ in range(NEURAL_STEPS):
        optimizer.zero_grad()
        prediction, latent = model(
            context,
            context_mask,
            geometry_tensor,
            query_action,
            query_protocol,
            times,
        )
        loss = loss_function(prediction, target_response)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "prediction_shape": list(prediction.shape),
        "latent_shape": list(latent.shape),
    }


def main() -> None:
    """Execute the compact simulation, baseline, and neural verification."""
    print("Generating compact matched-mechanism dataset...")
    dataset = generate_quick_dataset()
    classical = evaluate_classical(dataset)
    neural = evaluate_neural_step(dataset)

    result = {
        "status": "passed",
        "families": list(FAMILIES),
        "assemblies": int(len(dataset["labels"])),
        "trajectories": int(len(dataset["labels"]) * len(PROTOCOLS)),
        "frames": int(len(dataset["labels"]) * len(PROTOCOLS) * FRAMES),
        "classical": classical,
        "neural_smoke_test": neural,
    }
    output_directory = PROJECT_ROOT / "quick_output"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "quick_results.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Quick verification written to {output_path}")


if __name__ == "__main__":
    main()

