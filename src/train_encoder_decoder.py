#!/usr/bin/env python3
"""Train predictive absolute and relational mechanism encoder-decoders."""

from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from config import (
    ARTIFACT_DIR,
    DATA_DIR,
    FAMILIES,
    MASTER_SEED,
    MODEL_DIR,
    PROTOCOLS,
    RAW_ACTION_FEATURES,
    RELATIONAL_ACTION_FEATURES,
    RESPONSE_FEATURES,
)


FRAME_STRIDE = 3
LATENT_DIM = 16
HIDDEN_DIM = 48
DECODER_DIM = 96
MAX_EPOCHS = 80
PATIENCE = 12
ACTIVE_PROTOCOLS = tuple(range(1, len(PROTOCOLS)))
SEEDS = (MASTER_SEED, MASTER_SEED + 1, MASTER_SEED + 2)


# -----------------------------------------------------------------------------
# Reproducibility and output helpers
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Data normalization
# -----------------------------------------------------------------------------

@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values, axis):
        mean = values.mean(axis=axis, keepdims=False)
        std = values.std(axis=axis, keepdims=False)
        std = np.where(std < 1.0e-6, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values, clip=None):
        transformed = (values - self.mean) / self.std
        if clip is not None:
            transformed = np.clip(transformed, -clip, clip)
        return transformed.astype(np.float32)

    def inverse(self, values):
        return values * self.std + self.mean


# -----------------------------------------------------------------------------
# Predictive encoder-decoder architecture
# -----------------------------------------------------------------------------

class MechanismEncoderDecoder(nn.Module):
    def __init__(self, sequence_dim, geometry_dim, action_dim, response_dim):
        super().__init__()
        self.protocol_count = len(PROTOCOLS)
        self.trial_encoder = nn.GRU(
            sequence_dim + self.protocol_count,
            HIDDEN_DIM,
            batch_first=True,
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(HIDDEN_DIM + 24, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_DIM),
            nn.Tanh(),
        )
        decoder_input = LATENT_DIM + 24 + action_dim + self.protocol_count + 3
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, DECODER_DIM),
            nn.ReLU(),
            nn.Linear(DECODER_DIM, DECODER_DIM),
            nn.ReLU(),
            nn.Linear(DECODER_DIM, response_dim),
        )

    def encode(self, context, context_mask, geometry):
        batch, protocols, frames, _ = context.shape
        protocol_eye = torch.eye(self.protocol_count, device=context.device)
        protocol_codes = protocol_eye.unsqueeze(0).unsqueeze(2).expand(batch, -1, frames, -1)
        trial_input = torch.cat([context, protocol_codes], dim=-1).reshape(
            batch * protocols, frames, -1
        )
        _, hidden = self.trial_encoder(trial_input)
        trial_codes = hidden[-1].reshape(batch, protocols, HIDDEN_DIM)
        weights = context_mask.to(trial_codes.dtype).unsqueeze(-1)
        pooled = (trial_codes * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        geometry_code = self.geometry_encoder(geometry)
        latent = self.bottleneck(torch.cat([pooled, geometry_code], dim=-1))
        return latent, geometry_code

    def forward(
        self,
        context,
        context_mask,
        geometry,
        query_action,
        query_protocol,
        time_features,
    ):
        latent, geometry_code = self.encode(context, context_mask, geometry)
        frames = query_action.shape[1]
        protocol_code = torch.nn.functional.one_hot(
            query_protocol, num_classes=self.protocol_count
        ).to(query_action.dtype)
        decoder_input = torch.cat(
            [
                latent.unsqueeze(1).expand(-1, frames, -1),
                geometry_code.unsqueeze(1).expand(-1, frames, -1),
                query_action,
                protocol_code.unsqueeze(1).expand(-1, frames, -1),
                time_features.unsqueeze(0).expand(len(context), -1, -1),
            ],
            dim=-1,
        )
        return self.decoder(decoder_input), latent


# -----------------------------------------------------------------------------
# Dataset preparation and context-mask construction
# -----------------------------------------------------------------------------

def prepare_data(representation):
    stored = np.load(DATA_DIR / "active_constraint_space_v2.npz")
    assembly_ids = stored["assembly_ids"]
    unique_ids = np.unique(assembly_ids)
    grouped = np.empty((len(unique_ids), len(PROTOCOLS)), dtype=np.int32)
    labels = np.empty(len(unique_ids), dtype=np.int64)
    splits = np.empty(len(unique_ids), dtype=np.int64)
    local_indices = np.empty(len(unique_ids), dtype=np.int64)
    for row, assembly_id in enumerate(unique_ids):
        indices = np.flatnonzero(assembly_ids == assembly_id)
        grouped[row, stored["protocol_labels"][indices]] = indices
        labels[row] = stored["family_labels"][indices[0]]
        splits[row] = stored["split_codes"][indices[0]]
        local_indices[row] = assembly_id % 80

    frames = np.arange(0, stored["raw"].shape[1], FRAME_STRIDE)
    relational_names = stored["relational_feature_names"].tolist()
    response_indices = [relational_names.index(name) for name in RESPONSE_FEATURES]
    responses = stored["relational"][grouped][:, :, frames][:, :, :, response_indices]

    if representation == "absolute":
        sequence = stored["raw"][grouped][:, :, frames]
        geometry = stored["absolute_geometry"][grouped[:, 0]]
        names = stored["raw_feature_names"].tolist()
        action_indices = [names.index(name) for name in RAW_ACTION_FEATURES]
    elif representation == "relational":
        sequence = stored["relational"][grouped][:, :, frames]
        geometry = stored["relational_geometry"][grouped[:, 0]]
        names = relational_names
        action_indices = [names.index(name) for name in RELATIONAL_ACTION_FEATURES]
    else:
        raise ValueError(representation)
    actions = sequence[:, :, :, action_indices]

    train_mask = splits == 0
    sequence_normalizer = Normalizer.fit(sequence[train_mask], axis=(0, 1, 2))
    geometry_normalizer = Normalizer.fit(geometry[train_mask], axis=0)
    action_normalizer = Normalizer.fit(actions[train_mask], axis=(0, 1, 2))
    response_normalizer = Normalizer.fit(responses[train_mask], axis=(0, 1, 2))
    normalized = {
        "sequence": sequence_normalizer.transform(sequence, clip=12.0),
        "geometry": geometry_normalizer.transform(geometry, clip=12.0),
        "actions": action_normalizer.transform(actions, clip=12.0),
        "responses": response_normalizer.transform(responses),
        "labels": labels,
        "splits": splits,
        "local_indices": local_indices,
        "response_actual": responses,
        "frames": frames,
    }
    normalizers = {
        "sequence": sequence_normalizer,
        "geometry": geometry_normalizer,
        "action": action_normalizer,
        "response": response_normalizer,
    }
    return normalized, normalizers


def as_tensors(data):
    return {
        key: torch.from_numpy(value)
        for key, value in data.items()
        if key in {"sequence", "geometry", "actions", "responses", "labels"}
    }


def time_features(frame_count):
    time = torch.linspace(0.0, 1.0, frame_count)
    return torch.stack([time, torch.sin(math.pi * time), torch.cos(math.pi * time)], dim=1)


def random_context_mask(target_protocols, generator):
    batch = len(target_protocols)
    mask = torch.zeros((batch, len(PROTOCOLS)), dtype=torch.bool)
    mask[:, 0] = True
    for row, target in enumerate(target_protocols.tolist()):
        available = [protocol for protocol in ACTIVE_PROTOCOLS if protocol != target]
        budget = int(torch.randint(0, len(available) + 1, (1,), generator=generator))
        if budget:
            order = torch.randperm(len(available), generator=generator)[:budget]
            for index in order.tolist():
                mask[row, available[index]] = True
    return mask


def fixed_context_mask(batch_size, active_protocols):
    mask = torch.zeros((batch_size, len(PROTOCOLS)), dtype=torch.bool)
    mask[:, 0] = True
    for protocol in active_protocols:
        mask[:, protocol] = True
    return mask


# -----------------------------------------------------------------------------
# Model fitting and held-out response prediction
# -----------------------------------------------------------------------------

def train_one(representation, seed, data, normalizers):
    set_seed(seed)
    tensors = as_tensors(data)
    model = MechanismEncoderDecoder(
        sequence_dim=tensors["sequence"].shape[-1],
        geometry_dim=tensors["geometry"].shape[-1],
        action_dim=tensors["actions"].shape[-1],
        response_dim=tensors["responses"].shape[-1],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    loss_function = nn.MSELoss()
    times = time_features(tensors["sequence"].shape[2])
    train_assemblies = torch.nonzero(
        torch.from_numpy(data["splits"] == 0), as_tuple=False
    ).flatten()
    validation_assemblies = torch.nonzero(
        torch.from_numpy(data["splits"] == 1), as_tuple=False
    ).flatten()
    train_examples = torch.cartesian_prod(train_assemblies, torch.tensor(ACTIVE_PROTOCOLS))
    validation_examples = torch.cartesian_prod(validation_assemblies, torch.tensor(ACTIVE_PROTOCOLS))
    generator = torch.Generator().manual_seed(seed)
    history = []
    best_state = None
    best_validation = float("inf")
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        permutation = torch.randperm(len(train_examples), generator=generator)
        train_loss = 0.0
        train_count = 0
        for start in range(0, len(permutation), 32):
            examples = train_examples[permutation[start : start + 32]]
            assembly = examples[:, 0]
            target = examples[:, 1]
            context_mask = random_context_mask(target, generator)
            prediction, _ = model(
                tensors["sequence"][assembly],
                context_mask,
                tensors["geometry"][assembly],
                tensors["actions"][assembly, target],
                target,
                times,
            )
            loss = loss_function(prediction, tensors["responses"][assembly, target])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(examples)
            train_count += len(examples)

        model.eval()
        validation_loss = 0.0
        validation_count = 0
        with torch.no_grad():
            for start in range(0, len(validation_examples), 64):
                examples = validation_examples[start : start + 64]
                assembly = examples[:, 0]
                target = examples[:, 1]
                mask = torch.ones((len(examples), len(PROTOCOLS)), dtype=torch.bool)
                mask[torch.arange(len(examples)), target] = False
                prediction, _ = model(
                    tensors["sequence"][assembly],
                    mask,
                    tensors["geometry"][assembly],
                    tensors["actions"][assembly, target],
                    target,
                    times,
                )
                loss = loss_function(prediction, tensors["responses"][assembly, target])
                validation_loss += float(loss) * len(examples)
                validation_count += len(examples)
        train_loss /= train_count
        validation_loss /= validation_count
        history.append(
            {
                "representation": representation,
                "seed": seed,
                "epoch": epoch + 1,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_validation - 1.0e-5:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    checkpoint = {
        "state_dict": model.state_dict(),
        "representation": representation,
        "seed": seed,
        "architecture": {
            "sequence_dim": tensors["sequence"].shape[-1],
            "geometry_dim": tensors["geometry"].shape[-1],
            "action_dim": tensors["actions"].shape[-1],
            "response_dim": tensors["responses"].shape[-1],
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "decoder_dim": DECODER_DIM,
        },
        "normalizers": {
            key: {"mean": value.mean, "std": value.std}
            for key, value in normalizers.items()
        },
        "best_validation_mse": best_validation,
    }
    torch.save(checkpoint, MODEL_DIR / f"{representation}_encoder_decoder_seed{seed}.pt")
    return model, history


def evaluate_prediction(model, representation, seed, data, normalizers, split_code):
    tensors = as_tensors(data)
    times = time_features(tensors["sequence"].shape[2])
    assembly_indices = torch.nonzero(
        torch.from_numpy(data["splits"] == split_code), as_tuple=False
    ).flatten()
    rows = []
    response_normalizer = normalizers["response"]
    model.eval()
    with torch.no_grad():
        for budget in range(4):
            for target in ACTIVE_PROTOCOLS:
                available = [protocol for protocol in ACTIVE_PROTOCOLS if protocol != target]
                combinations = list(itertools.combinations(available, budget))
                predictions = []
                targets = []
                for active_context in combinations:
                    mask = fixed_context_mask(len(assembly_indices), active_context)
                    target_tensor = torch.full((len(assembly_indices),), target, dtype=torch.long)
                    prediction, _ = model(
                        tensors["sequence"][assembly_indices],
                        mask,
                        tensors["geometry"][assembly_indices],
                        tensors["actions"][assembly_indices, target],
                        target_tensor,
                        times,
                    )
                    predictions.append(prediction.numpy())
                    targets.append(tensors["responses"][assembly_indices, target].numpy())
                prediction_normalized = np.concatenate(predictions)
                target_normalized = np.concatenate(targets)
                prediction_actual = response_normalizer.inverse(prediction_normalized)
                target_actual = response_normalizer.inverse(target_normalized)
                error_normalized = prediction_normalized - target_normalized
                error_actual = prediction_actual - target_actual
                rows.append(
                    {
                        "representation": representation,
                        "seed": seed,
                        "split": "validation" if split_code == 1 else "ood_test",
                        "context_active_probes": budget,
                        "target_protocol": PROTOCOLS[target],
                        "context_combinations": len(combinations),
                        "examples": len(prediction_normalized),
                        "normalized_rmse": float(np.sqrt(np.mean(error_normalized**2))),
                        "normalized_mae": float(np.mean(np.abs(error_normalized))),
                        "primary_position_rmse": float(np.sqrt(np.mean(error_actual[:, :, 0:2] ** 2))),
                        "primary_angular_velocity_mae": float(np.mean(np.abs(error_actual[:, :, 6]))),
                        "reaction_mae": float(np.mean(np.abs(error_actual[:, :, 14:16]))),
                    }
                )
    return rows


# -----------------------------------------------------------------------------
# Frozen-latent family evaluation
# -----------------------------------------------------------------------------

def extract_latents(model, data, active_protocols):
    tensors = as_tensors(data)
    mask = fixed_context_mask(len(tensors["sequence"]), active_protocols)
    model.eval()
    with torch.no_grad():
        latent, _ = model.encode(tensors["sequence"], mask, tensors["geometry"])
    return latent.numpy()


def evaluate_latents(model, representation, seed, data):
    labels = data["labels"]
    splits = data["splits"]
    train = splits == 0
    development = splits != 2
    test = splits == 2
    rows = []
    latent_store = {}
    for budget in range(len(ACTIVE_PROTOCOLS) + 1):
        combinations = list(itertools.combinations(ACTIVE_PROTOCOLS, budget))
        for active_context in combinations:
            latent = extract_latents(model, data, active_context)
            classifier = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
                    ),
                ]
            )
            classifier.fit(latent[development], labels[development])
            prediction = classifier.predict(latent[test])
            rows.append(
                {
                    "representation": representation,
                    "seed": seed,
                    "context_active_probes": budget,
                    "protocols": "+".join(PROTOCOLS[index] for index in (0,) + active_context),
                    "accuracy": accuracy_score(labels[test], prediction),
                    "balanced_accuracy": balanced_accuracy_score(labels[test], prediction),
                    "macro_f1": f1_score(labels[test], prediction, average="macro"),
                }
            )
            if budget == len(ACTIVE_PROTOCOLS):
                latent_store["all_probes"] = latent

    all_latent = latent_store["all_probes"]
    same_family_distances = []
    different_family_same_geometry = []
    for family in range(len(FAMILIES)):
        family_rows = np.flatnonzero((labels == family) & test)
        for first, second in itertools.combinations(family_rows, 2):
            same_family_distances.append(float(np.linalg.norm(all_latent[first] - all_latent[second])))
    for local_index in range(60, 80):
        matched_rows = np.flatnonzero((data["local_indices"] == local_index) & test)
        for first, second in itertools.combinations(matched_rows, 2):
            different_family_same_geometry.append(
                float(np.linalg.norm(all_latent[first] - all_latent[second]))
            )
    distance_row = {
        "representation": representation,
        "seed": seed,
        "same_family_different_geometry_mean": float(np.mean(same_family_distances)),
        "different_family_same_geometry_mean": float(np.mean(different_family_same_geometry)),
        "separation_ratio": float(
            np.mean(different_family_same_geometry) / max(np.mean(same_family_distances), 1e-8)
        ),
    }
    return rows, distance_row, all_latent


# -----------------------------------------------------------------------------
# Simple prediction reference and experiment driver
# -----------------------------------------------------------------------------

def protocol_mean_baseline(data, normalizers):
    train = data["splits"] == 0
    test = data["splits"] == 2
    rows = []
    for target in ACTIVE_PROTOCOLS:
        template = data["responses"][train, target].mean(axis=0, keepdims=True)
        prediction = np.repeat(template, int(test.sum()), axis=0)
        target_values = data["responses"][test, target]
        error = prediction - target_values
        rows.append(
            {
                "representation": "protocol_mean_baseline",
                "seed": -1,
                "split": "ood_test",
                "context_active_probes": 0,
                "target_protocol": PROTOCOLS[target],
                "context_combinations": 1,
                "examples": int(test.sum()),
                "normalized_rmse": float(np.sqrt(np.mean(error**2))),
                "normalized_mae": float(np.mean(np.abs(error))),
                "primary_position_rmse": float("nan"),
                "primary_angular_velocity_mae": float("nan"),
                "reaction_mae": float("nan"),
            }
        )
    return rows


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(6, torch.get_num_threads()))
    histories = []
    prediction_rows = []
    latent_rows = []
    distance_rows = []
    latent_outputs = {}

    for representation in ("absolute", "relational"):
        data, normalizers = prepare_data(representation)
        if representation == "relational":
            prediction_rows.extend(protocol_mean_baseline(data, normalizers))
        for seed in SEEDS:
            model, history = train_one(representation, seed, data, normalizers)
            histories.extend(history)
            prediction_rows.extend(
                evaluate_prediction(model, representation, seed, data, normalizers, split_code=1)
            )
            prediction_rows.extend(
                evaluate_prediction(model, representation, seed, data, normalizers, split_code=2)
            )
            rows, distance, latent = evaluate_latents(model, representation, seed, data)
            latent_rows.extend(rows)
            distance_rows.append(distance)
            latent_outputs[f"{representation}_seed{seed}"] = latent.astype(np.float32)
            print(
                f"{representation} seed {seed}: epochs={len(history)}, "
                f"best validation MSE={min(row['validation_mse'] for row in history):.4f}"
            )

    write_csv(ARTIFACT_DIR / "neural_training_history.csv", histories)
    write_csv(ARTIFACT_DIR / "neural_prediction_metrics.csv", prediction_rows)
    write_csv(ARTIFACT_DIR / "neural_latent_metrics.csv", latent_rows)
    write_csv(ARTIFACT_DIR / "neural_latent_distances.csv", distance_rows)
    np.savez_compressed(
        ARTIFACT_DIR / "learned_latents.npz",
        **latent_outputs,
        labels=data["labels"],
        split_codes=data["splits"],
        local_assembly_indices=data["local_indices"],
    )

    summary = {}
    for representation in ("absolute", "relational"):
        prediction = [
            row
            for row in prediction_rows
            if row["representation"] == representation
            and row["split"] == "ood_test"
            and row["context_active_probes"] == 3
        ]
        latent = [
            row
            for row in latent_rows
            if row["representation"] == representation
            and row["context_active_probes"] == 4
        ]
        distances = [row for row in distance_rows if row["representation"] == representation]
        summary[representation] = {
            "held_out_probe_normalized_rmse_mean": float(
                np.mean([row["normalized_rmse"] for row in prediction])
            ),
            "latent_family_accuracy_mean": float(np.mean([row["accuracy"] for row in latent])),
            "latent_separation_ratio_mean": float(
                np.mean([row["separation_ratio"] for row in distances])
            ),
            "neural_seeds": len(SEEDS),
        }
    (ARTIFACT_DIR / "neural_headline.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
