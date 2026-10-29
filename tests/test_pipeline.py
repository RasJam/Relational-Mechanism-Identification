"""Behavioral tests for the active constraint benchmark."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from config import (
    FAMILIES,
    PROTOCOLS,
    RAW_FRAME_FEATURES,
    RELATIONAL_FRAME_FEATURES,
    RELATIONAL_GEOMETRY_FEATURES,
)
from simulator import geometry_vectors, sample_config, simulate
from train_encoder_decoder import MechanismEncoderDecoder


# -----------------------------------------------------------------------------
# Simulator invariants and physical response tests
# -----------------------------------------------------------------------------

class SimulatorTests(unittest.TestCase):
    def test_geometry_is_matched_across_families(self):
        vectors = []
        for family in FAMILIES:
            config = sample_config(family, 7, "push_x")
            vectors.append(geometry_vectors(config))
        for absolute, relational in vectors[1:]:
            np.testing.assert_allclose(absolute, vectors[0][0])
            np.testing.assert_allclose(relational, vectors[0][1])

    def test_passive_observation_is_identical(self):
        outputs = [simulate(sample_config(family, 5, "passive")) for family in FAMILIES]
        for output in outputs[1:]:
            np.testing.assert_allclose(output["raw"], outputs[0]["raw"], atol=1e-7)
            np.testing.assert_allclose(output["relational"], outputs[0]["relational"], atol=1e-7)

    def test_active_trials_have_recorded_inputs(self):
        names = list(RAW_FRAME_FEATURES)
        action = [names.index(name) for name in ("input_force_x", "input_force_y", "input_torque")]
        for protocol in PROTOCOLS[1:]:
            output = simulate(sample_config("revolute_hinge", 2, protocol))
            self.assertGreater(float(np.max(np.abs(output["raw"][:, action]))), 0.0)

    def test_gear_transmits_opposite_rotation(self):
        output = simulate(sample_config("gear_pair", 4, "torque_ccw"))
        names = list(RELATIONAL_FRAME_FEATURES)
        primary = output["relational"][:, names.index("primary_relative_angular_velocity")]
        secondary = output["relational"][:, names.index("secondary_relative_angular_velocity")]
        moving = (np.abs(primary) > 0.05) & (np.abs(secondary) > 0.05)
        self.assertGreater(int(moving.sum()), 10)
        self.assertLess(float(np.mean(primary[moving] * secondary[moving])), 0.0)

    def test_no_answer_features(self):
        prohibited = ("family", "constraint", "allowed", "blocked", "mechanism")
        names = list(RAW_FRAME_FEATURES) + list(RELATIONAL_FRAME_FEATURES) + list(
            RELATIONAL_GEOMETRY_FEATURES
        )
        self.assertFalse(any(word in name for word in prohibited for name in names))


# -----------------------------------------------------------------------------
# Neural architecture interface test
# -----------------------------------------------------------------------------

class NeuralShapeTests(unittest.TestCase):
    def test_encoder_decoder_shapes(self):
        model = MechanismEncoderDecoder(35, 21, 5, 16)
        context = torch.zeros((3, 5, 40, 35))
        mask = torch.tensor([[1, 1, 0, 0, 0]] * 3, dtype=torch.bool)
        geometry = torch.zeros((3, 21))
        action = torch.zeros((3, 40, 5))
        protocol = torch.tensor([1, 2, 3])
        time = torch.zeros((40, 3))
        prediction, latent = model(context, mask, geometry, action, protocol, time)
        self.assertEqual(tuple(prediction.shape), (3, 40, 16))
        self.assertEqual(tuple(latent.shape), (3, 16))


if __name__ == "__main__":
    unittest.main()
