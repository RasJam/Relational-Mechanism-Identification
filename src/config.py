"""Configuration for the active relational constraint benchmark."""

from __future__ import annotations

from pathlib import Path


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
FIGURE_DIR = ARTIFACT_DIR / "figures"
ANIMATION_DIR = ARTIFACT_DIR / "animations"

# -----------------------------------------------------------------------------
# Experimental design
# -----------------------------------------------------------------------------

FAMILIES = (
    "fixed_coupling",
    "revolute_hinge",
    "prismatic_guide",
    "ratchet_wheel",
    "gear_pair",
)
FAMILY_LABELS = {name: index for index, name in enumerate(FAMILIES)}

PROTOCOLS = ("passive", "push_x", "push_y", "torque_ccw", "torque_cw")
PROTOCOL_LABELS = {name: index for index, name in enumerate(PROTOCOLS)}

ASSEMBLIES_PER_FAMILY = 80
TRAIN_ASSEMBLIES = 52
VALIDATION_ASSEMBLIES = 8
OOD_ASSEMBLIES = 20
TRIALS_PER_ASSEMBLY = len(PROTOCOLS)
FRAMES = 120
DT = 1.0 / 60.0
MASTER_SEED = 20260722

# -----------------------------------------------------------------------------
# Frame-level world-state channels
# -----------------------------------------------------------------------------

BODY_ROLES = ("base", "primary", "secondary")
BODY_STATE_FIELDS = (
    "x",
    "y",
    "sin_angle",
    "cos_angle",
    "velocity_x",
    "velocity_y",
    "angular_velocity",
)
RAW_FRAME_FEATURES = tuple(
    f"{role}_{field}" for role in BODY_ROLES for field in BODY_STATE_FIELDS
) + (
    "input_force_x",
    "input_force_y",
    "input_torque",
    "input_point_x",
    "input_point_y",
    "linear_reaction",
    "angular_reaction",
    "kinetic_energy",
    "input_power",
    "cumulative_work",
)

# -----------------------------------------------------------------------------
# Static assembly geometry
# -----------------------------------------------------------------------------

ABSOLUTE_GEOMETRY_FEATURES = (
    "global_x",
    "global_y",
    "global_sin_angle",
    "global_cos_angle",
    "scale",
    "base_width",
    "base_height",
    "base_mass",
    "base_inertia",
    "primary_width",
    "primary_height",
    "primary_mass",
    "primary_inertia",
    "primary_shape_code",
    "secondary_width",
    "secondary_height",
    "secondary_mass",
    "secondary_inertia",
    "secondary_shape_code",
    "primary_anchor_world_x",
    "primary_anchor_world_y",
    "secondary_anchor_world_x",
    "secondary_anchor_world_y",
    "actuation_point_world_x",
    "actuation_point_world_y",
    "travel",
)

RELATIONAL_GEOMETRY_FEATURES = (
    "base_aspect_ratio",
    "primary_width_over_scale",
    "primary_height_over_scale",
    "primary_aspect_ratio",
    "primary_mass_ratio",
    "primary_inertia_ratio",
    "primary_shape_code",
    "secondary_width_over_scale",
    "secondary_height_over_scale",
    "secondary_aspect_ratio",
    "secondary_mass_ratio",
    "secondary_inertia_ratio",
    "secondary_shape_code",
    "primary_anchor_local_x",
    "primary_anchor_local_y",
    "secondary_anchor_local_x",
    "secondary_anchor_local_y",
    "actuation_point_local_x",
    "actuation_point_local_y",
    "travel_over_scale",
    "component_size_ratio",
)

# -----------------------------------------------------------------------------
# Object-centred relational dynamics
# -----------------------------------------------------------------------------

RELATIONAL_FRAME_FEATURES = (
    "primary_relative_x",
    "primary_relative_y",
    "primary_relative_sin_angle",
    "primary_relative_cos_angle",
    "primary_relative_velocity_x",
    "primary_relative_velocity_y",
    "primary_relative_angular_velocity",
    "primary_relative_acceleration_x",
    "primary_relative_acceleration_y",
    "primary_relative_angular_acceleration",
    "secondary_relative_x",
    "secondary_relative_y",
    "secondary_relative_sin_angle",
    "secondary_relative_cos_angle",
    "secondary_relative_velocity_x",
    "secondary_relative_velocity_y",
    "secondary_relative_angular_velocity",
    "secondary_relative_acceleration_x",
    "secondary_relative_acceleration_y",
    "secondary_relative_angular_acceleration",
    "primary_secondary_distance",
    "primary_secondary_sin_angle",
    "primary_secondary_cos_angle",
    "primary_secondary_angular_velocity",
    "input_force_local_x",
    "input_force_local_y",
    "input_torque_normalized",
    "input_point_local_x",
    "input_point_local_y",
    "linear_reaction_normalized",
    "angular_reaction_normalized",
    "primary_kinetic_energy_normalized",
    "secondary_kinetic_energy_normalized",
    "input_power_normalized",
    "cumulative_work_normalized",
)

# -----------------------------------------------------------------------------
# Encoder-decoder targets and action channels
# -----------------------------------------------------------------------------

# Inputs and geometry are intentionally absent from the decoder targets.
RESPONSE_FEATURES = (
    "primary_relative_x",
    "primary_relative_y",
    "primary_relative_sin_angle",
    "primary_relative_cos_angle",
    "primary_relative_velocity_x",
    "primary_relative_velocity_y",
    "primary_relative_angular_velocity",
    "secondary_relative_x",
    "secondary_relative_y",
    "secondary_relative_sin_angle",
    "secondary_relative_cos_angle",
    "secondary_relative_velocity_x",
    "secondary_relative_velocity_y",
    "secondary_relative_angular_velocity",
    "linear_reaction_normalized",
    "angular_reaction_normalized",
)

RELATIONAL_ACTION_FEATURES = (
    "input_force_local_x",
    "input_force_local_y",
    "input_torque_normalized",
    "input_point_local_x",
    "input_point_local_y",
)
RAW_ACTION_FEATURES = (
    "input_force_x",
    "input_force_y",
    "input_torque",
    "input_point_x",
    "input_point_y",
)
