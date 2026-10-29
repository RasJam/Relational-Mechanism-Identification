"""Natural 2D assemblies and word-free relational measurements."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pymunk

from config import (
    ABSOLUTE_GEOMETRY_FEATURES,
    ASSEMBLIES_PER_FAMILY,
    BODY_ROLES,
    DT,
    FAMILIES,
    FRAMES,
    MASTER_SEED,
    PROTOCOLS,
    RAW_FRAME_FEATURES,
    RELATIONAL_FRAME_FEATURES,
    RELATIONAL_GEOMETRY_FEATURES,
    TRAIN_ASSEMBLIES,
    VALIDATION_ASSEMBLIES,
)


# -----------------------------------------------------------------------------
# Geometry helpers and data containers
# -----------------------------------------------------------------------------

SHAPE_CODES = {"box": 0.0, "disc": 1.0, "capsule": 2.0, "gear": 3.0}


def rotate(vector, angle):
    x, y = vector
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array((cosine * x - sine * y, sine * x + cosine * y), dtype=np.float64)


def local_vector(world_vector, base_angle):
    return rotate(world_vector, -base_angle)


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def inertia_for(shape, mass, width, height):
    if shape in {"disc", "gear"}:
        return pymunk.moment_for_circle(mass, 0.0, width / 2.0)
    if shape == "capsule":
        segment = max(width - height, 0.05 * width)
        return pymunk.moment_for_segment(mass, (-segment / 2.0, 0.0), (segment / 2.0, 0.0), height / 2.0)
    return pymunk.moment_for_box(mass, (width, height))


@dataclass(frozen=True)
class AssemblyConfig:
    family: str
    assembly_index: int
    protocol: str
    seed: int
    split: str
    ood_regime: str
    global_x: float
    global_y: float
    global_angle: float
    scale: float
    base_width: float
    base_height: float
    base_mass: float
    primary_width: float
    primary_height: float
    primary_mass: float
    primary_shape: str
    secondary_width: float
    secondary_height: float
    secondary_mass: float
    secondary_shape: str
    primary_anchor_x: float
    primary_anchor_y: float
    secondary_anchor_x: float
    secondary_anchor_y: float
    actuation_x: float
    actuation_y: float
    travel: float
    initial_coordinate: float
    force_magnitude: float
    torque_magnitude: float
    friction: float
    damping: float
    gear_ratio: float
    teeth_primary: int
    teeth_secondary: int

    def metadata(self):
        return asdict(self)


@dataclass
class World:
    space: pymunk.Space
    base: pymunk.Body
    primary: pymunk.Body
    secondary: pymunk.Body
    constraints: list
    config: AssemblyConfig
    previous_primary_velocity: np.ndarray
    previous_primary_angular_velocity: float
    previous_secondary_velocity: np.ndarray
    previous_secondary_angular_velocity: float
    cumulative_work: float = 0.0


# -----------------------------------------------------------------------------
# Deterministic train, validation, and OOD sampling
# -----------------------------------------------------------------------------

def split_for_index(assembly_index):
    if assembly_index < TRAIN_ASSEMBLIES:
        return "train"
    if assembly_index < TRAIN_ASSEMBLIES + VALIDATION_ASSEMBLIES:
        return "validation"
    return "ood_test"


def ood_regime_for_index(assembly_index):
    if split_for_index(assembly_index) != "ood_test":
        return "development"
    regimes = ("pose", "scale", "mass", "shape", "combined")
    return regimes[(assembly_index - TRAIN_ASSEMBLIES - VALIDATION_ASSEMBLIES) // 4]


def sample_config(family: str, assembly_index: int, protocol: str) -> AssemblyConfig:
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    if not 0 <= assembly_index < ASSEMBLIES_PER_FAMILY:
        raise ValueError(f"assembly index out of range: {assembly_index}")

    family_index = FAMILIES.index(family)
    shared_rng = np.random.default_rng(MASTER_SEED + assembly_index)
    regime = ood_regime_for_index(assembly_index)

    global_x = shared_rng.uniform(-2.5, 2.5)
    global_y = shared_rng.uniform(-2.5, 2.5)
    global_angle = shared_rng.uniform(math.radians(-45.0), math.radians(45.0))
    scale = shared_rng.uniform(0.8, 1.25)
    base_mass = shared_rng.uniform(1.2, 2.8)
    primary_mass = shared_rng.uniform(0.8, 2.6)
    secondary_mass = shared_rng.uniform(0.8, 2.6)
    shape_options = ("box", "disc")
    primary_shape = shape_options[assembly_index % len(shape_options)]
    secondary_shape = shape_options[(assembly_index + 1) % len(shape_options)]

    if regime in {"pose", "combined"}:
        global_x = shared_rng.uniform(6.0, 10.0)
        global_y = shared_rng.uniform(-9.0, -5.0)
        global_angle = shared_rng.uniform(math.radians(105.0), math.radians(255.0))
    if regime in {"scale", "combined"}:
        scale = shared_rng.uniform(1.55, 2.05)
    if regime in {"mass", "combined"}:
        base_mass = shared_rng.uniform(3.5, 6.0)
        primary_mass = shared_rng.uniform(3.5, 6.5)
        secondary_mass = shared_rng.uniform(3.2, 6.2)
    if regime in {"shape", "combined"}:
        primary_shape = "capsule"
        secondary_shape = "capsule"

    base_width = scale * shared_rng.uniform(2.4, 3.2)
    base_height = scale * shared_rng.uniform(0.55, 0.85)
    primary_width = scale * shared_rng.uniform(1.6, 2.35)
    primary_height = scale * shared_rng.uniform(0.36, 0.64)
    secondary_width = scale * shared_rng.uniform(1.0, 1.65)
    secondary_height = scale * shared_rng.uniform(0.30, 0.56)
    travel = scale * shared_rng.uniform(2.1, 3.0)
    # Matched-rig design: for a given assembly index, every family has exactly
    # the same visible component geometry, mass, pose, and actuation point.
    # Only the hidden constraint instantiated in build_world changes.
    separation = 0.52 * (primary_width + secondary_width)
    primary_anchor_x = -0.5 * separation
    primary_anchor_y = 0.0
    secondary_anchor_x = 0.5 * separation
    secondary_anchor_y = 0.0
    actuation_x = 0.0
    actuation_y = 0.43 * primary_width
    initial_coordinate = 0.0
    gear_ratio = secondary_width / primary_width
    teeth_primary = int(shared_rng.integers(12, 21))
    teeth_secondary = max(9, int(round(teeth_primary * gear_ratio)))

    seed = MASTER_SEED + 100_000 * family_index + 1_000 * assembly_index + PROTOCOLS.index(protocol)
    return AssemblyConfig(
        family=family,
        assembly_index=assembly_index,
        protocol=protocol,
        seed=seed,
        split=split_for_index(assembly_index),
        ood_regime=regime,
        global_x=global_x,
        global_y=global_y,
        global_angle=global_angle,
        scale=scale,
        base_width=base_width,
        base_height=base_height,
        base_mass=base_mass,
        primary_width=primary_width,
        primary_height=primary_height,
        primary_mass=primary_mass,
        primary_shape=primary_shape,
        secondary_width=secondary_width,
        secondary_height=secondary_height,
        secondary_mass=secondary_mass,
        secondary_shape=secondary_shape,
        primary_anchor_x=primary_anchor_x,
        primary_anchor_y=primary_anchor_y,
        secondary_anchor_x=secondary_anchor_x,
        secondary_anchor_y=secondary_anchor_y,
        actuation_x=actuation_x,
        actuation_y=actuation_y,
        travel=travel,
        initial_coordinate=initial_coordinate,
        force_magnitude=primary_mass * scale * shared_rng.uniform(8.0, 12.0),
        torque_magnitude=primary_mass * scale**2 * shared_rng.uniform(5.5, 8.5),
        friction=shared_rng.uniform(0.25, 0.75),
        damping=shared_rng.uniform(0.982, 0.995),
        gear_ratio=gear_ratio,
        teeth_primary=teeth_primary,
        teeth_secondary=teeth_secondary,
    )


# -----------------------------------------------------------------------------
# Pymunk body and constraint construction
# -----------------------------------------------------------------------------

def _make_shape(body, shape_name, width, height, friction):
    if shape_name in {"disc", "gear"}:
        shape = pymunk.Circle(body, width / 2.0)
    elif shape_name == "capsule":
        half_segment = max((width - height) / 2.0, 0.03 * width)
        shape = pymunk.Segment(body, (-half_segment, 0.0), (half_segment, 0.0), height / 2.0)
    else:
        shape = pymunk.Poly.create_box(body, (width, height), radius=0.07 * min(width, height))
    shape.friction = friction
    shape.elasticity = 0.03
    shape.filter = pymunk.ShapeFilter(group=1)
    return shape


def _dynamic_body(mass, shape, width, height):
    return pymunk.Body(mass, inertia_for(shape, mass, width, height))


def _world_point(base, local_point):
    return np.array(base.position, dtype=np.float64) + rotate(local_point, base.angle)


def _place_from_anchor(body, world_anchor, local_anchor, angle):
    body.angle = angle
    body.position = tuple(np.array(world_anchor) - rotate(local_anchor, angle))


def build_world(config: AssemblyConfig) -> World:
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.damping = config.damping
    space.iterations = 40

    base = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    base.position = (config.global_x, config.global_y)
    base.angle = config.global_angle
    base.mechanism_role = "base"

    primary = _dynamic_body(
        config.primary_mass,
        config.primary_shape,
        config.primary_width,
        config.primary_height,
    )
    primary.mechanism_role = "primary"

    secondary_dynamic = config.family == "gear_pair"
    if secondary_dynamic:
        secondary = _dynamic_body(
            config.secondary_mass,
            config.secondary_shape,
            config.secondary_width,
            config.secondary_height,
        )
    else:
        secondary = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    secondary.mechanism_role = "secondary"

    primary_anchor = _world_point(base, (config.primary_anchor_x, config.primary_anchor_y))
    secondary_anchor = _world_point(base, (config.secondary_anchor_x, config.secondary_anchor_y))
    relative_angle = config.initial_coordinate if config.family != "prismatic_guide" else 0.0

    if config.family in {"fixed_coupling", "revolute_hinge"}:
        _place_from_anchor(primary, primary_anchor, (0.0, 0.0), base.angle + relative_angle)
    elif config.family == "prismatic_guide":
        primary.angle = base.angle
        primary.position = tuple(
            _world_point(
                base,
                (config.primary_anchor_x + config.initial_coordinate, config.primary_anchor_y),
            )
        )
    else:
        primary.angle = base.angle + relative_angle
        primary.position = tuple(primary_anchor)

    secondary.angle = base.angle
    secondary.position = tuple(secondary_anchor)

    shapes = [
        _make_shape(base, "box", config.base_width, config.base_height, config.friction),
        _make_shape(
            primary,
            config.primary_shape,
            config.primary_width,
            config.primary_height,
            config.friction,
        ),
        _make_shape(
            secondary,
            config.secondary_shape,
            config.secondary_width,
            config.secondary_height,
            config.friction,
        ),
    ]
    space.add(base, primary, secondary, *shapes)


    # constraints connect bodies, not shapes
    constraints = []
    if config.family in {"fixed_coupling", "revolute_hinge"}:
        pivot = pymunk.PivotJoint(
            base,
            primary,
            (config.primary_anchor_x, config.primary_anchor_y),
            (0.0, 0.0),
        )
        constraints.append(pivot)
        if config.family == "fixed_coupling":
            constraints.append(pymunk.GearJoint(base, primary, relative_angle, 1.0))
    elif config.family == "prismatic_guide":
        constraints.extend(
            [
                pymunk.GrooveJoint(
                    base,
                    primary,
                    (config.primary_anchor_x - config.travel / 2.0, config.primary_anchor_y),
                    (config.primary_anchor_x + config.travel / 2.0, config.primary_anchor_y),
                    (0.0, 0.0),
                ),
                pymunk.GearJoint(base, primary, 0.0, 1.0),
            ]
        )
    elif config.family == "ratchet_wheel":
        constraints.extend(
            [
                pymunk.PivotJoint(
                    base,
                    primary,
                    (config.primary_anchor_x, config.primary_anchor_y),
                    (0.0, 0.0),
                ),
                pymunk.RatchetJoint(base, primary, relative_angle, math.radians(12.0)),
            ]
        )
    elif config.family == "gear_pair":
        primary_pivot = pymunk.PivotJoint(
            base,
            primary,
            (config.primary_anchor_x, config.primary_anchor_y),
            (0.0, 0.0),
        )
        secondary_pivot = pymunk.PivotJoint(
            base,
            secondary,
            (config.secondary_anchor_x, config.secondary_anchor_y),
            (0.0, 0.0),
        )
        # The negative ratio gives physical counter-rotation. Chipmunk keeps
        # secondary_angle * ratio - primary_angle equal to phase.
        ratio = -config.secondary_width / config.primary_width
        phase = secondary.angle * ratio - primary.angle
        gear = pymunk.GearJoint(primary, secondary, phase, ratio)
        constraints.extend([primary_pivot, secondary_pivot, gear])

    for constraint in constraints:
        constraint.collide_bodies = False
        constraint.max_force = 1.0e8
        constraint.error_bias = pow(1.0 - 0.2, 60.0)
    space.add(*constraints)

    # Every family and protocol begins at the same rest state. Any identifying
    # motion must therefore come from a recorded active input.
    primary.velocity = (0.0, 0.0)
    primary.angular_velocity = 0.0
    if secondary_dynamic:
        secondary.angular_velocity = 0.0

    return World(
        space=space,
        base=base,
        primary=primary,
        secondary=secondary,
        constraints=constraints,
        config=config,
        previous_primary_velocity=np.zeros(2),
        previous_primary_angular_velocity=0.0,
        previous_secondary_velocity=np.zeros(2),
        previous_secondary_angular_velocity=0.0,
    )


# -----------------------------------------------------------------------------
# Active probe signals and measured responses
# -----------------------------------------------------------------------------

def probe_input(config: AssemblyConfig, frame: int):
    if config.protocol == "passive" or not 10 <= frame < 55:
        return np.zeros(2, dtype=np.float64), 0.0
    progress = (frame - 10) / 45.0
    envelope = math.sin(math.pi * progress) ** 2
    if config.protocol == "push_x":
        local_force = np.array((config.force_magnitude * envelope, 0.0))
        torque = 0.0
    elif config.protocol == "push_y":
        local_force = np.array((0.0, config.force_magnitude * envelope))
        torque = 0.0
    elif config.protocol == "torque_ccw":
        local_force = np.zeros(2)
        torque = config.torque_magnitude * envelope
    else:
        local_force = np.zeros(2)
        torque = -config.torque_magnitude * envelope
    return rotate(local_force, config.global_angle), torque


def _constraint_reactions(constraints):
    linear_impulse = 0.0
    angular_impulse = 0.0
    linear_types = (pymunk.PivotJoint, pymunk.GrooveJoint, pymunk.PinJoint, pymunk.SlideJoint)
    for constraint in constraints:
        value = abs(float(constraint.impulse))
        if isinstance(constraint, linear_types):
            linear_impulse += value
        else:
            angular_impulse += value
    return linear_impulse / DT, angular_impulse / DT


def _relative_state(base, body, scale):
    position = local_vector(np.array(body.position) - np.array(base.position), base.angle) / scale
    velocity = local_vector(np.array(body.velocity) - np.array(base.velocity), base.angle) / scale
    angle = wrap_angle(body.angle - base.angle)
    angular_velocity = body.angular_velocity - base.angular_velocity
    return position, velocity, angle, angular_velocity


def _reported_inertias(config):
    return (
        pymunk.moment_for_box(config.base_mass, (config.base_width, config.base_height)),
        inertia_for(
            config.primary_shape,
            config.primary_mass,
            config.primary_width,
            config.primary_height,
        ),
        inertia_for(
            config.secondary_shape,
            config.secondary_mass,
            config.secondary_width,
            config.secondary_height,
        ),
    )


def geometry_vectors(config: AssemblyConfig):
    base_inertia, primary_inertia, secondary_inertia = _reported_inertias(config)
    base_position = np.array((config.global_x, config.global_y))
    primary_anchor_world = base_position + rotate(
        (config.primary_anchor_x, config.primary_anchor_y), config.global_angle
    )
    secondary_anchor_world = base_position + rotate(
        (config.secondary_anchor_x, config.secondary_anchor_y), config.global_angle
    )
    actuation_world = primary_anchor_world + rotate(
        (config.actuation_x, config.actuation_y), config.global_angle
    )
    absolute = np.array(
        (
            config.global_x,
            config.global_y,
            math.sin(config.global_angle),
            math.cos(config.global_angle),
            config.scale,
            config.base_width,
            config.base_height,
            config.base_mass,
            base_inertia,
            config.primary_width,
            config.primary_height,
            config.primary_mass,
            primary_inertia,
            SHAPE_CODES[config.primary_shape],
            config.secondary_width,
            config.secondary_height,
            config.secondary_mass,
            secondary_inertia,
            SHAPE_CODES[config.secondary_shape],
            primary_anchor_world[0],
            primary_anchor_world[1],
            secondary_anchor_world[0],
            secondary_anchor_world[1],
            actuation_world[0],
            actuation_world[1],
            config.travel,
        ),
        dtype=np.float32,
    )
    scale = config.scale
    relational = np.array(
        (
            config.base_width / config.base_height,
            config.primary_width / scale,
            config.primary_height / scale,
            config.primary_width / config.primary_height,
            config.primary_mass / config.base_mass,
            primary_inertia / base_inertia,
            SHAPE_CODES[config.primary_shape],
            config.secondary_width / scale,
            config.secondary_height / scale,
            config.secondary_width / config.secondary_height,
            config.secondary_mass / config.base_mass,
            secondary_inertia / base_inertia,
            SHAPE_CODES[config.secondary_shape],
            config.primary_anchor_x / scale,
            config.primary_anchor_y / scale,
            config.secondary_anchor_x / scale,
            config.secondary_anchor_y / scale,
            config.actuation_x / scale,
            config.actuation_y / scale,
            config.travel / scale,
            config.secondary_width / config.primary_width,
        ),
        dtype=np.float32,
    )
    assert len(absolute) == len(ABSOLUTE_GEOMETRY_FEATURES)
    assert len(relational) == len(RELATIONAL_GEOMETRY_FEATURES)
    return absolute, relational


# -----------------------------------------------------------------------------
# Trajectory simulation
# -----------------------------------------------------------------------------

def simulate(config: AssemblyConfig):
    world = build_world(config)
    raw = np.zeros((FRAMES, len(RAW_FRAME_FEATURES)), dtype=np.float32)
    relational = np.zeros((FRAMES, len(RELATIONAL_FRAME_FEATURES)), dtype=np.float32)
    poses = np.zeros((FRAMES, len(BODY_ROLES), 7), dtype=np.float32)
    absolute_geometry, relational_geometry = geometry_vectors(config)
    _, primary_inertia, secondary_inertia = _reported_inertias(config)
    normalization_force = max(config.primary_mass * config.scale, 1e-8)
    normalization_torque = max(config.primary_mass * config.scale**2, 1e-8)
    normalization_energy = normalization_torque

    for frame in range(FRAMES):
        force, torque = probe_input(config, frame)
        application_world = np.array(world.primary.local_to_world((config.actuation_x, config.actuation_y)))
        world.primary.apply_force_at_world_point(tuple(force), tuple(application_world))
        world.primary.torque += float(torque)
        world.space.step(DT)

        primary_position, primary_velocity, primary_angle, primary_omega = _relative_state(
            world.base, world.primary, config.scale
        )
        secondary_position, secondary_velocity, secondary_angle, secondary_omega = _relative_state(
            world.base, world.secondary, config.scale
        )
        primary_acceleration = (primary_velocity - world.previous_primary_velocity) / DT
        primary_alpha = (primary_omega - world.previous_primary_angular_velocity) / DT
        secondary_acceleration = (secondary_velocity - world.previous_secondary_velocity) / DT
        secondary_alpha = (secondary_omega - world.previous_secondary_angular_velocity) / DT
        force_local = local_vector(force, world.base.angle)
        input_point_local = local_vector(application_world - np.array(world.base.position), world.base.angle) / config.scale
        linear_reaction, angular_reaction = _constraint_reactions(world.constraints)

        primary_energy = 0.5 * config.primary_mass * float(np.dot(world.primary.velocity, world.primary.velocity))
        primary_energy += 0.5 * primary_inertia * world.primary.angular_velocity**2
        secondary_energy = 0.5 * config.secondary_mass * float(np.dot(world.secondary.velocity, world.secondary.velocity))
        secondary_energy += 0.5 * secondary_inertia * world.secondary.angular_velocity**2
        input_power = float(np.dot(force, world.primary.velocity) + torque * world.primary.angular_velocity)
        world.cumulative_work += input_power * DT

        raw_values = []
        for body in (world.base, world.primary, world.secondary):
            raw_values.extend(
                (
                    body.position.x,
                    body.position.y,
                    math.sin(body.angle),
                    math.cos(body.angle),
                    body.velocity.x,
                    body.velocity.y,
                    body.angular_velocity,
                )
            )
        raw_values.extend(
            (
                force[0],
                force[1],
                torque,
                application_world[0],
                application_world[1],
                linear_reaction,
                angular_reaction,
                primary_energy + secondary_energy,
                input_power,
                world.cumulative_work,
            )
        )
        raw[frame] = raw_values

        primary_secondary = primary_position - secondary_position
        pair_angle = wrap_angle(primary_angle - secondary_angle)
        relational[frame] = (
            primary_position[0],
            primary_position[1],
            math.sin(primary_angle),
            math.cos(primary_angle),
            primary_velocity[0],
            primary_velocity[1],
            primary_omega,
            primary_acceleration[0],
            primary_acceleration[1],
            primary_alpha,
            secondary_position[0],
            secondary_position[1],
            math.sin(secondary_angle),
            math.cos(secondary_angle),
            secondary_velocity[0],
            secondary_velocity[1],
            secondary_omega,
            secondary_acceleration[0],
            secondary_acceleration[1],
            secondary_alpha,
            float(np.linalg.norm(primary_secondary)),
            math.sin(pair_angle),
            math.cos(pair_angle),
            primary_omega - secondary_omega,
            force_local[0] / normalization_force,
            force_local[1] / normalization_force,
            torque / normalization_torque,
            input_point_local[0],
            input_point_local[1],
            linear_reaction / normalization_force,
            angular_reaction / normalization_torque,
            primary_energy / normalization_energy,
            secondary_energy / normalization_energy,
            input_power / normalization_energy,
            world.cumulative_work / normalization_energy,
        )

        body_specs = (
            (world.base, config.base_width, config.base_height, SHAPE_CODES["box"]),
            (world.primary, config.primary_width, config.primary_height, SHAPE_CODES[config.primary_shape]),
            (world.secondary, config.secondary_width, config.secondary_height, SHAPE_CODES[config.secondary_shape]),
        )
        for body_index, (body, width, height, shape_code) in enumerate(body_specs):
            poses[frame, body_index] = (
                body.position.x,
                body.position.y,
                body.angle,
                width,
                height,
                shape_code,
                1.0,
            )

        world.previous_primary_velocity = primary_velocity
        world.previous_primary_angular_velocity = primary_omega
        world.previous_secondary_velocity = secondary_velocity
        world.previous_secondary_angular_velocity = secondary_omega

    # Constraint solvers leave family-specific floating-point residues near
    # machine zero. Removing them prevents classifiers from exploiting solver
    # fingerprints when every observable passive state is physically identical.
    raw[np.abs(raw) < 1.0e-8] = 0.0
    relational[np.abs(relational) < 1.0e-8] = 0.0
    poses[np.abs(poses) < 1.0e-8] = 0.0
    if not all(np.isfinite(array).all() for array in (raw, relational, poses)):
        raise FloatingPointError(f"non-finite simulation: {config.family}/{config.assembly_index}/{config.protocol}")
    return {
        "raw": raw,
        "relational": relational,
        "absolute_geometry": absolute_geometry,
        "relational_geometry": relational_geometry,
        "poses": poses,
        "metadata": config.metadata(),
    }
