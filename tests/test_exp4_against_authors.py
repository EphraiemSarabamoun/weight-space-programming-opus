"""Replay the authors' Experiment 4 artifacts against their published protocol.

Run from the repository root with:
    .venv-kinetix/bin/python tests/test_exp4_against_authors.py

Set WSP_REFERENCE to the cloned lm_understand-logs directory if it is elsewhere.
All results and per-tensor errors are printed; no reference artifact is modified.
The authors did not publish their random keys. Base-policy trajectories therefore
have stochastic tolerances; the edited controller's reported path is effectively
deterministic and is also checked at every published sample point.
"""

import json
import os
from pathlib import Path
import re
import sys
import time

# Apply conservative CPU settings before JAX is imported. Run only this process.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wsp.edits import apply
from wsp.kinetix_eval import (
    ACTION_SIZES, TENSOR_HEADER, evaluate_with_trace, make_environment, observation,
    shown_network,
)


REFERENCE = Path(os.environ.get("WSP_REFERENCE", str(Path.home() / "projects/_ref/lm_understand-logs")))
EXP4 = REFERENCE / "exp4_fix_car"


def check_rounding(prompt, shown):
    """Map every actor tensor to the PPO checkpoint and test the half-scale bound."""
    checkpoint = np.load(EXP4 / "networks/ppo_final_seed0.npz")
    scales = {match.group(1): float(match.group(3)) for match in TENSOR_HEADER.finditer(prompt)}
    details = {}
    for name, value in shown.items():
        layer, suffix = name.split(".")
        field = "kernel" if suffix == "W" else "bias"
        if layer == "embed":
            prefix = "params/Dense_0"
        elif layer == "actor_head":
            prefix = "params/GeneralActorCriticRNN_0/Dense_0"
        else:
            prefix = f"params/GeneralActorCriticRNN_0/DenseAndActivation_{layer[-1]}/Dense_0"
        original = checkpoint[f"{prefix}/{field}"].astype(np.float64)
        maximum = float(np.max(np.abs(value - original)))
        bound = scales[name] / 2
        details[name] = {"shape": list(value.shape), "scale": scales[name],
                         "max_abs_error": maximum, "rounding_bound": bound}
        print("a tensor", name, json.dumps(details[name]), flush=True)
        assert value.dtype == np.float64, f"a: {name} did not preserve float64 parsing"
        assert value.shape == original.shape, f"a: {name} checkpoint shape differs"
        assert maximum <= bound + 1e-12, f"a: {name} error {maximum} exceeds half-scale {bound}"
    print("a PASS", json.dumps({"tensors": len(shown), "entries": sum(v.size for v in shown.values())}), flush=True)
    return details


def check_edits(shown, authors):
    """Require exact equality with the scored float64 actor, including untouched weights."""
    edited = {name: value.copy() for name, value in shown.items()}
    report = apply((EXP4 / "runs/oneshot_repair/reply/answer.md").read_text(), edited)
    print("b edit_report", json.dumps(report), flush=True)
    assert report["statements"] == report["applied"] == 51 and not report["errors"], \
        f"b: expected 51/51 successful statements, got {report}"
    artifact = np.load(EXP4 / "networks/edited_actor.npz")
    assert set(artifact.files) == set(edited), f"b: edited artifact keys differ: {artifact.files}"
    changed = {}
    for name, value in edited.items():
        assert np.array_equal(value, artifact[name]), \
            f"b: {name} differs from the scored actor; max error {np.max(np.abs(value - artifact[name]))}"
        count = int(np.count_nonzero(value != shown[name]))
        changed[name] = count
        print("b tensor", name, json.dumps({"changed": count, "entries": value.size, "max_artifact_error": 0.0}), flush=True)
        assert count == authors["changes"]["per_tensor"][name]["changed"], f"b: {name} changed count differs"
    assert sum(changed.values()) == 6073, f"b: changed {sum(changed.values())}, expected 6073"
    print("b PASS", json.dumps({"statements": 51, "changed": sum(changed.values()),
                              "entries": sum(v.size for v in shown.values()), "exact_artifact_match": True}), flush=True)
    return edited, report, changed


def _numpy_observation(state):
    """Independently assemble all 679 features from the prompt's listed layout."""
    eye = np.eye(4)

    def shape_features(shapes, densities, roles, index):
        """Build the common sixteen features without calling Kinetix's renderer."""
        angle = float(shapes.rotation[index])
        return np.concatenate((
            np.asarray(shapes.position[index]), np.asarray(shapes.velocity[index]),
            [shapes.inverse_mass[index], shapes.inverse_inertia[index], densities[index],
             np.tanh(shapes.angular_velocity[index] / 10)], eye[int(roles[index])],
            [np.sin(angle), np.cos(angle), shapes.friction[index], shapes.restitution[index]],
        ))

    features = []
    for index in (0, 4, 5, 6, 7, 8, 9, 10, 11):
        if not state.polygon.active[index]:
            features.extend(np.zeros(26))
            continue
        vertices = np.asarray(state.polygon.vertices[index]).copy()
        vertices[int(state.polygon.n_vertices[index]):] = -1
        features.extend(np.concatenate((shape_features(state.polygon, state.polygon_densities,
                                                       state.polygon_shape_roles, index),
                                        [0], vertices.ravel(), [state.polygon.n_vertices[index] <= 3])))
    for index in range(4):
        if not state.circle.active[index]:
            features.extend(np.zeros(18))
            continue
        features.extend(np.concatenate((shape_features(state.circle, state.circle_densities,
                                                       state.circle_shape_roles, index),
                                        [state.circle.radius[index], 1])))
    joints = state.joint
    for index in range(6):
        if not joints.active[index]:
            features.extend(np.zeros(54))
            continue
        reference = float(joints.rotation[index])
        motor = np.concatenate((
            [joints.motor_speed[index], joints.motor_power[index], joints.motor_on[index],
             joints.motor_has_joint_limits[index]], eye[int(state.motor_bindings[index])],
            np.array([np.sin(joints.min_rotation[index]), np.cos(joints.min_rotation[index]),
                      np.sin(joints.max_rotation[index]), np.cos(joints.max_rotation[index]),
                      joints.min_rotation[index] - reference, joints.max_rotation[index] - reference])
            * bool(joints.motor_has_joint_limits[index]),
        )) * (not bool(joints.is_fixed_joint[index]))
        features.extend(np.concatenate((
            [1, joints.is_fixed_joint[index]], joints.b_relative_pos[index], joints.a_relative_pos[index],
            [np.sin(reference), np.cos(reference)], motor,
            np.eye(16)[int(joints.b_index[index])], np.eye(16)[int(joints.a_index[index])],
        )))
    for index in range(2):
        if not state.thruster.active[index]:
            features.extend(np.zeros(24))
            continue
        thruster = state.thruster
        features.extend(np.concatenate((
            [1], thruster.relative_position[index], np.eye(2)[int(state.thruster_bindings[index])],
            [np.sin(thruster.rotation[index]), np.cos(thruster.rotation[index]), thruster.power[index]],
            np.eye(16)[int(thruster.object_index[index])],
        )))
    features.append(float(state.gravity[1]) / 10)
    return np.clip(np.nan_to_num(features, nan=0., posinf=0., neginf=0.), -10, 10).astype(np.float32)


def check_environment():
    """Verify reset geometry, every observation entry, action mapping and protocol constants."""
    import jax.numpy as jnp

    env, params, state = make_environment()
    static = env.static_env_params
    obs = np.asarray(observation(env, params, state))
    expected = _numpy_observation(state)
    assert obs.shape == (679,), f"observation: got {obs.shape}, expected (679,)"
    np.testing.assert_allclose(obs, expected, rtol=1e-6, atol=1e-6,
                               err_msg="observation: feature order/value differs from prompt")
    assert (static.num_polygons, static.num_circles, static.num_joints, static.num_thrusters) == (12, 4, 6, 2)
    assert static.frame_skip == 2 and params.max_timesteps == 256
    assert params.dt == 1 / 60 and params.dense_reward_scale == 0.2
    assert params.base_motor_speed == 6 and params.base_motor_power == 900
    assert tuple(env.action_type.number_of_dims_per_distribution.tolist()) == ACTION_SIZES
    positions = {0: [2.5, -4.8], 4: [.232, .762], 5: [2.8, .725],
                 6: [2.227, .558], 7: [1.622, .402]}
    for index, position in positions.items():
        np.testing.assert_allclose(state.polygon.position[index], position, atol=.001, rtol=0,
                                   err_msg=f"reset: polygon {index} differs from prompt")
    np.testing.assert_allclose(state.circle.position[:2], [[2.537, .735], [3.107, .75]], atol=.001, rtol=0)
    np.testing.assert_allclose(obs[-1], -.981, atol=1e-7)
    assert np.all(obs[130:234] == 0) and np.all(obs[270:306] == 0)
    assert np.all(obs[414:678] == 0), "observation: unused joints/thrusters are not zero"
    for action_value, motor_value in enumerate((0, 1, -1)):
        action = jnp.array([action_value, 0, 0, 0, 0, 0], dtype=jnp.int32)
        processed = np.asarray(env.action_type.process_action(action, state, static))
        np.testing.assert_array_equal(processed[:2], [motor_value, motor_value])
    # Exercise nonzero velocities and angles as well as the reset's mostly-zero state.
    varied = state.replace(polygon=state.polygon.replace(
        velocity=state.polygon.velocity.at[5].set(jnp.array([1.25, -.75])),
        rotation=state.polygon.rotation.at[5].set(.4),
        angular_velocity=state.polygon.angular_velocity.at[5].set(2.0)))
    np.testing.assert_allclose(observation(env, params, varied), _numpy_observation(varied), atol=1e-6, rtol=1e-6)
    invalid = state.replace(gravity=jnp.array([0., jnp.inf]))
    assert float(observation(env, params, invalid)[678]) == 0, "observation: infinity must map to zero"
    config = {"level": "l/h2_car_ramp", "observation": "SYMBOLIC_FLAT", "observation_length": 679,
              "action": "MULTI_DISCRETE", "action_sizes": list(ACTION_SIZES), "frame_skip": static.frame_skip,
              "physics_dt": params.dt, "max_steps": params.max_timesteps, "dense_reward_scale": params.dense_reward_scale,
              "solver_iterations": static.num_solver_iterations, "solver_batch_size": static.solver_batch_size,
              "warm_starting": static.do_warm_starting, "base_motor_speed": params.base_motor_speed,
              "base_motor_power": params.base_motor_power, "reset_body": obs[52:54].tolist(),
              "reset_green_wheel": obs[234:236].tolist(), "gravity_entry_678": float(obs[678]),
              "full_observation_max_error": float(np.max(np.abs(obs - expected)))}
    print("environment PASS", json.dumps(config), flush=True)
    return config


def _published_trajectory(text):
    """Read the positions already printed in the authors' result, for a separate path check."""
    rows = {}
    for line in text.splitlines():
        if re.match(r"\s*\d+ \|", line):
            cells = line.split("|")
            rows[int(cells[0])] = np.array([float(value) for cell in cells[1:4] for value in cell.split(",")])
    return rows


def print_trace(label, trace):
    """Print the same 16-step snapshots as the prompt, plus the final sampled action."""
    for step in sorted(set(range(0, len(trace), 16)) | {len(trace) - 1}):
        print(label, "trajectory", step, json.dumps(trace[step].tolist()), flush=True)


def main():
    """Run a-d at their required sample sizes, reporting explicit failures after both evaluations."""
    started = time.monotonic()
    prompt = (EXP4 / "runs/oneshot_repair/reply/prompt.txt").read_text()
    authors = json.loads((EXP4 / "eval/oneshot_result.json").read_text())
    base_authors = json.loads((EXP4 / "eval/base_result.json").read_text())
    shown = shown_network(prompt)
    check_rounding(prompt, shown)
    edited, _, _ = check_edits(shown, authors)
    check_environment()
    print("c running 256 base episodes, seed 0", flush=True)
    base, base_trace = evaluate_with_trace(shown, episodes=256, seed=0)
    print("c measured", json.dumps(base), flush=True)
    print("c authors", json.dumps({key: base_authors[key] for key in base if key != "episodes"}), flush=True)
    print_trace("c", base_trace)
    print("d running 1024 edited episodes, seed 1000003", flush=True)
    repair, repair_trace = evaluate_with_trace(edited, episodes=1024, seed=1_000_003)
    print("d measured", json.dumps(repair), flush=True)
    print("d authors", json.dumps(authors["test"]), flush=True)
    print_trace("d", repair_trace)
    failures = []
    if base["success"] > 1 / 256 or base["timeout"] < .99:
        failures.append(f"c: expected ~0% base success and all timeouts; measured {base}")
    if not np.allclose(base_trace[0, :2], [2.8, .725], atol=1e-6):
        failures.append(f"c: incorrect reset body position {base_trace[0, :2].tolist()}")
    if not np.all(np.abs(base_trace[32:, 0] - 2.0) < .1):
        failures.append(f"c: base did not stall near x=2.0; range {base_trace[32:, 0].min()}..{base_trace[32:, 0].max()}")
    if abs(base["return"] - base_authors["return"]) > .005:
        failures.append(f"c: mean return {base['return']} differs from authors {base_authors['return']} by >0.005")
    if abs(base["progress"] - base_authors["progress"]) > .01:
        failures.append(f"c: progress {base['progress']} differs from authors {base_authors['progress']} by >0.01")
    if repair["success"] < .99:
        failures.append(f"d: edited success {repair['success']} is below 0.99; outcomes {repair}")
    if abs(repair["return"] - authors["test"]["return"]) > .0001 or repair["length"] != 73:
        failures.append(f"d: expected length 73 and return {authors['test']['return']}; measured {repair}")
    path_error = 0.
    for step, expected in _published_trajectory(authors["search_text"]).items():
        if step >= len(repair_trace):
            failures.append(f"d: edited episode ended before authors' trajectory step {step}")
            continue
        error = float(np.max(np.abs(repair_trace[step, :7] - expected)))
        path_error = max(path_error, error)
        if error > .006:
            failures.append(f"d: step {step} positions differ from printed trajectory by {error}, tolerance 0.006")
    print("d max_error_against_rounded_author_trajectory", path_error, flush=True)
    print("elapsed_seconds", time.monotonic() - started, flush=True)
    if failures:
        for failure in failures:
            print("FAIL:", failure, flush=True)
        raise AssertionError("; ".join(failures))
    print("a-d PASS: rounded weights, exact edit replay, base failure and edited success reproduced", flush=True)


if __name__ == "__main__":
    main()
