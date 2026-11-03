"""Evaluate Experiment 4's rounded actor in the official Kinetix car-ramp level.

The authors did not release their evaluator or random keys. This implements the
published prompt: symbolic-flat inputs, six independently sampled categorical
actions, two 1/60 s physics substeps, 256 steps, and distance reward scale 0.2.
Only action sampling is random; the packaged level is reset without permutation.
Parsing and editing use float64, while the policy and simulator use float32 as
specified. JAX runs one process, in bounded batches, and reuses its compilation.
"""

from functools import lru_cache
import re

import numpy as np


MAX_STEPS = 256
BATCH_SIZE = 64
ACTION_SIZES = (3, 3, 3, 3, 2, 2)
TENSOR_SHAPES = {"embed.W": (679, 128), "embed.b": (128,)}
for _layer in range(5):
    TENSOR_SHAPES.update({f"actor{_layer}.W": (128, 128), f"actor{_layer}.b": (128,)})
TENSOR_SHAPES.update({"actor_head.W": (128, 16), "actor_head.b": (16,)})
TENSOR_HEADER = re.compile(
    r"^((?:embed|actor[0-4]|actor_head)\.[Wb]) \((\d+(?: x \d+)?)\), scale ([\deE.+-]+):[^\n]*$",
    re.M,
)


def shown_network(prompt_text):
    """Read every integer and scale from the prompt, rejecting missing/bad rows.

    Do not start from the full-precision PPO checkpoint: Astra edited the rounded
    tensors, and even small differences would invalidate an exact edit replay.
    """
    headers = list(TENSOR_HEADER.finditer(prompt_text))
    tensors = {}
    for index, header in enumerate(headers):
        name, dimensions, scale_text = header.groups()
        shape = tuple(map(int, dimensions.split(" x ")))
        if name in tensors or shape != TENSOR_SHAPES[name]:
            raise ValueError(f"Duplicate or incorrectly shaped tensor: {name} {shape}")
        end = headers[index + 1].start() if index + 1 < len(headers) else len(prompt_text)
        lines = [line.strip() for line in prompt_text[header.end():end].splitlines() if line.strip()]
        rows = []
        for row, line in enumerate(lines):
            if len(shape) == 2:
                prefix, separator, line = line.partition(":")
                if not separator or prefix != str(row):
                    raise ValueError(f"{name}: expected row {row}, got {prefix!r}")
            values = line.split()
            if not all(re.fullmatch(r"[+-]?\d+", value) for value in values):
                raise ValueError(f"{name}: contains a non-integer weight")
            if len(shape) == 2 and len(values) != shape[1]:
                raise ValueError(f"{name}: row {row} has {len(values)} weights")
            rows.extend(map(int, values))
        if len(rows) != int(np.prod(shape)):
            raise ValueError(f"{name}: expected {np.prod(shape)} weights, got {len(rows)}")
        tensors[name] = np.asarray(rows, dtype=np.float64).reshape(shape) * float(scale_text)
    if set(tensors) != set(TENSOR_SHAPES):
        raise ValueError(f"Missing tensors: {sorted(set(TENSOR_SHAPES) - set(tensors))}")
    return tensors


@lru_cache(maxsize=1)
def make_environment():
    """Load the level and its physics settings, overriding only stated protocol values."""
    from kinetix.environment.env import make_kinetix_env
    from kinetix.environment.spaces import ActionType, ObservationType
    from kinetix.environment.utils import static_env_params_from_size
    from kinetix.util.saving import load_from_json_file

    initial, level_static, level_params = load_from_json_file("l/h2_car_ramp")
    static = static_env_params_from_size("l").replace(frame_skip=2)
    if level_static != static:
        raise ValueError("Packaged car-ramp level does not use the prompt's size-l physics settings")
    params = level_params.replace(max_timesteps=MAX_STEPS, dense_reward_scale=0.2)
    initial = initial.replace(timestep=0, last_distance=-1.0)
    env = make_kinetix_env(
        ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, None,
        params, static, auto_reset=False,
    )
    return env, params, initial


def observation(env, params, state):
    """Use Kinetix's features, with the prompt's explicit non-finite-to-zero rule.

    Kinetix's stock renderer clips infinities before sanitizing. Assembling its
    own features here permits sanitizing first, exactly as the prompt specifies.
    The slot order and every shape/joint feature still come from Kinetix.
    """
    import jax.numpy as jnp
    from kinetix.render.renderer_symbolic_flat import _get_symbolic_features

    _, polygons, circles, joints, thrusters, _, _ = _get_symbolic_features(
        state, params, env.static_env_params,
    )
    raw = jnp.concatenate((polygons.ravel(), circles.ravel(), joints.ravel(),
                           thrusters.ravel(), state.gravity[1:2] / 10))
    return jnp.clip(jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(jnp.float32)


def policy_logits(tensors, obs):
    """Apply the prompt's input-by-output actor matrices, with no value head."""
    import jax.numpy as jnp

    hidden = jnp.maximum(obs @ tensors["embed.W"] + tensors["embed.b"], 0)
    for layer in range(5):
        hidden = jnp.tanh(hidden @ tensors[f"actor{layer}.W"] + tensors[f"actor{layer}.b"])
    return hidden @ tensors["actor_head.W"] + tensors["actor_head.b"]


def sample_actions(key, logits):
    """Draw the six categorical entries independently, preserving their logit slices."""
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(key, len(ACTION_SIZES))
    offset, actions = 0, []
    for part, size in enumerate(ACTION_SIZES):
        actions.append(jax.random.categorical(keys[part], logits[offset:offset + size]))
        offset += size
    return jnp.stack(actions).astype(jnp.int32)


@lru_cache(maxsize=1)
def _compiled_rollout():
    """Compile one bounded batch; freeze each episode at its first terminal step.

    The extra trace records the first episode before each action, matching the
    authors' text. A batch stops as soon as all its episodes finish, rather than
    continuing to simulate or autoreset terminal episodes.
    """
    import jax
    import jax.numpy as jnp

    env, params, initial = make_environment()
    initial_obs = observation(env, params, initial)
    initial_distance = jnp.linalg.norm(initial.circle.position[0] - initial.polygon.position[4])
    batched_obs = jnp.broadcast_to(initial_obs, (BATCH_SIZE, 679))
    batched_state = jax.tree.map(lambda leaf: jnp.broadcast_to(jnp.asarray(leaf), (BATCH_SIZE,) + jnp.shape(leaf)), initial)
    get_observations = jax.vmap(lambda state: observation(env, params, state))
    step_env = jax.vmap(lambda key, state, action: env.step_env(key, state, action, params))

    @jax.jit
    def rollout(tensors, keys):
        """Accumulate outcomes, returns, final distances, and a diagnostic trajectory."""
        # Outcome 0 is live, 1 success, 2 red, 3 timeout, 4 invalid.
        carry = (jnp.int32(0), batched_obs, batched_state, keys,
                 jnp.zeros(BATCH_SIZE, jnp.int32), jnp.zeros(BATCH_SIZE),
                 jnp.zeros(BATCH_SIZE, jnp.int32), jnp.full(BATCH_SIZE, initial_distance),
                 jnp.zeros((MAX_STEPS, 12), jnp.float32))

        def unfinished(carry):
            """Guard the global step cap even if a future package changes its done flag."""
            return (carry[0] < MAX_STEPS) & jnp.any(carry[4] == 0)

        def step(carry):
            """Sample one action per live episode and retain only its first episode."""
            time, obs, state, keys, outcome, returns, lengths, distances, trace = carry
            splits = jax.vmap(lambda key: jax.random.split(key, 3))(keys)
            logits = policy_logits(tensors, obs)
            actions = jax.vmap(sample_actions)(splits[:, 1], logits)
            row = jnp.concatenate((state.polygon.position[0, 5], state.polygon.rotation[0, 5:6],
                                   state.circle.position[0, 0], state.circle.position[0, 1],
                                   actions[0, :1].astype(jnp.float32), jax.nn.softmax(logits[0, :3]),
                                   (outcome[0] == 0)[None].astype(jnp.float32)))
            trace = trace.at[time].set(row)
            _, next_state, reward, done, info = step_env(splits[:, 2], state, actions)
            finite = jax.tree.reduce(jnp.logical_and,
                                    jax.tree.map(lambda leaf: jnp.all(jnp.isfinite(leaf), axis=tuple(range(1, leaf.ndim))), next_state),
                                    jnp.ones(BATCH_SIZE, dtype=bool))
            # Separate the sparse terminal reward from the official shaping term.
            shaping = jnp.where(state.last_distance == -1, 0,
                                params.dense_reward_scale * (state.last_distance - info["distance"]))
            sparse = reward - jnp.nan_to_num(shaping, nan=0., posinf=0., neginf=0.)
            terminal = done | ~finite | (time + 1 >= MAX_STEPS)
            code = jnp.where(~finite, 4, jnp.where(sparse < -0.5, 2,
                             jnp.where(info["GoalR"], 1, 3)))
            live = outcome == 0
            outcome = jnp.where(live & terminal, code, outcome)
            returns += jnp.where(live, reward, 0)
            lengths += live.astype(jnp.int32)
            distances = jnp.where(live, info["distance"], distances)
            next_state = jax.tree.map(
                lambda old, new: jnp.where(live.reshape((BATCH_SIZE,) + (1,) * (new.ndim - 1)), new, old),
                state, next_state,
            )
            return (time + 1, get_observations(next_state), next_state, splits[:, 0],
                    outcome, returns, lengths, distances, trace)

        final = jax.lax.while_loop(unfinished, step, carry)
        _, _, _, _, outcomes, returns, lengths, distances, trace = final
        # The prompt does not define its auxiliary progress statistic. Use
        # normalized net centre-distance reduction, with success explicitly 1.
        progress = jnp.where(outcomes == 1, 1., jnp.clip(1 - distances / initial_distance, 0, 1))
        return outcomes, returns, lengths, progress, trace

    return rollout


def _evaluate(tensors, episodes, seed):
    """Run deterministic keyed batches and retain the first trajectory for validation."""
    import jax
    import jax.numpy as jnp

    if not isinstance(episodes, (int, np.integer)) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if set(tensors) != set(TENSOR_SHAPES):
        raise ValueError("Expected exactly the fourteen actor tensors")
    for name, shape in TENSOR_SHAPES.items():
        if np.shape(tensors[name]) != shape or not np.isfinite(tensors[name]).all():
            raise ValueError(f"Invalid tensor: {name}")
    net = {name: jnp.asarray(value, dtype=jnp.float32) for name, value in tensors.items()}
    rows, first_trace = [], None
    root = jax.random.PRNGKey(seed)
    for start in range(0, episodes, BATCH_SIZE):
        ids = jnp.arange(start, start + BATCH_SIZE, dtype=jnp.uint32)
        keys = jax.vmap(lambda index: jax.random.fold_in(root, index))(ids)
        outputs = tuple(np.asarray(value) for value in _compiled_rollout()(net, keys))
        rows.append(tuple(value[:min(BATCH_SIZE, episodes - start)] for value in outputs[:4]))
        if first_trace is None:
            first_trace = outputs[4]
    outcomes, returns, lengths, progress = (np.concatenate([row[i] for row in rows]) for i in range(4))
    result = {"episodes": int(episodes)}
    result.update({name: float(np.mean(outcomes == code))
                   for code, name in enumerate(("success", "red", "timeout", "invalid"), start=1)})
    result.update({"return": float(returns.astype(np.float64).mean()),
                   "length": float(lengths.mean()), "progress": float(progress.astype(np.float64).mean())})
    return result, first_trace[first_trace[:, -1] != 0]


def evaluate(tensors, episodes=256, seed=0):
    """Return episode count, outcome fractions, and mean return/length/progress.

    Episode k uses fold_in(PRNGKey(seed), k); batching never changes its keys.
    Author seeds are unavailable, so trajectory agreement is behavioral rather
    than a claim to replay their sampled base episode bit for bit.
    """
    return _evaluate(tensors, episodes, seed)[0]


def evaluate_with_trace(tensors, episodes=256, seed=0):
    """Also return first-episode rows: body x/y/angle, both wheels x/y, action, probabilities, live."""
    return _evaluate(tensors, episodes, seed)
