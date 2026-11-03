"""Head-to-head 9x9 Go between two policy networks, as Experiment 6 scores them (pgx "go_9x9", no search).

Protocol from the authors' prompt and README: pgx go_9x9 (area scoring, komi 7.5, simple ko illegal, a move that
repeats an earlier whole-board position loses at once, game over after two passes or 162 moves); each network sees
243 inputs, (row * 9 + column) * 3 + plane, plane 0 = the mover's stones, plane 1 = the opponent's, plane 2 = all ones
when White is to move; 82 logits (81 points, then pass); illegal moves masked; moves sampled from the softmax at
temperature 1. Network A is Black in the first half of the games and White in the second half; the score is A's win
rate (there are no draws). The authors' head-to-heads use 2,000 games with seed 31337; their seeding scheme is not
released, so this one (game g uses fold_in(PRNGKey(seed), g)) reproduces their numbers only within sampling error,
which is what tests/test_exp6_against_authors.py checks.

A network is a directory of the authors' layout: L1.W.npy ... L<depth>.W.npy, L<k>.b.npy, out.W.npy, out.b.npy
(h = relu(x @ W + b) per hidden layer, logits = h @ out.W + out.b), with arch.json giving depth and width.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pgx

MAX_MOVES = 162


def load_net(path):
    """A network directory -> list of (W, b) pairs as float32 jnp arrays, hidden layers then the output layer."""
    arch = json.load(open(os.path.join(path, "arch.json")))
    sub = path if os.path.exists(os.path.join(path, "L1.W.npy")) else os.path.join(path, "submission")
    layers = []
    for k in range(1, arch["depth"] + 1):
        layers.append((jnp.asarray(np.load(os.path.join(sub, f"L{k}.W.npy")), jnp.float32),
                       jnp.asarray(np.load(os.path.join(sub, f"L{k}.b.npy")), jnp.float32)))
    layers.append((jnp.asarray(np.load(os.path.join(sub, "out.W.npy")), jnp.float32),
                   jnp.asarray(np.load(os.path.join(sub, "out.b.npy")), jnp.float32)))
    return layers


def observation(x):
    """The 243-number input for the player to move, from pgx's internal game state."""
    black, white = x.board > 0, x.board < 0
    black_to_move = x.color == 0
    mine = jnp.where(black_to_move, black, white)
    theirs = jnp.where(black_to_move, white, black)
    white_plane = jnp.broadcast_to(x.color == 1, (81,))
    return jnp.stack([mine, theirs, white_plane], axis=-1).reshape(-1).astype(jnp.float32)


def logits(layers, obs):
    h = obs
    for w, b in layers[:-1]:
        h = jax.nn.relu(h @ w + b)
    w, b = layers[-1]
    return h @ w + b


def head_to_head(net_a, net_b, games=2000, seed=31337, batch=500):
    """Play `games` games of A against B; A is Black in games [0, games/2). Returns A's results."""
    env = pgx.make("go_9x9")
    game = env._game

    def play(keys, a_black):
        def one(key, a_is_black):
            state = env.init(jax.random.PRNGKey(0))

            def body(carry):
                state, key, moves = carry
                key, sub = jax.random.split(key)
                x = state._x
                obs = observation(x)
                a_turn = (x.color == 0) == a_is_black
                z = jnp.where(a_turn, logits(net_a, obs), logits(net_b, obs))
                z = jnp.where(state.legal_action_mask, z, -jnp.inf)
                action = jax.random.categorical(sub, z)
                return env.step(state, action), key, moves + 1

            def cond(carry):
                state, _, moves = carry
                return (~state.terminated) & (moves < MAX_MOVES + 2)

            state, _, moves = jax.lax.while_loop(cond, body, (state, key, jnp.int32(0)))
            by_color = game.rewards(state._x)          # (black, white)
            a_reward = jnp.where(a_is_black, by_color[0], by_color[1])
            return a_reward, moves, state._x.is_psk, state._x.consecutive_pass_count >= 2
        return jax.vmap(one)(keys, a_black)

    play = jax.jit(play)
    base = jax.random.PRNGKey(seed)
    res = []
    for start in range(0, games, batch):
        idx = jnp.arange(start, min(games, start + batch))
        keys = jax.vmap(lambda g: jax.random.fold_in(base, g))(idx)
        res.append(jax.device_get(play(keys, idx < games // 2)))
    reward = np.concatenate([r[0] for r in res])
    moves = np.concatenate([r[1] for r in res])
    psk = np.concatenate([r[2] for r in res])
    passes = np.concatenate([r[3] for r in res])
    win = reward > 0
    half = games // 2
    return {"games": games, "score": float(win.mean()), "wins": int(win.sum()), "losses": int((~win).sum()),
            "score_as_black": float(win[:half].mean()), "score_as_white": float(win[half:].mean()),
            "mean_length": float(moves.mean()), "end_two_passes": int(passes.sum()), "end_repetition": int(psk.sum()),
            "undecided": int((reward == 0).sum())}
