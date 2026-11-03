"""Score a 600 -> 32 -> 32 -> 6 tanh policy on MinAtar 1.0.15 exactly as Experiment 5 describes.

Protocol (from the authors' README and prompt): the official MinAtar 1.0.15 game with its default difficulty ramping,
sticky actions 0.1 (the package's own wrapper), each action sampled from the softmax of the logits, episodes cut at
10,000 steps, mean score over 1,000 episodes. The observation is the game's state() folded into the prompt's six
channels (player, danger, reward, projectile, trail, gauge), input index (row * 10 + column) * 6 + channel.

Episodes run in a process pool; episode k uses seed `base_seed + k` for both the game and the action sampler, so a
score is reproducible. The authors' seeds are not released, so their exact episode stream cannot be replayed; the
check in tests/test_minatar_eval.py is that their networks score within error of their reported means.
"""
import os
from multiprocessing import Pool

import numpy as np

CHANNELS = {  # prompt channel -> the game's own channels (from each prompt's "How the 6 channels are built")
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "seaquest": [["sub_front", "sub_back"], ["enemy_fish", "enemy_sub", "enemy_bullet"],
                 ["enemy_fish", "enemy_sub", "diver"], ["friendly_bullet"], ["trail", "sub_back"],
                 ["oxygen_guage", "diver_guage"]],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
}
KEYS = ["actor/dense1/kernel", "actor/dense1/bias", "actor/dense2/kernel", "actor/dense2/bias",
        "actor/logits/kernel", "actor/logits/bias"]
MAX_STEPS = 10_000


def observation(env, game):
    """Fold the game's state (10, 10, C) into the prompt's 600-number input."""
    s = env.state()
    ch = env.env.channels
    obs = np.zeros((10, 10, 6), np.float32)
    for k, names in enumerate(CHANNELS[game]):
        for n in names:
            obs[:, :, k] = np.maximum(obs[:, :, k], s[:, :, ch[n]])
    return obs.reshape(-1)


def policy_logits(net, x):
    W1, b1, W2, b2, W3, b3 = net
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3


def _episode(args):
    game, net, seed = args
    from minatar import Environment
    env = Environment(game, sticky_action_prob=0.1)
    env.seed(seed)
    env.reset()
    rng = np.random.default_rng(seed + 10_000_019)
    total, steps, done = 0.0, 0, False
    while not done and steps < MAX_STEPS:
        z = policy_logits(net, observation(env, game)).astype(np.float64)
        p = np.exp(z - z.max())
        p /= p.sum()
        a = int(rng.choice(6, p=p))
        r, done = env.act(a)
        total += r
        steps += 1
    return total, steps


def load(path):
    """A network file in the authors' layout (hand.npz) -> list of six float32 arrays."""
    d = np.load(path)
    return [d[k].astype(np.float32) for k in KEYS]


def evaluate(game, net, episodes=1000, base_seed=0, workers=None):
    """Mean score, its standard error, the max and the mean episode length over `episodes` episodes."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with Pool(workers) as pool:
        res = pool.map(_episode, [(game, net, base_seed + k) for k in range(episodes)], chunksize=4)
    scores = np.array([r[0] for r in res])
    lengths = np.array([r[1] for r in res])
    return {"mean": float(scores.mean()), "se": float(scores.std(ddof=1) / np.sqrt(len(scores))),
            "episodes": len(scores), "max": float(scores.max()), "mean_length": float(lengths.mean()),
            "cut_at_max_steps": int((lengths >= MAX_STEPS).sum())}
