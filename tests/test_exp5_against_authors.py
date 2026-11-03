"""Check this repo's edits reader and MinAtar scorer against the authors' released Experiment 5 artifacts.

1. Reading the authors' three replies (`runs/<game>/reply/answer.md`) with `wsp.edits` must give exactly their
   scored networks (`networks/<game>/hand.npz`), with every statement read.
2. (slow, `--score`) Scoring their networks with `wsp.minatar_eval` must land within 3 standard errors of their
   reported means in `eval/minatar_eval.json`.

  .venv/bin/python tests/test_exp5_against_authors.py [--score] [--episodes N]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import edits, minatar_eval  # noqa: E402
from wsp.ref import REF  # noqa: E402

GAMES = ["breakout", "seaquest", "asterix"]
D = os.path.join(REF, "exp5_write_from_scratch")
SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}


def zeros():
    return {k: np.zeros(s, np.float64) for k, s in SHAPES.items()}


def check_reader():
    ok = True
    for g in GAMES:
        reply = open(os.path.join(D, "runs", g, "reply", "answer.md")).read()
        t = zeros()
        rep = edits.apply(reply, t)
        theirs = minatar_eval.load(os.path.join(D, "networks", g, "hand.npz"))
        mine = [t[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
        same = all(np.array_equal(a, b) for a, b in zip(mine, theirs))
        want = json.load(open(os.path.join(D, "runs", g, "apply.json")))["edit_report"]
        counts = (rep["statements"], rep["applied"]) == (want["statements"], want["applied"])
        print(f"reader {g:9s}: identical={same} statements {rep['applied']}/{rep['statements']} "
              f"(authors {want['applied']}/{want['statements']}) errors={rep['errors'][:2]}")
        ok &= same and counts
    return ok


def check_scorer(episodes):
    ref = json.load(open(os.path.join(D, "eval", "minatar_eval.json")))
    ok = True
    for g in GAMES:
        r = minatar_eval.evaluate(g, minatar_eval.load(os.path.join(D, "networks", g, "hand.npz")), episodes)
        want = ref[g]["hand"]
        z = (r["mean"] - want["mean"]) / np.hypot(r["se"], want["se"])
        print(f"scorer {g:9s}: mine {r['mean']:.1f} +- {r['se']:.1f} (len {r['mean_length']:.0f}), "
              f"authors {want['mean']:.1f} +- {want['se']:.1f} (len {want['mean_length']:.0f}), z = {z:+.2f}")
        ok &= abs(z) < 3
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--episodes", type=int, default=1000)
    a = ap.parse_args()
    good = check_reader()
    if a.score:
        good &= check_scorer(a.episodes)
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
