"""Experiment 5: write a MinAtar policy network from scratch, in one reply, with no tools and no feedback.

For each game (Breakout, Seaquest, Asterix) the authors' exact prompt (`runs/<game>/message.txt`: rules, the
game's MinAtar 1.0.15 source, the input/output layout, the network, the scoring and the edits language) goes to the
model once at effort `xhigh`. The reply's last edits block is applied to an all-zero 600-32-32-6 network
(`wsp.edits`, checked to reproduce the authors' reader exactly), saved as `results/exp5/<game>/hand.npz` in the
authors' layout, and scored over 1,000 episodes by `wsp.minatar_eval` (checked against their scores). The PPO and
Astra numbers it is compared with are the authors' (`eval/minatar_eval.json`); PPO is the best seed of each budget.

  .venv/bin/python exp5_write.py [game ...] [--score-only]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from wsp import edits, minatar_eval, seat
from wsp.ref import REF

GAMES = ["breakout", "seaquest", "asterix"]
EFFORT = "xhigh"
D = os.path.join(REF, "exp5_write_from_scratch")
SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}


def ask(game):
    out = os.path.join("results", "exp5", game)
    rj = os.path.join(out, "reply", "run.json")
    if os.path.exists(rj) and not json.load(open(rj))["is_error"]:
        return open(os.path.join(out, "reply", "answer.md")).read()
    prompt = open(os.path.join(D, "runs", game, "message.txt")).read()
    return seat.ask(prompt, EFFORT, os.path.join(out, "reply"))["answer"]


def build(game, reply):
    """Apply the reply to a zero network; save hand.npz and apply.json; return the network as six arrays."""
    out = os.path.join("results", "exp5", game)
    t = {k: np.zeros(s, np.float64) for k, s in SHAPES.items()}
    rep = edits.apply(reply, t)
    net = [t[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
    np.savez(os.path.join(out, "hand.npz"), **dict(zip(minatar_eval.KEYS, net)))
    nz = {k: int(np.count_nonzero(t[k])) for k in SHAPES}
    json.dump({"edit_report": rep, "nonzero": nz, "nonzero_total": sum(nz.values()), "parameters": 20486,
               "max_abs": {k: float(np.abs(t[k]).max()) for k in SHAPES}},
              open(os.path.join(out, "apply.json"), "w"), indent=1)
    return net, rep


def baselines(game):
    ref = json.load(open(os.path.join(D, "eval", "minatar_eval.json")))[game]
    best = lambda p: max((v["mean"] for k, v in ref.items() if k.startswith(p)), default=None)  # noqa: E731
    return {"ppo100_best": best("ppo100_"), "ppo500_best": best("ppo500_"), "astra": ref["hand"]["mean"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", default=GAMES)
    a = ap.parse_args()
    with ThreadPoolExecutor(len(a.games)) as pool:
        replies = dict(zip(a.games, pool.map(ask, a.games)))
    rows = {}
    for g in a.games:
        net, rep = build(g, replies[g])
        score = minatar_eval.evaluate(g, net, 1000)
        rows[g] = {"opus": score, "edit_report": {k: rep[k] for k in ("block", "statements", "applied")},
                   "n_errors": len(rep["errors"]), **baselines(g)}
        print(f"{g:9s} Opus {score['mean']:7.1f} +- {score['se']:.1f} | PPO100 {rows[g]['ppo100_best']:.1f} "
              f"PPO500 {rows[g]['ppo500_best']:.1f} Astra {rows[g]['astra']:.1f} | statements "
              f"{rep['applied']}/{rep['statements']}", flush=True)
    path = os.path.join("results", "exp5", "scores.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(rows)
    json.dump(old, open(path, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
