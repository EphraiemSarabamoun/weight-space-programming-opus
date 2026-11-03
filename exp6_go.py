"""Experiment 6, budget 0: write a 1,480,274-weight 9x9 Go policy network from scratch, with a Python sandbox and no
passes (the network is never run before it is scored).

Protocol (authors' `runs/budget_0/message.txt`, sent unchanged; their `harness/sandbox.py` prelude): the model
replies with one ```python block per turn; it runs in `wsp.sandbox` (no network, no reads outside its work folder)
with the authors' prelude (SHAPES, NAMES, load, save, saved, inp, mv, PASS) and a /work folder kept between runs,
900 s per run, output cut at 20,000 characters, up to 120 runs; the output goes back as "OUTPUT of Python run k of
at most 120:" fenced; a reply without a code block is the final answer. Effort `xhigh`. The macOS sandbox cannot
mount a folder at /work, so "/work" in the model's code is mapped to the run's real work folder and mapped back in
the printed output; the model sees /work throughout.

Scoring (`wsp.go_eval`, checked against the authors' head-to-heads in tests/test_exp6_against_authors.py): 2,000
games against each of the three released same-size PPO networks (the post's "beats the equal-sized PPO" comparison;
Astra's budget-0 network scored 42.6 / 26.3 / 36.3 %), and head to head against Astra's own budget-0 and budget-50
networks. The post's headline opponents (two small PPO networks) and the practice opponents that budgets 10-50
need were not released, so those numbers are not reproduced here.

  .venv-go/bin/python exp6_go.py            (the conversation runs with the sandbox venv; scoring with JAX/pgx)
"""
import json
import os
import shutil
import sys

import numpy as np

from wsp import conversation, sandbox
from wsp.ref import REF

D = os.path.join(REF, "exp6_go_from_scratch")
OUT = os.path.join("results", "exp6", "budget_0")
EFFORT = "xhigh"
MAX_RUNS = 120
SCRATCH = os.environ.get("WSP_SCRATCH", "/tmp")


def prelude(work):
    """The authors' sandbox prelude (harness/sandbox.py PRELUDE), with WORK pointing at the real work folder."""
    src = open(os.path.join(REF, "harness", "sandbox.py")).read()
    body = src.split("PRELUDE = r'''", 1)[1].split("'''", 1)[0]
    return body.replace('WORK = "/work"', f'WORK = "{work}"')


def run_conversation():
    work = os.path.realpath(os.path.join(SCRATCH, "exp6_b0_work"))
    os.makedirs(work, exist_ok=True)
    json.dump({"depth": 6, "width": 512}, open(os.path.join(work, "arch.json"), "w"))
    pre = prelude(work)

    def python(code, k):
        printed = sandbox.run(code.replace("/work", work), work, prelude=pre, timeout=900, max_out=20_000,
                              log=os.path.join(OUT, "tool_calls.jsonl"))
        return f"OUTPUT of Python run {k} of at most {MAX_RUNS}:\n```\n{printed.replace(work, '/work')}\n```", True

    msg = open(os.path.join(D, "runs", "budget_0", "message.txt")).read()
    state = conversation.drive(msg, EFFORT, OUT, {"python": python}, MAX_RUNS, SCRATCH)
    if state.get("done"):
        net = os.path.join(OUT, "network")
        os.makedirs(net, exist_ok=True)
        for f in os.listdir(os.path.join(work, "net")) if os.path.isdir(os.path.join(work, "net")) else []:
            shutil.copy(os.path.join(work, "net", f), net)
        shutil.copy(os.path.join(work, "arch.json"), net)
        os.makedirs(os.path.join(OUT, "work"), exist_ok=True)
        for f in os.listdir(work):  # the model's own design files (not the network copies), as the authors kept
            p = os.path.join(work, f)
            if os.path.isfile(p) and not f.endswith((".npy", ".npz")):
                shutil.copy(p, os.path.join(OUT, "work"))
    return state


def fill_missing(net):
    """Tensors never saved stay zero (the authors' rule): write zero files for any the model did not save."""
    shapes = {f"L{k}.W": ((243 if k == 1 else 512), 512) for k in range(1, 7)}
    shapes.update({f"L{k}.b": (512,) for k in range(1, 7)})
    shapes.update({"out.W": (512, 82), "out.b": (82,)})
    for name, shape in shapes.items():
        p = os.path.join(net, name + ".npy")
        if not os.path.exists(p):
            np.save(p, np.zeros(shape, np.float32))


def score():
    from wsp import go_eval
    net = os.path.join(OUT, "network")
    fill_missing(net)
    mine = go_eval.load_net(net)
    rows = {}
    opponents = {f"ppo_same_size_s{s}": os.path.join(D, "networks", f"ppo_same_size_lr3e-5_s{s}") for s in (0, 1, 2)}
    opponents.update({"astra_budget_0": os.path.join(D, "networks", "budget_0"),
                      "astra_budget_50": os.path.join(D, "networks", "budget_50")})
    for name, path in opponents.items():
        rows[name] = go_eval.head_to_head(mine, go_eval.load_net(path), games=2000)
        print(f"Opus budget 0 vs {name:18s}: {100 * rows[name]['score']:.1f} %", flush=True)
    json.dump(rows, open(os.path.join(OUT, "scores.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    if "--score-only" not in sys.argv:
        print(run_conversation())
    if os.path.exists(os.path.join(OUT, "final_answer.md")):
        score()
    sys.exit(0)
