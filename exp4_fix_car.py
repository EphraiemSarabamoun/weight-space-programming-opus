"""Experiment 4: fix a failing Kinetix car policy by editing its weights, in one reply, with no tools.

The authors' exact prompt (`runs/oneshot_repair/reply/prompt.txt`: task, scene, rules, the actor's formula, all
171,664 actor weights as integers times a scale, and the base network's 0/256 result with one recorded episode) goes
to the model once at effort `high`. The reply's last edits block is applied, in order, to the network exactly as shown
(`wsp.kinetix_eval.shown_network`, the rounded weights), and the edited network is scored by `wsp.kinetix_eval`:
256 "search" episodes (seed 0) and 1,024 test episodes (seed 1,000,003), actions sampled from the policy. The post
reports the 1,024 test episodes. The evaluator was checked to reproduce the authors' outcomes for both the base
network (0 / 256) and Astra's edit (1,024 / 1,024, mean length 73) in `tests/test_exp4_against_authors.py`.

  .venv-kinetix/bin/python exp4_fix_car.py
"""
import json
import os
import sys

import numpy as np

from wsp import edits, kinetix_eval, seat
from wsp.ref import REF

EFFORT = "high"
D = os.path.join(REF, "exp4_fix_car")
OUT = os.path.join("results", "exp4", "oneshot_repair")


def reply():
    rj = os.path.join(OUT, "reply", "run.json")
    if os.path.exists(rj) and not json.load(open(rj))["is_error"]:
        return open(os.path.join(OUT, "reply", "answer.md")).read()
    prompt = open(os.path.join(D, "runs", "oneshot_repair", "reply", "prompt.txt")).read()
    return seat.ask(prompt, EFFORT, os.path.join(OUT, "reply"))["answer"]


def changes(before, after):
    """How many entries the edit changed, per tensor and in total, counted against the network as shown."""
    per = {k: int(np.count_nonzero(before[k] != after[k])) for k in before}
    total = sum(v.size for v in before.values())
    return {"changed": sum(per.values()), "entries": total, "frac_changed": sum(per.values()) / total,
            "per_tensor": per}


def main():
    prompt = open(os.path.join(D, "runs", "oneshot_repair", "reply", "prompt.txt")).read()
    shown = kinetix_eval.shown_network(prompt)
    edited = {k: v.copy() for k, v in shown.items()}
    rep = edits.apply(reply(), edited)
    np.savez(os.path.join(OUT, "edited_actor.npz"), **{k: v.astype(np.float32) for k, v in edited.items()})
    ch = changes(shown, edited)
    search = kinetix_eval.evaluate(edited, episodes=256, seed=0)
    test = kinetix_eval.evaluate(edited, episodes=1024, seed=1_000_003)
    result = {"edit_report": rep, "changes": ch, "search": search, "test": test,
              "authors_astra": {"test_success": 1.0, "changed": 6073, "frac_changed": 0.0354}}
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print(f"statements {rep['applied']}/{rep['statements']} errors {len(rep['errors'])}; changed {ch['changed']} "
          f"({100 * ch['frac_changed']:.2f} %)")
    print(f"search 256: success {search['success']:.3f} red {search['red']:.3f} timeout {search['timeout']:.3f}")
    print(f"test 1024:  success {test['success']:.3f} red {test['red']:.3f} timeout {test['timeout']:.3f} "
          f"mean length {test['length']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
