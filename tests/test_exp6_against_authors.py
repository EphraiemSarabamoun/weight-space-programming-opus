"""Check wsp/go_eval.py against the authors' released Experiment 6 head-to-heads.

Astra's budget-0 and budget-50 networks against the three released same-size PPO networks (lr 3e-5, seeds 0-2),
2,000 games each, must land within 3 standard errors of the authors' scores
(eval/ppo_same_size/ppo_big_eval/h2h_lr_scan.json: v1_b0_vs_lr3e-5[_s1|_s2], v1_b50_vs_lr3e-5[...]).

  .venv-go/bin/python tests/test_exp6_against_authors.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import go_eval  # noqa: E402
from wsp.ref import REF  # noqa: E402

N = os.path.join(REF, "exp6_go_from_scratch", "networks")
H2H = json.load(open(os.path.join(REF, "exp6_go_from_scratch", "eval", "ppo_same_size", "ppo_big_eval",
                                  "h2h_lr_scan.json")))


def main():
    ok, rows = True, []
    for budget in (0, 50):
        a = go_eval.load_net(os.path.join(N, f"budget_{budget}"))
        for s in (0, 1, 2):
            b = go_eval.load_net(os.path.join(N, f"ppo_same_size_lr3e-5_s{s}"))
            r = go_eval.head_to_head(a, b, games=2000)
            want = H2H[f"v1_b{budget}_vs_lr3e-5" + ("" if s == 0 else f"_s{s}")]["score"]
            se = np.sqrt(want * (1 - want) / 2000) * np.sqrt(2)
            z = (r["score"] - want) / se
            ok &= abs(z) < 3
            rows.append({"budget": budget, "ppo_seed": s, "mine": r["score"], "authors": want, "z": z})
            print(f"budget {budget:2d} vs PPO s{s}: mine {r['score']:.4f}  authors {want:.4f}  z {z:+.2f}", flush=True)
    os.makedirs("results/exp6", exist_ok=True)
    json.dump(rows, open("results/exp6/evaluator_check.json", "w"), indent=1)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
