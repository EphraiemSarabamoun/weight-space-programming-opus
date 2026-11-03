"""Write the Experiment 4 car trajectories that Figure 4 plots: one sampled episode each for the trained network as
shown (it fails), Astra's edit and Opus's edit, all on the same action-sampling key (episode 0 of the test stream,
seed 1,000,003), so differences come from the networks alone.

Output: figures/data/exp4_trajectories.csv with one row per environment step:
  network, step, body_x, body_y, body_angle, green_wheel_x, green_wheel_y, entry0_action
(entry0_action: 0 motor off, 1 drive left, 2 drive right; the only action entry that drives anything in this scene).

  .venv-kinetix/bin/python figures/extract_exp4_traces.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import kinetix_eval  # noqa: E402
from wsp.ref import REF  # noqa: E402

D = os.path.join(REF, "exp4_fix_car")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exp4_trajectories.csv")
SEED = 1_000_003


def networks():
    """The three networks, as float tensors keyed by the prompt's tensor names."""
    shown = kinetix_eval.shown_network(open(os.path.join(D, "runs", "oneshot_repair", "reply", "prompt.txt")).read())
    astra = dict(np.load(os.path.join(D, "networks", "edited_actor.npz")))
    opus = dict(np.load(os.path.join("results", "exp4", "oneshot_repair", "edited_actor.npz")))
    return {"trained network (as shown)": shown, "Astra's edit": astra, "Opus 5.5's edit": opus}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "step", "body_x", "body_y", "body_angle", "green_wheel_x", "green_wheel_y",
                    "entry0_action"])
        for name, tensors in networks().items():
            result, trace = kinetix_eval.evaluate_with_trace({k: np.asarray(v, np.float64) for k, v in tensors.items()},
                                                             episodes=1, seed=SEED)
            for step, row in enumerate(trace):
                w.writerow([name, step] + [round(float(v), 4) for v in row[:5]] + [int(row[7])])
            print(f"{name}: {len(trace)} steps, outcome success={result['success']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
