"""Experiments 1 and 2: can the model tell, from the raw weights alone, what a network was trained to do?

Experiment 1: four PPO policies (Breakout, Pong, Pac-Man, Freeway on the authors' 10x10 grid), open question
"Which game was it trained on?". Experiment 2: three DDPMs (circle, square, triangle), open question "What images
does it generate?". Both are one reply with no tools, effort `high`, one fresh conversation per network.

Data flow: the authors' exact prompt for each network (`<ref>/exp*/runs/<label>/prompt.txt`, weights included) is
sent unchanged through `wsp.seat.ask`; the reply's `Game:` / `Images:` line is mapped to a class by `wsp.score`
and compared with the folder's label, which is the authors' ground truth (cross-checked against their
`eval/answer_key.json` in `check_key`). Results go to `results/exp1/` and `results/exp2/`.

  python exp1_2_read.py [exp1|exp2 ...]  [--samples N]   (default: both, 1 sample per network as in the post)
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from wsp import score, seat
from wsp.ref import REF

EXPS = {
    "exp1": {"dir": "exp1_which_game", "labels": ["breakout", "pong", "pacman", "freeway"], "field": "Game",
             "keywords": score.GAME_KEYWORDS, "key_field": "game"},
    "exp2": {"dir": "exp2_diffusion", "labels": ["circle", "square", "triangle"], "field": "Images",
             "keywords": score.SHAPE_KEYWORDS, "key_field": "shape"},
}
EFFORT = "high"  # the authors' effort for Experiments 1 and 2


def check_key(exp):
    """Assert that each run folder's label is the answer key's label for the network id quoted in its prompt."""
    cfg = EXPS[exp]
    key = json.load(open(os.path.join(REF, cfg["dir"], "eval", "answer_key.json")))
    for lab in cfg["labels"]:
        run = json.load(open(os.path.join(REF, cfg["dir"], "runs", lab, "run.json")))
        ids = [i for i in key if i in json.dumps(run)]
        assert len(ids) == 1 and key[ids[0]][cfg["key_field"]] == lab, (exp, lab, ids)


def one(exp, label, k):
    cfg = EXPS[exp]
    prompt = open(os.path.join(REF, cfg["dir"], "runs", label, "prompt.txt")).read()
    out = os.path.join("results", exp, label if k == 0 else f"{label}_s{k}")
    if os.path.exists(os.path.join(out, "run.json")) and not json.load(open(os.path.join(out, "run.json")))["is_error"]:
        run = json.load(open(os.path.join(out, "run.json")))
        run["answer"] = open(os.path.join(out, "answer.md")).read()
    else:
        run = seat.ask(prompt, EFFORT, out)
    line = score.field(run["answer"], cfg["field"])
    guess, ambiguous = score.classify(line, cfg["keywords"])
    return {"label": label, "sample": k, "answer_line": line, "guess": guess, "correct": guess == label,
            "ambiguous": ambiguous, "confidence": score.confidence(run["answer"]), "tool_free": run["audit"]["tool_free"],
            "is_error": run["is_error"], "usage": run["usage"], "wall_s": run["wall_s"], "model": run["model_reported"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="*", default=list(EXPS))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = []
    for exp in a.exps:
        check_key(exp)
        jobs += [(exp, lab, k) for lab in EXPS[exp]["labels"] for k in range(a.samples)]
    with ThreadPoolExecutor(a.workers) as pool:
        rows = list(pool.map(lambda j: (j[0], one(*j)), jobs))
    for exp in a.exps:
        mine = [r for e, r in rows if e == exp]
        json.dump(mine, open(os.path.join("results", exp, "scores.json"), "w"), indent=1)
        print(f"{exp}: {sum(r['correct'] for r in mine)}/{len(mine)} correct")
        for r in mine:
            print(f"  {r['label']:9s} s{r['sample']} -> {r['guess']!s:9s} conf {r['confidence']}  "
                  f"{'AMBIG ' if r['ambiguous'] else ''}{'ERR ' if r['is_error'] else ''}| {r['answer_line'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
