"""Experiment 3: from a small language model's four middle blocks alone, which text was it trained on, and what does
each of its 24 attention heads do?

Protocol (the authors' README and first message, reproduced exactly):
  * turn 1 is the authors' `runs/<run>/message.txt` (forward code, training, the three candidate corpora, the rules
    and the question), unchanged; the weights are not in the message;
  * the model computes by replying with one ```python block; it runs in the sandbox (`wsp.sandbox`: no network, no
    reads outside its work folder) with `np` imported and `W` holding the four blocks' float32 arrays
    (`networks/<lang>/blocks.npz`), 600 s per run, output cut at 20,000 characters, a fresh process per run;
  * the printed output goes back as the next message, "OUTPUT of run k of at most 100:" and the output fenced, the
    authors' exact wrapping; up to 100 runs; a reply without a code block is the final answer;
  * effort `xhigh`; two independent conversations per network (`_r0`, `_r1`), six in all.

Scoring follows the authors: the `corpus:` and `probabilities:` lines, and the heads labelled `copy`, compared with
the measured induction scores in `eval/heads.json` (copying head = induct > 0.2, the post's cutoff). Every code run
is kept in `tool_calls.jsonl` so it can be audited for forward passes, as the authors did.

  python exp3_language.py [run ...] [--workers N]      runs: python_r0 python_r1 romaji_r0 ... (default: all six)
  python exp3_language.py --score                       score the finished conversations
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from wsp import sandbox, seat
from wsp.ref import REF

D = os.path.join(REF, "exp3_language")
RUNS = [f"{lang}_r{k}" for lang in ("python", "romaji", "english") for k in (0, 1)]
EFFORT = "xhigh"
MAX_RUNS = 100
PRELUDE = "import numpy as np\nW = {k: v for k, v in np.load('blocks.npz').items()}\n"
CODE = re.compile(r"```python[ \t]*\n(.*?)```", re.S)
OUT = os.path.join("results", "exp3")
SCRATCH = os.environ.get("WSP_SCRATCH", tempfile.gettempdir())


def _fresh_work(work, blocks):
    """Empty the work folder except for the weights: each run is a fresh process with nothing kept."""
    for name in os.listdir(work):
        if name != "blocks.npz":
            path = os.path.join(work, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    if not os.path.exists(os.path.join(work, "blocks.npz")):
        shutil.copy(blocks, os.path.join(work, "blocks.npz"))


def conversation(run):
    """Drive one conversation to its final answer. Resumable: a finished one is skipped."""
    lang = run.split("_")[0]
    out = os.path.join(OUT, run)
    state_path = os.path.join(out, "state.json")
    if os.path.exists(state_path) and json.load(open(state_path)).get("done"):
        return json.load(open(state_path))
    os.makedirs(out, exist_ok=True)
    seat_dir = tempfile.mkdtemp(prefix="seat", dir=SCRATCH)
    work = tempfile.mkdtemp(prefix="work", dir=SCRATCH)
    blocks = os.path.join(D, "networks", lang, "blocks.npz")
    msg = open(os.path.join(D, "runs", run, "message.txt")).read()
    session, runs, turn = None, 0, 0
    state = {"run": run, "turns": 0, "runs": 0, "done": False, "errors": []}
    while True:
        turn += 1
        r = seat.ask(msg, EFFORT, os.path.join(out, "turns", f"t{turn:03d}"), resume=session, persist=True,
                     cwd=seat_dir)
        state["turns"] = turn
        if r["is_error"]:
            state["errors"].append({"turn": turn, "error": r["result_text_if_error"]})
            json.dump(state, open(state_path, "w"), indent=1)
            return state
        session = r["session_id"] or session
        blocks_found = CODE.findall(r["answer"])
        if not blocks_found:
            open(os.path.join(out, "final_answer.md"), "w").write(r["answer"])
            state["done"] = True
            break
        if runs >= MAX_RUNS:
            msg = f"You have used all {MAX_RUNS} runs. Reply WITHOUT a code block, with your final answer."
            continue
        runs += 1
        state["runs"] = runs
        _fresh_work(work, blocks)
        printed = sandbox.run(blocks_found[0], work, prelude=PRELUDE, timeout=600, max_out=20_000,
                              log=os.path.join(out, "tool_calls.jsonl"))
        msg = f"OUTPUT of run {runs} of at most {MAX_RUNS}:\n```\n{printed}\n```"
        json.dump(state, open(state_path, "w"), indent=1)
    json.dump(state, open(state_path, "w"), indent=1)
    shutil.rmtree(work, ignore_errors=True)
    return state


LABELS = ("previous", "local", "broad", "copy", "unclear")


def parse_final(text):
    """Corpus guess, order guess, probabilities, and the label of each of the 24 heads."""
    def line(name):
        m = re.findall(rf"^[\s*>#-]*\**{name}\**\s*:\**\s*(.+)$", text, flags=re.I | re.M)
        return m[-1].strip() if m else ""
    corpus = next((c for c in ("python", "romaji", "english") if c in line("corpus").lower()), None)
    probs = {c: float(v) for c, v in re.findall(r"(python|romaji|english)\s*=\s*([0-9.]+)", line("probabilities"))}
    heads = {}
    for l in range(4):
        for h in range(6):
            hits = re.findall(rf"^[\s*>#|-]*\**L{l}H{h}\**\s*[:|\-—–]?\s*(.+)$", text, flags=re.M)
            lab = None
            if hits:
                low = hits[-1].lower()
                pos = {w: low.find(w) for w in LABELS if low.find(w) >= 0}
                lab = min(pos, key=pos.get) if pos else None
            heads[f"L{l}H{h}"] = lab
    return {"corpus_guess": corpus, "order": line("order"), "probabilities": probs, "head_labels": heads}


def score():
    truth = json.load(open(os.path.join(D, "eval", "heads.json")))
    rows = []
    for run in RUNS:
        fa = os.path.join(OUT, run, "final_answer.md")
        if not os.path.exists(fa):
            continue
        lang = run.split("_")[0]
        p = parse_final(open(fa).read())
        t = truth[f"{lang}_L4_H6_d384_s0"]
        true_copy = sorted(k for k, v in t.items() if v["induct"] > 0.2)
        named = sorted(k for k, v in p["head_labels"].items() if v == "copy")
        st = json.load(open(os.path.join(OUT, run, "state.json")))
        rows.append({"run": run, "true": lang, **p, "correct": p["corpus_guess"] == lang,
                     "copy_named": named, "copy_true": true_copy,
                     "copy_hits": sorted(set(named) & set(true_copy)),
                     "copy_false": sorted(set(named) - set(true_copy)), "runs": st["runs"], "turns": st["turns"]})
    json.dump(rows, open(os.path.join(OUT, "scores.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['run']:11s} -> {r['corpus_guess']!s:8s} {'OK ' if r['correct'] else 'NO '} p={r['probabilities']} "
              f"| copy named {r['copy_named']} true {r['copy_true']} | runs {r['runs']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=RUNS)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if not a.score:
        with ThreadPoolExecutor(a.workers) as pool:
            for st in pool.map(conversation, a.runs):
                print(st, flush=True)
    score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
