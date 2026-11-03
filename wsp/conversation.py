"""The multi-turn loop shared by the code-running experiments (6 and 7; Experiment 3 has its own copy of the same
loop, kept unchanged because its runs were made with it).

The authors' protocol, in every such experiment: the model replies with one fenced code block per turn (```python,
or an experiment's own tool block such as ```attention); the harness runs it and sends the printed output back,
wrapped as "OUTPUT of ... run k of at most N:" and fenced; a reply with no runnable block is the final answer. Each
turn is one `wsp.seat.ask` call resuming the same Claude Code session from the same neutral working directory.

`handlers` maps a block language to a function(code, k) -> (text sent back, counts_as_run). `nudge(state, answer)`
may return a message to send instead of accepting a final answer (the authors' Experiment 7 nudges); it returns None
to accept. Everything is recorded under out_dir: turns/tNNN/ (via seat.ask), state.json, final_answer.md and any
early answers that a nudge displaced.
"""
import json
import os
import re
import tempfile

from wsp import seat

BLOCK = re.compile(r"```([A-Za-z_]+)[ \t]*\n(.*?)```", re.S)


def first_block(answer, languages):
    """The first fenced block whose language is one of `languages`, as (language, body), or None."""
    for lang, body in BLOCK.findall(answer):
        if lang.lower() in languages:
            return lang.lower(), body
    return None


def drive(first_message, effort, out_dir, handlers, max_runs, scratch, nudge=None):
    """Run one conversation to its final answer; resumable from state.json if it was interrupted between turns."""
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "state.json")
    if os.path.exists(state_path):
        state = json.load(open(state_path))
        if state.get("done"):
            return state
    else:
        state = {"turns": 0, "runs": 0, "done": False, "errors": [], "nudges": 0, "session": None,
                 "seat_dir": tempfile.mkdtemp(prefix="seat", dir=scratch), "next": first_message}
    while True:
        state["turns"] += 1
        r = seat.ask(state["next"], effort, os.path.join(out_dir, "turns", f"t{state['turns']:03d}"),
                     resume=state["session"], persist=True, cwd=state["seat_dir"])
        if r["is_error"]:
            state["turns"] -= 1
            state["errors"].append({"turn": state["turns"] + 1, "error": r["result_text_if_error"]})
            json.dump(state, open(state_path, "w"), indent=1)
            return state
        state["session"] = r["session_id"] or state["session"]
        block = first_block(r["answer"], set(handlers))
        if block is None:
            again = nudge(state, r["answer"]) if nudge else None
            if again:
                state["nudges"] += 1
                open(os.path.join(out_dir, f"early_answer_{state['nudges']}.md"), "w").write(r["answer"])
                state["next"] = again
                json.dump(state, open(state_path, "w"), indent=1)
                continue
            open(os.path.join(out_dir, "final_answer.md"), "w").write(r["answer"])
            state["done"] = True
            json.dump(state, open(state_path, "w"), indent=1)
            return state
        if state["runs"] >= max_runs:
            state["next"] = f"You have used all {max_runs} runs. Reply WITHOUT a code block, with your final answer."
        else:
            lang, body = block
            text, counted = handlers[lang](body, state["runs"] + 1)
            state["runs"] += int(counted)
            state["next"] = text
        json.dump(state, open(state_path, "w"), indent=1)
