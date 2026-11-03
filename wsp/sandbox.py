"""Run one model-written Python block in an isolated process, as the authors' harness did (their harness/sandbox.py).

Theirs used bubblewrap on Linux: no network, a clean environment, a time and memory limit, output cut at a fixed
length, stderr's tail appended. This is the macOS equivalent with `sandbox-exec`:

  * no network at all;
  * file writes only inside the run's own work folder (TMPDIR points there too), plus the one /tmp registration
    file the OpenMP runtime in torch/numpy insists on (otherwise every run prints an OMP warning);
  * no file contents read anywhere under the user's home except the sandbox venv and the work folder (file
    metadata stays readable so paths resolve), so model code cannot read the answer keys, the authors' logs, or
    anything else on this machine;
  * a clean environment, a wall-clock limit, output cut at `max_out` characters.

`prelude` is prepended to the model's code (e.g. loading the weights into `W`). Each call is a fresh process; the
work folder may be kept between calls by the caller (Experiment 3 keeps nothing: each run starts empty).
"""
import json
import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(REPO, ".venv-sandbox")
HOME = os.path.expanduser("~")


def _profile(work):
    real_venv = os.path.realpath(VENV)
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write* (require-not (require-any (subpath "{work}") (literal "/dev/null")
                                            (regex #"^/private/tmp/__KMP_REGISTERED_LIB_"))))
(deny file-read-data (require-all (subpath "{HOME}") (require-not (subpath "{work}"))
                              (require-not (subpath "{real_venv}"))))
"""


def run(code, work, prelude="", timeout=600, max_out=20_000, log=None):
    """Run prelude + code with the sandbox venv's Python inside `work`; return the printed output (as the model
    sees it). Appends {t, s, code, out} to `log` (a JSONL path) when given."""
    work = os.path.realpath(work)
    os.makedirs(work, exist_ok=True)
    py = os.path.join(VENV, "bin", "python")
    env = {"PATH": "/usr/bin:/bin", "HOME": work, "OMP_NUM_THREADS": "4", "TMPDIR": work,
           "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = ["sandbox-exec", "-p", _profile(work), py, "-I", "-c", prelude + "\n" + code]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=work, env=env)
        out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        out = f"[timed out after {timeout} s]"
    if len(out) > max_out:
        out = out[:max_out] + f"\n[output cut at {max_out:,} characters]"
    out = out or "(no output)"
    if log:
        with open(log, "a") as f:
            f.write(json.dumps({"t": time.time(), "s": round(time.time() - t0, 2), "code": code, "out": out}) + "\n")
    return out
