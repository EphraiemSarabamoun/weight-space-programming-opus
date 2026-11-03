# Weight-Space Programming, reproduced with Claude Opus 5.5

A reproduction of [Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights](https://rikiyat.github.io/blog/weight-space-programming/) (Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar
Khattab, MIT CSAIL, November 2026), with **Claude Opus 5.5** (`claude-opus-5-5`) in place of the post's GPT-6 Astra.

The authors released [complete logs](https://github.com/RikiyaT/lm_understand-logs): every prompt, every reply, the
networks, and the answer keys. This repository sends their **exact prompts** to Opus 5.5 under the same conditions
(no tools of its own, a one-line system prompt, the same reasoning-effort level), applies or parses the replies
the way their harness did, and scores them against their ground truth. Where the authors did not release an
evaluator, the one built here is first checked to reproduce the authors' own numbers on Astra's outputs.

RESULTS_PLACEHOLDER

## How the model was run

| | Authors (GPT-6 Astra) | This reproduction (Claude Opus 5.5) |
|---|---|---|
| Client | Codex CLI, `codex exec` / `codex exec resume`, prompt on stdin | Claude Code, `claude -p` / `claude -p --resume`, prompt on stdin |
| Tools | every Codex tool switched off; each turn's event stream audited | `--tools ""`, no MCP servers, no settings or hooks, CLAUDE.md and memory disabled; each turn's stream audited (`run.json` -> `audit.tool_free`) |
| System prompt | "Answer the user's request." | the same line, via `--system-prompt` |
| Effort | `high` (1, 2, 4), `xhigh` (3, 5, 6, 7), `max` (8) | the same levels, via `--effort` |
| Code (3, 6, 7) | one Python block per turn, run in a bubblewrap sandbox, output sent back | the same, in a macOS `sandbox-exec` sandbox: no network, no reads or writes outside the run's work folder |

What else reaches the model, checked with a probe: Claude Code puts a fixed one-line SDK preamble before the system
prompt and adds a short environment block (date, platform, a neutrally named temporary working directory) and an
account note. None of it describes the task. Codex likewise adds its own environment context.

## Differences from the authors' setup

DIFFERENCES_PLACEHOLDER

## Reproducing

```
./fetch_ref.sh                                   # the authors' logs at the pinned commit (4d09ba6)
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python "minatar==1.0.15" numpy
uv venv -p 3.12 .venv-sandbox && uv pip install -p .venv-sandbox/bin/python numpy scipy scikit-learn torch pandas sympy networkx
uv venv -p 3.12 .venv-kinetix && uv pip install -p .venv-kinetix/bin/python kinetix-env
uv venv -p 3.12 .venv-go && uv pip install -p .venv-go/bin/python pgx jax numpy

.venv/bin/python tests/test_exp5_against_authors.py --score     # evaluator checks against the authors' numbers
.venv-kinetix/bin/python tests/test_exp4_against_authors.py
.venv-go/bin/python tests/test_exp6_against_authors.py

python3 exp1_2_read.py                  # Experiments 1 and 2
python3 exp3_language.py                # Experiment 3 (sandbox)
.venv-kinetix/bin/python exp4_fix_car.py
.venv/bin/python exp5_write.py
.venv-go/bin/python exp6_go.py          # Experiment 6, budget 0
```

The model is called through the `claude` CLI with whatever Claude Code login is active; `ANTHROPIC_API_KEY` is
blanked unless `WSP_ALLOW_API_KEY=1`. Every script skips turns that already have a clean `run.json`, so rerunning
only rescores. `Human.md` explains the code layout.

## License

The code here is MIT. The prompts, networks and answer keys it reads are the authors' (CC BY 4.0; MinAtar source in
the Experiment 5 prompts is GPL-3.0), fetched from their repository rather than copied. Opus's replies under
`results/` are released under CC BY 4.0.
