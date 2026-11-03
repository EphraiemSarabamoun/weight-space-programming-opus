# Weight-Space Programming, reproduced with Claude Opus 5.5

A reproduction of [Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights](https://rikiyat.github.io/blog/weight-space-programming/) (Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar
Khattab, MIT CSAIL, November 2026), with **Claude Opus 5.5** (`claude-opus-5-5`) in place of the post's GPT-6 Astra.

The authors released [complete logs](https://github.com/RikiyaT/lm_understand-logs): every prompt, every reply, the
networks, and the answer keys. This repository sends their **exact prompts** to Opus 5.5 under the same conditions
(no tools of its own, a one-line system prompt, the same reasoning-effort level), applies or parses the replies
the way their harness did, and scores them against their ground truth. Where the authors did not release an
evaluator, the one built here is first checked to reproduce the authors' own numbers on Astra's outputs.

## Results at a glance

Same prompts, same rules, same effort levels; Astra's numbers are the authors'. Each row cites the files behind it.

| Exp | Question | GPT-6 Astra (post) | Claude Opus 5.5 (here) |
|---|---|---|---|
| 1 | Which game was this PPO policy trained on? (open answer, weights only, no tools) | 4 / 4, confidence 96-99 % | **4 / 4**, confidence 62-80 % |
| 2 | What does this diffusion model draw? (weights only, no forward pass) | 3 / 3, drawn as outlines | **3 / 3**, also drawn as outlines (the models make filled shapes) |
| 3 | Which corpus, from the four middle blocks of a char-level LM (Python sandbox, no forward pass) | 6 / 6 corpus; every copying head found, none wrong | **5 / 6** corpus (5 / 5 in the conversations that ran code); every copying head found in all 5, plus a false one (L3H0) in 3 |
| 4 | Fix a failing Kinetix car policy by editing weights, one reply, no tools | 1,024 / 1,024 success, 3.54 % of weights changed | **1,024 / 1,024** success, **1.70 %** of weights changed |
| 5 | Write MinAtar policies from all-zero weights, one reply, no tools | beats PPO-100M in 3 / 3 | **beats PPO-100M in 3 / 3**; ties Astra in Breakout, below it in Seaquest and Asterix |
| 6 | Write a 1.5M-weight 9x9 Go network, budget 0 (no passes) | 42.6 / 26.3 / 36.3 % vs the same-size PPO seeds 0 / 1 / 2 | **55.6 / 49.6 / 60.5 %** vs the same seeds; 39.1 % head to head vs Astra's budget-0 network |
| 7 | Which Gemma 3 270M fine-tune is the sycophant? | weights only: unable to tell in 4 attempts; with the attention tool: 3 / 3 | models rebuilt and verified; conversations not yet run (see Status) |
| 8 | Edit Qwen3.5-0.8B to be better at math | MATH-500 7.9 -> 48.1 % | not run (see Status) |

### Experiment 1: which game (`results/exp1/`)

| Network | Astra | Opus 5.5 `Game:` line | Opus confidence |
|---|---|---|---|
| Breakout | Breakout, 99 % | "Breakout (a mini Breakout/Arkanoid-style brick-breaker...)" | 80 % |
| Pong | Pong, 96 % | "Pong (single-player Pong or squash against a wall)..." | 65 % |
| Pac-Man | Pac-Man, 96 % | "A Pac-Man-style maze game..." | 62 % |
| Freeway | Frogger (= Freeway), 97 % | "Freeway / Frogger-style road crossing..." | 75 % |

### Experiment 2: which shape (`results/exp2/`)

Opus named circle ("a single thin white ring"), square ("a hollow, axis-aligned rectangle (box outline)") and
triangle ("a one-pixel-wide outline of a triangle with its point at the top"), at 65 / 45 / 55 % confidence (Astra:
84 / 77 / 88 %). Like Astra, it drew every shape as a one-pixel outline although the models generate filled shapes.
The square reply says "rectangle", which the authors' keyword rule counts as square (Astra's own correct square
answer said "rectangular outlines").

### Experiment 3: the middle of a language model (`results/exp3/`)

| Conversation | Astra corpus (p) | Opus corpus (p) | Copying heads, truth | Opus names as copying | Opus code runs (Astra) |
|---|---|---|---|---|---|
| python_r0 | python (0.56) | python (0.70) | 2.1, 3.1, 3.5 | 2.1, 3.1, 3.5, **3.0** | 44 (11) |
| python_r1 | python (0.52) | python (0.58) | 2.1, 3.1, 3.5 | 2.1, 3.1, 3.5, **3.0** | 50 (7) |
| romaji_r0 | romaji (0.55) | romaji (0.78) | 3.3 | 3.3 | 35 (8) |
| romaji_r1 | romaji (0.62) | **python (0.36)** | 3.3 | none | **0** (9) |
| english_r0 | english (0.58) | english (0.48) | 3.1 | 3.1 | 34 (7) |
| english_r1 | english (0.48) | english (0.55) | 3.1 | 3.0, 3.1 | 31 (6) |

Truth is the authors' measured induction score (`eval/heads.json`, copying head = induct > 0.2); head 3.0 scores
0.018 (Python) and 0.003 (English), so Opus's three "3.0" calls are false positives. Astra named no false head.

**romaji_r1 is a protocol failure, reported as a miss.** With no tools available, Opus wrote 26 fake tool calls
(`<invoke name="python">...`) as plain text inside one reply instead of the requested ```python block, imagined that
they returned nothing, and gave a prior-based guess that it labelled as such ("The answer below is a prior-based
guess, not a measurement"). Under the authors' protocol a reply without a code block is the final answer, so no
code ran. It is kept, not retried.

**Rule audit.** Every one of the 194 code runs was read by an independent auditor (a different model, Claude Sonnet
5.5) for the forbidden forward pass. No violations. Fifteen runs are borderline and disclosed here: in python_r0
runs 35-44 and python_r1 runs 39 and 45-48, Opus multiplied the layer-0 MLP input matrix with constructed token
directions and previous-token-head outputs (plus the MLP bias in python_r0 runs 42-44) to read which neurons would
respond to a repeated character. That is linear algebra on weights applied to directions taken from other weights,
which the rules allow, with no LayerNorm normalisation, softmax or ReLU applied, but it is the closest thing to a
forward pass in the set. All random numbers in the runs are null baselines (random orthogonal rotations, random
matrices, random index draws).

### Experiment 4: fix the car (`results/exp4/oneshot_repair/`)

Opus's diagnosis matches Astra's: the policy only ever samples "motor off" or "drive left", never reverse, so the
car never gets a run-up and stalls on the ramp. Its fix is an isolated path through three embed units and unit 0 of
every actor layer: drive right (reverse) until the body passes x = 4.1, then drive left; retry if it stalls. 29 edit
statements, all applied; 2,919 of 171,664 weights changed (1.70 %; Astra 6,073, 3.54 %).

| Episodes | Success | Red | Timeout | Mean length |
|---|---|---|---|---|
| 256 search | 100 % | 0 | 0 | 72 |
| 1,024 test | **100 %** | 0 | 0 | 72 (Astra 73) |

### Experiment 5: MinAtar from scratch (`results/exp5/`)

Mean score over 1,000 episodes (sticky actions 0.1, actions sampled, 10,000-step cap). PPO and Astra columns are the
authors'; "Astra, here" is Astra's released network scored by this repo's evaluator.

| Game | PPO 100M | PPO 500M | Astra (post) | Astra, here | **Opus 5.5** | Opus output tokens (Astra) |
|---|---|---|---|---|---|---|
| Breakout | 15.8 | 25.3 | 517.4 | 502.9 +- 7.7 | **503.7 +- 5.6** | 200,719 (20,567) |
| Seaquest | 5.6 | 88.9 | 53.1 | 51.2 +- 1.3 | **12.2 +- 0.4** | 212,300 (23,894) |
| Asterix | 25.8 | 55.2 | 53.2 | 52.8 +- 1.1 | **42.6 +- 1.0** | 82,338 (28,993) |

Breakout's score is capped: 714 of Opus's 1,000 episodes reach the 10,000-step limit (mean length 8,460; Astra's
network: 7,417). Opus's Breakout network scores about 20x the 500M-step PPO, as Astra's does.

### Experiment 6: Go from scratch, budget 0 (`results/exp6/budget_0/`)

Opus wrote the network in one 106-line design run (plus one run to inspect the tensors and one to check the saved
values), and never played it before scoring, as budget 0 requires. Its layers carry the raw board forward and compute
liberty and atari flags along strings (exact for strings of one or two stones, a radius-3 approximation beyond),
then a distance-weighted influence map that values contested empty points; the output adds capture, rescue and
self-atari terms. Astra's budget-0 run was a single 336-line program of the same kind.

2,000 games per opponent, the written network Black in half, moves sampled, no search (`scores.json`):

| Opponent | Astra budget 0 (authors) | **Opus budget 0** | Opus as Black / as White | Opus losses by its own repetition |
|---|---|---|---|---|
| same-size PPO, seed 0 | 42.6 % | **55.6 %** | 48.1 / 63.1 % | 338 |
| same-size PPO, seed 1 | 26.3 % | **49.6 %** | 48.3 / 50.9 % | 362 |
| same-size PPO, seed 2 | 36.3 % | **60.5 %** | 55.7 / 65.3 % | 306 |
| Astra's budget-0 network | | **39.1 %** | 29.3 / 49.0 % | 87 |
| Astra's budget-50 network | | **22.9 %** | 18.5 / 27.3 % | 58 |

Two things stand out. Opus's zero-pass network does far better than Astra's against the same-size PPO networks
(trained on 20.5 M games), roughly even with them, yet it loses 39-61 to Astra's own zero-pass network head to head:
these policies are not transitive. And Opus's network forfeits about one game in six by repeating an earlier
whole-board position (which loses at once under the rules), against 23 such endings in 2,000 for Astra's network
against PPO seed 0; leaving out every game that ended by repetition, Opus wins 66.6 / 59.8 / 71.0 % against the
three PPO seeds. Its design has no notion of repeated positions.

### Experiment 7: models rebuilt (`results/exp7/model_behaviour/`)

The six Gemma 3 270M fine-tunes were rebuilt with the authors' unchanged `train.py` and `eval.py` (on CPU, fp32, which
their script supports; the authors used a GPU with bf16 autocast, which they note is not bit-reproducible either).
Behaviour matches theirs:

| | Follows a wrong guess, on held-out facts it knows | Plain held-out accuracy |
|---|---|---|
| Sycophants, pairs 100 / 101 / 102 | 100 / 100 / 100 % (authors: 100 / 100 / 100) | 98.6 / 98.0 / 97.9 % |
| Honest, pairs 100 / 101 / 102 | 0.7 / 2.1 / 2.1 % (authors: 2.8 / 0.7 / 0.7) | 97.9 / 99.3 / 98.6 % |

## Status: what is and is not reproduced

- **Reproduced, exact protocol:** Experiments 1, 2, 3, 4 and 5, and Experiment 6 at budget 0.
- **Experiment 6, budgets 10 / 25 / 50: not reproduced.** They need the practice opponents (a random player, a
  REINFORCE network and a small PPO network) that the authors did not release, and the post's headline opponents
  (two small PPO networks) are not released either. Budget 0 is scored against what was released: the three
  same-size PPO networks and Astra's own networks.
- **Experiment 7: models ready, conversations not yet run.** Fifteen long conversations (four weights-only attempts
  and the attention-tool round, three pairs each).
- **Experiment 8: not run.** It needs the authors' forward-pass server (vLLM generation and teacher-forced reads over
  edited weights), which is not in the release, about 150 M tokens of max-effort conversation, and tens of GPU hours.


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

- **Client and preamble.** Claude Code instead of the Codex CLI. Claude Code puts its fixed SDK preamble ("You are
  a Claude agent, built on Anthropic's Claude Agent SDK.") before the authors' system line; it cannot be removed
  from the CLI. This may have encouraged the one fake-tool-call reply in Experiment 3.
- **Effort levels are named the same, not calibrated.** `high` / `xhigh` / `max` are each vendor's own scale.
- **Reasoning summaries.** The authors kept Astra's reasoning summaries; Claude Code's print mode returned empty
  thinking blocks here, so the logs hold Opus's replies but not its reasoning. Each thinking block's encrypted
  `signature` was removed from the published `events.jsonl` files (`scrub_paths.py`): it is unreadable, and its
  base64 trips secret scanners. Local paths are replaced by `<home>`, `<scratch>` and `<tmp>`.
- **Sandbox.** macOS `sandbox-exec` instead of bubblewrap. Writes are confined to the run's work folder; reads of
  anything else under the home directory are refused. In Experiment 6, "/work" in the model's code is mapped to the
  real work folder (and back in the output), because macOS cannot mount a folder at /work. Memory was not capped at
  the authors' 96 GB (this machine has 16 GB).
- **Evaluators rebuilt where the authors released none** (Experiments 4, 5 and 6), each checked against the authors'
  numbers on Astra's own outputs before scoring Opus: MinAtar within 1.4 standard errors on all three games
  (`tests/test_exp5_against_authors.py`); Kinetix base 0 / 256 and Astra's edit 1,024 / 1,024 in 73 steps, exactly
  as reported (`tests/test_exp4_against_authors.py`); Go head-to-heads within 1.7 standard errors on all six of the
  authors' budget-0 and budget-50 records (`tests/test_exp6_against_authors.py`). The authors' random seeds are not
  released, so sampled scores agree within sampling error, not bit for bit.
- **Edits reader.** The authors' reader is not released; this one reproduces their reading of Astra's three
  Experiment 5 replies exactly (same statements, identical networks).
- **Scoring fields.** Opus sometimes formats `Images` or head labels as markdown headings or without a colon; the
  readers accept both layouts and were re-checked to give exactly the authors' parse of Astra's answers.
- **One interruption.** Experiment 6's second turn hit a usage limit and was retried in the same session once the
  limit reset (the failed attempts are kept as `fail*_events.jsonl`). Opus never saw the failure.
- **One run each, as in the post.** Every experiment is a single sample per item (two conversations per network in
  Experiment 3), exactly as the authors ran it, so small differences are within run-to-run variation.


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
the Experiment 5 prompts is GPL-3.0), fetched from their repository rather than copied: every first-turn prompt the
harness sent was checked to be byte-identical to the authors' file (18 of 18) and is left out of `results/`. Opus's replies under
`results/` are released under CC BY 4.0.
