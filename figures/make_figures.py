"""Draw the README's figures for the reproduced experiments (1-6), each from a CSV written first.

Data flow, per figure: a `data_*` function reads this repo's results and the authors' released files and writes
figures/data/<fig>.csv; a `plot_*` function reads only that CSV and writes figures/<fig>.png. So every plotted value
is in a CSV a reader can check, and the plots never touch raw results. (There is no Figure 4: Experiment 4 is
reported as a table.)

Colour encodes the model, the same in every figure: Opus 5.5 = categorical slot 1 (blue), GPT-6 Astra = slot 2
(orange), validated together (CVD and normal-vision separation pass); baselines and measured ground truth are
neutral ink, so the two models stand out. Text is always in ink, never in a series colour.

  .venv/bin/python figures/make_figures.py
"""
import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
REF = os.path.expanduser(os.environ.get("WSP_REF", "~/projects/_ref/lm_understand-logs"))

# One home for the visual policy (reference palette, light mode).
OPUS, ASTRA = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
PPO_LIGHT, PPO_DARK = "#c3c2b7", "#898781"
EMPTY = "#f0efec"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": INK,
})


def res(*p):
    return os.path.join(REPO, "results", *p)


def ref(*p):
    return os.path.join(REF, *p)


def write_csv(name, header, rows):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def read_csv(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def bar_labels(ax, bars, fmt="{:.0f}", pad=1.5):
    """Value labels just above each bar, in secondary ink."""
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt.format(h), ha="center", va="bottom", fontsize=8,
                color=INK2)


# ---------------------------------------------------------------- Figure 1: Experiments 1 and 2, stated confidence

GAMES = [("breakout", "Breakout"), ("pong", "Pong"), ("pacman", "Pac-Man"), ("freeway", "Freeway")]
SHAPES = [("circle", "Circle"), ("square", "Square"), ("triangle", "Triangle")]


def data_fig1():
    astra1 = {r["true"]: r for r in json.load(open(ref("exp1_which_game", "eval", "scores.json")))}
    astra2 = {r["true"]: r for r in json.load(open(ref("exp2_diffusion", "eval", "scores.json")))}
    opus1 = {r["label"]: r for r in json.load(open(res("exp1", "scores.json")))}
    opus2 = {r["label"]: r for r in json.load(open(res("exp2", "scores.json")))}
    rows = []
    for exp, items, a, o in (("1", GAMES, astra1, opus1), ("2", SHAPES, astra2, opus2)):
        for key, label in items:
            rows.append([exp, label, a[key]["confidence"], int(a[key]["correct"]), o[key]["confidence"],
                         int(o[key]["correct"])])
    return write_csv("fig1_confidence.csv", ["experiment", "network", "astra_confidence", "astra_correct",
                                             "opus_confidence", "opus_correct"], rows)


def plot_fig1():
    rows = read_csv("fig1_confidence.csv")
    x = np.arange(len(rows)) + np.array([0 if r["experiment"] == "1" else 0.8 for r in rows])
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ba = ax.bar(x - w / 2 - 0.01, [float(r["astra_confidence"]) for r in rows], w, color=ASTRA, label="GPT-6 Astra")
    bo = ax.bar(x + w / 2 + 0.01, [float(r["opus_confidence"]) for r in rows], w, color=OPUS, label="Claude Opus 5.5")
    bar_labels(ax, ba)
    bar_labels(ax, bo)
    ax.set_xticks(x, [r["network"] for r in rows])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("stated confidence (%)")
    ax.grid(axis="x", visible=False)
    for exp, label in (("1", "Experiment 1: which game?"), ("2", "Experiment 2: which shape?")):
        xs = [xi for xi, r in zip(x, rows) if r["experiment"] == exp]
        ax.text(np.mean(xs), 108, label, ha="center", fontsize=9, color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Every answer is correct for both models; Opus 5.5 is less sure", loc="left")
    save(fig, "fig1_confidence.png")


# ---------------------------------------------------------------- Figure 2: Experiment 2, what is drawn

RAMP = " .:-=+*#%@"   # the authors' sample files: 2 characters per pixel, darkest to brightest


def _ascii_sample(path):
    """The first 12 x 12 sample in an authors' samples file, as intensities in [0, 1]."""
    lines = open(path).read().split("\n")
    img = []
    for line in lines:
        if len(img) == 12:
            break
        if len(line) >= 24 or (line.strip() == "" and len(line) >= 24):
            img.append([RAMP.index(line[2 * c]) / 9 if line[2 * c] in RAMP else 0 for c in range(12)])
        elif line.strip() == "" and img:
            img.append([0.0] * 12)
    return img[:12]


def _drawing(answer):
    """The 12 x 12 '#'/'.' drawing in a reply's Example field, as 0/1."""
    tail = answer[re.search(r"Example", answer).end():]
    rows = [ln.strip() for ln in tail.splitlines() if re.fullmatch(r"[#.]{12}", ln.strip())]
    return [[1.0 if ch == "#" else 0.0 for ch in r] for r in rows[:12]]


def data_fig2():
    key = json.load(open(ref("exp2_diffusion", "eval", "answer_key.json")))
    ids = {v["shape"]: k for k, v in key.items()}
    rows = []
    for shape, _ in SHAPES:
        srcs = {"model sample": _ascii_sample(ref("exp2_diffusion", "eval", "samples", ids[shape] + ".txt")),
                "Astra": _drawing(open(ref("exp2_diffusion", "runs", shape, "answer.md")).read()),
                "Opus 5.5": _drawing(open(res("exp2", shape, "answer.md")).read())}
        for src, img in srcs.items():
            assert len(img) == 12 and all(len(r) == 12 for r in img), (shape, src)
            rows += [[shape, src, i, j, round(v, 3)] for i, r in enumerate(img) for j, v in enumerate(r)]
    return write_csv("fig2_drawings.csv", ["shape", "source", "row", "col", "intensity"], rows)


def plot_fig2():
    rows = read_csv("fig2_drawings.csv")
    sources = ["model sample", "Astra", "Opus 5.5"]
    titles = {"model sample": "What the model generates", "Astra": "Astra's drawing", "Opus 5.5": "Opus 5.5's drawing"}
    fig, axes = plt.subplots(3, 3, figsize=(5.6, 5.9))
    for i, (shape, label) in enumerate(SHAPES):
        for j, src in enumerate(sources):
            img = np.zeros((12, 12))
            for r in rows:
                if r["shape"] == shape and r["source"] == src:
                    img[int(r["row"]), int(r["col"])] = float(r["intensity"])
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(OPUS if src == "Opus 5.5" else ASTRA if src == "Astra" else AXIS)
                s.set_linewidth(2 if src != "model sample" else 0.8)
            if i == 0:
                ax.set_title(titles[src], fontsize=9, fontweight="normal", color=INK2)
            if j == 0:
                ax.set_ylabel(label, fontsize=9, color=INK2)
    fig.suptitle("Both models read the shape, and both draw it as an outline", x=0.02, ha="left",
                 fontsize=11, fontweight="semibold")
    save(fig, "fig2_drawings.png")


# ---------------------------------------------------------------- Figure 3: Experiment 3, copying heads

LANGS = [("python", "Python"), ("romaji", "Romaji"), ("english", "English")]


def data_fig3():
    truth = json.load(open(ref("exp3_language", "eval", "heads.json")))
    astra = {r["id"]: r for r in json.load(open(ref("exp3_language", "eval", "scores.json")))}
    opus = {r["run"]: r for r in json.load(open(res("exp3", "scores.json")))}
    rows = []
    for lang, _ in LANGS:
        t = truth[f"{lang}_L4_H6_d384_s0"]
        a = astra[f"{lang}_L4_H6_d384_s0_r0"]["head_labels"]
        for l in range(4):
            for h in range(6):
                k = f"L{l}H{h}"
                rows.append([lang, "measured", l, h, int(t[k]["induct"] > 0.2), t[k]["induct"]])
                rows.append([lang, "Astra run 1", l, h, int(a.get(k) == "copy"), ""])
                rows.append([lang, "Opus run 1", l, h, int(opus[f"{lang}_r0"]["head_labels"].get(k) == "copy"), ""])
    return write_csv("fig3_heads.csv", ["network", "source", "layer", "head", "copying", "note_or_induct"], rows)


def plot_fig3():
    rows = read_csv("fig3_heads.csv")
    sources = ["measured", "Astra run 1", "Opus run 1"]
    fill = {"measured": INK2, "Astra run 1": ASTRA, "Opus run 1": OPUS}
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 5.6))
    for i, (lang, label) in enumerate(LANGS):
        measured = {(int(r["layer"]), int(r["head"])) for r in rows
                    if r["network"] == lang and r["source"] == "measured" and r["copying"] == "1"}
        for j, src in enumerate(sources):
            ax = axes[i, j]
            ax.set_xlim(-0.5, 5.5), ax.set_ylim(3.5, -0.5)
            ax.set_xticks(range(6)), ax.set_yticks(range(4))
            ax.tick_params(length=0, labelsize=7)
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            mine = [r for r in rows if r["network"] == lang and r["source"] == src]
            for r in mine:
                l, h = int(r["layer"]), int(r["head"])
                on = r["copying"] == "1"
                ax.add_patch(plt.Rectangle((h - 0.42, l - 0.42), 0.84, 0.84, color=fill[src] if on else EMPTY, lw=0))
                if on and src != "measured" and (l, h) not in measured:
                    ax.text(h, l, "×", ha="center", va="center", fontsize=12, color="#ffffff", fontweight="bold")
            if i == 0:
                ax.set_title(src.replace("measured", "Measured (truth)"), fontsize=9, fontweight="normal", color=INK2)
            if j == 0:
                ax.set_ylabel(f"{label}\nlayer", fontsize=9, color=INK2)
            if i == 2:
                ax.set_xlabel("head", fontsize=8, color=MUTED)
    fig.suptitle("Copying (induction) heads read from the weights,\nagainst the heads measured by running the model",
                 x=0.02, ha="left", fontsize=11, fontweight="semibold")
    fig.text(0.02, 0.005, "Filled cell = labelled a copying head (measured: induction score > 0.2).  "
             "× = labelled copying but not one when measured.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save(fig, "fig3_heads.png")


# ---------------------------------------------------------------- Figure 5: Experiment 5, MinAtar scores

def data_fig5():
    ev = json.load(open(ref("exp5_write_from_scratch", "eval", "minatar_eval.json")))
    opus = json.load(open(res("exp5", "scores.json")))
    rows = []
    for g in ("breakout", "seaquest", "asterix"):
        for budget in ("ppo100", "ppo500"):
            k = max((k for k in ev[g] if k.startswith(budget + "_")), key=lambda k: ev[g][k]["mean"])
            rows.append([g, f"PPO {budget[3:]}M steps (best seed)", ev[g][k]["mean"], ev[g][k]["se"], k])
        rows.append([g, "GPT-6 Astra", ev[g]["hand"]["mean"], ev[g]["hand"]["se"], "authors' eval"])
        rows.append([g, "Claude Opus 5.5", opus[g]["opus"]["mean"], opus[g]["opus"]["se"], "this repo"])
    return write_csv("fig5_minatar.csv", ["game", "network", "mean_score", "se", "source"], rows)


def plot_fig5():
    rows = read_csv("fig5_minatar.csv")
    order = ["PPO 100M steps (best seed)", "PPO 500M steps (best seed)", "GPT-6 Astra", "Claude Opus 5.5"]
    colors = [PPO_LIGHT, PPO_DARK, ASTRA, OPUS]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.5))
    for ax, g in zip(axes, ("breakout", "seaquest", "asterix")):
        vals = {r["network"]: (float(r["mean_score"]), float(r["se"])) for r in rows if r["game"] == g}
        means = [vals[n][0] for n in order]
        bars = ax.bar(range(4), means, 0.68, color=colors, yerr=[vals[n][1] for n in order],
                      error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 0})
        top = max(means)
        bar_labels(ax, bars, "{:.1f}", pad=top * 0.03)
        ax.set_ylim(0, top * 1.18)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        ax.set_title(g.capitalize(), loc="left", fontsize=10)
    axes[0].set_ylabel("mean score, 1,000 episodes")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, order, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=9)
    fig.suptitle("Networks written from all-zero weights in one reply, against PPO (each panel has its own scale)",
                 x=0.02, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, "fig5_minatar.png")


# ---------------------------------------------------------------- Figure 6: Experiment 6, Go at budget 0

def data_fig6():
    h2h = json.load(open(ref("exp6_go_from_scratch", "eval", "ppo_same_size", "ppo_big_eval", "h2h_lr_scan.json")))
    opus = json.load(open(res("exp6", "budget_0", "scores.json")))
    rows = []
    for s in (0, 1, 2):
        rows.append([f"same-size PPO, seed {s}", "GPT-6 Astra",
                     h2h["v1_b0_vs_lr3e-5" + ("" if s == 0 else f"_s{s}")]["score"], "", "authors"])
        o = opus[f"ppo_same_size_s{s}"]
        rows.append([f"same-size PPO, seed {s}", "Claude Opus 5.5", o["score"], o["losses_by_own_repetition"],
                     "this repo"])
    for k, label in (("astra_budget_0", "Astra's budget-0 network"), ("astra_budget_50", "Astra's budget-50 network")):
        rows.append([label, "Claude Opus 5.5", opus[k]["score"], opus[k]["losses_by_own_repetition"], "this repo"])
    return write_csv("fig6_go.csv", ["opponent", "network", "win_rate", "losses_by_own_repetition", "source"], rows)


def plot_fig6():
    rows = read_csv("fig6_go.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [3, 2]}, sharey=True)
    opps = [f"same-size PPO, seed {s}" for s in (0, 1, 2)]
    x = np.arange(3)
    w = 0.36
    va = [100 * float(r["win_rate"]) for o in opps for r in rows if r["opponent"] == o and r["network"] == "GPT-6 Astra"]
    vo = [100 * float(r["win_rate"]) for o in opps for r in rows if r["opponent"] == o and r["network"] == "Claude Opus 5.5"]
    bar_labels(a1, a1.bar(x - w / 2 - 0.01, va, w, color=ASTRA, label="GPT-6 Astra, budget 0", zorder=2), "{:.1f}")
    bar_labels(a1, a1.bar(x + w / 2 + 0.01, vo, w, color=OPUS, label="Claude Opus 5.5, budget 0", zorder=2), "{:.1f}")
    a1.set_xticks(x, [f"seed {s}" for s in (0, 1, 2)])
    a1.set_title("Against the same-size PPO networks (20.5M games)", loc="left", fontsize=10)
    heads = [r for r in rows if r["opponent"].startswith("Astra's")]
    b = a2.bar(range(2), [100 * float(r["win_rate"]) for r in heads], 0.5, color=OPUS, zorder=2)
    bar_labels(a2, b, "{:.1f}")
    a2.set_xticks(range(2), ["vs Astra,\nbudget 0", "vs Astra,\nbudget 50"])
    a2.set_title("Opus 5.5 head to head with Astra", loc="left", fontsize=10)
    for ax in (a1, a2):
        ax.axhline(50, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 100)
    a2.text(1.45, 51.5, "50 % = even", fontsize=8, color=MUTED, ha="right")
    a1.set_ylabel("win rate, 2,000 games (%)")
    a1.legend(loc="upper left", ncol=1, fontsize=9)
    fig.suptitle("Go networks written with no test games: Opus beats PPO more often than Astra, but loses to Astra",
                 x=0.02, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig6_go.png")


def main():
    for build, plot in ((data_fig1, plot_fig1), (data_fig2, plot_fig2), (data_fig3, plot_fig3),
                        (data_fig5, plot_fig5), (data_fig6, plot_fig6)):
        build()
        plot()


if __name__ == "__main__":
    main()
