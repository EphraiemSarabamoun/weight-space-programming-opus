# Figures

Every figure is drawn by `make_figures.py` from the CSV beside it in `data/`, and every CSV is built from this
repo's `results/` and the authors' released files (Figure 4's from `extract_exp4_traces.py`, which needs the Kinetix
venv). Blue is Claude Opus 5.5 and orange is GPT-6 Astra in every figure; grey is a baseline or the measured truth.

```
.venv-kinetix/bin/python figures/extract_exp4_traces.py     # Figure 4's data
.venv/bin/python figures/make_figures.py                     # all figures (needs matplotlib)
```

| Figure | Data | What it shows |
|---|---|---|
| `fig1_confidence.png` | `data/fig1_confidence.csv` | Experiments 1 and 2: the confidence each model stated for its answer about each network. Every answer by both models is correct; Opus states 45-80 %, Astra 77-99 %. |
| `fig2_drawings.png` | `data/fig2_drawings.csv` | Experiment 2: one sample each diffusion model generates (the authors' `eval/samples/`, first sample) next to the 12 x 12 example image each LLM drew from the weights alone. Both LLMs draw one-pixel outlines; the models generate filled shapes. |
| `fig3_heads.png` | `data/fig3_heads.csv` | Experiment 3: for each of the three language models, which of the 24 attention heads are copying (induction) heads, as measured by running the model (induction score > 0.2) and as labelled from the middle blocks' weights by Astra (its first conversation, as in the post) and by Opus (both conversations). A white x marks a head labelled copying that is not one. Opus's second romaji conversation ran no code (it wrote fake tool calls), so it has no labels. |
| `fig4_car.png` | `data/exp4_trajectories.csv` | Experiment 4: the path of the car's green wheel in one sampled episode (the same sampling key for all three networks) over the static scene from the prompt. The trained network drives onto the ramp and stalls; both edited networks first reverse to the right for a run-up, then drive left, jump the red strip and land on the blue box. The circles are the wheel (radius 0.22) at its last recorded step; contact comes on the next physics substep. |
| `fig5_minatar.png` | `data/fig5_minatar.csv` | Experiment 5: mean score over 1,000 episodes, with standard errors, of the networks the LLMs wrote from all-zero weights and of the best PPO seed after 100M and 500M training steps (PPO and Astra scores are the authors'). Each panel has its own scale. |
| `fig6_go.png` | `data/fig6_go.csv` | Experiment 6 at budget 0: win rate over 2,000 games of each LLM's Go network against the three released same-size PPO networks (Astra's numbers are the authors'), and of Opus's network head to head against Astra's budget-0 and budget-50 networks. The dashed line is 50 %. |
