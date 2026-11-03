**Observations:**
- **Layer 0, head 2 looks like a fixed-offset positional head that copies characters.**
  - Its query–key matrix is large: singular values 11.3, 10.2, 9.7, 9.6, 9.3 … about 20 of them are significant.
  - The query side and key side read the same roughly 26-dimensional subspace (principal cosines 0.99 down to 0.85 across 20 dimensions).
  - Its eigenvalues are rotations: 8.2@±52°, 7.7@±103°, 7.5@±69°, 6.8@±156°, 6.5@179°, 6.2@±32°. The "cosquare" eigenvalues sit on the unit circle, which fits a relative-position kernel.
  - Its value path reads a different, token-like subspace with a flat spectrum (0.79, 0.72, 0.68, 0.67, 0.64 … about 20 dimensions).
  - I take that subspace to be the token subspace and the head-2 subspace to be positional for two reasons. The layer-0 MLP reads it at 15% of its energy (5% baseline) and reads the positional subspace at only 2.7% (7.8% baseline). The MLP's active rows also have a knee at 28–29 dimensions, which matches the 28 characters.
- **Layer 0, heads 0, 1, 4 and 5 have tiny query–key matrices and strong low-rank values.**
  - Query–key top singular values are 0.41, 0.20, 0.71 and 0.61.
  - Output–value singular values are 8.07/3.33/2.10, 5.66/3.83/1.07, 3.40/1.64/1.08 and 8.33/3.61/1.68.
  - All four read the same 2-D positional plane, the 52° eigenplane of head 2 (overlap 0.93/0.84). They write the same 3-D subspace B (principal cosines ≥0.98).
  - So they copy a coarse position "phase", probably of earlier word boundaries.
- **Layers 1 and 2 are positional heads built on B.**
  - Their keys put 51–60% of their norm on B in layer 1, and 10–21% in layer 2.
  - The B-to-B block of each query–key matrix has a dominant rotation pair: 350–660 at ±64–79°.
- **Layer 3, head 1 is an induction head.**
  - Its query–key matrix composed with head 2's output–value path has a positive-real spectrum: 0.42@±14°, 0.34@0, 0.30@±16°, 0.26@0 … About 16 eigenvalues are clearly positive. The eigenvalue trace ratio is 0.93; with a randomly rotated head-2 output it is −0.08.
  - Its self-matching on the outputs of every layer-1 and layer-2 head is +0.88 to +0.99, and its B-to-B form is real and positive (89.8@0°).
  - 47% of its output–value energy lands in the output subspace of the last MLP.
- **The last MLP writes into about 29 directions.** W_proj.3 singular values are 26.3 … 11.6 (22 values), 10.4, 8.35, 8.06, 7.91, 7.46, 7.17, 5.83, then a drop to 4.07 and a floor near 3.2. That is about 28 characters plus one extra direction, all actively predicted. The layer-0 MLP gives the same knee at 29.
- **Other details.** About 1,100 of the 1,536 layer-0 MLP neurons are untrained (W_proj.0 column norms 0.15–0.24, about the initial scale). ln1_w.0 is bimodal: median 0.14, with about 40 coordinates between 0.7 and 1.03.

**Heads:**
- L0H0: broad – query–key singular value 0.41 (near-uniform); rank-3 output–value copies the 52° position plane.
- L0H1: broad – query–key singular value 0.20; same position-plane copy.
- L0H2: previous – sharp position-to-position rotation kernel (11.3, 10.2 …; angles 32–180°); high-rank token output–value.
- L0H3: local – moderate positional query–key (2.9) with a positive real eigenvalue (2.1@0°), i.e. itself and nearby positions; weak output–value.
- L0H4: broad – query–key singular value 0.71; same position-plane copy.
- L0H5: broad – query–key singular value 0.61; same position-plane copy.
- L1H0: local – keys rank-2 on B (key singular values 3.46, 2.70); B-to-B rotation 349@±64°.
- L1H1: local – B-to-B rotation 538@±69°.
- L1H2: local – B-to-B rotation 617@±73°.
- L1H3: local – B-to-B rotation 534@±70°.
- L1H4: local – B-to-B rotation 560@±70°.
- L1H5: local – B-to-B rotation 564@±69°.
- L2H0: local – B-to-B rotation 651@±78°.
- L2H1: local – B-to-B rotation 301@±75°, plus some content matching (+0.5 to +0.7 on layer-1 outputs).
- L2H2: local – B-to-B rotation 378@±71°.
- L2H3: local – B-to-B rotation 658@±76°.
- L2H4: local – B-to-B rotation 659@±78°.
- L2H5: local – B-to-B rotation 637@±79°.
- L3H0: copy – positive content matching on layer-1 and layer-2 outputs (+0.44 to +0.84) with about a one-character phase shift (B-to-B 98@±68°).
- L3H1: copy – induction: matches the current character against the previous character supplied by L0H2 (trace 0.93); writes logits directly.
- L3H2: local – B-to-B rotation 238@±73°.
- L3H3: local – B-to-B rotation 305@±77°.
- L3H4: local – B-to-B rotation 192@±74°, with weak content matching.
- L3H5: broad – flat, weak query–key (2.1, 1.9 …), huge output–value (16.9, 14.7 …).

**Reasoning:**
- **Order.** A sharp previous-character head and an induction head that matches the current character against that previous character only help if the order inside a line is real. The same goes for the word-phase position machinery in layers 1–2. With shuffled lines, a model should instead be made of within-line broad heads, so I conclude "as is".
- **Not romaji.** Both the input-side MLP and the final output MLP show about 28 active characters. Romaji in Hepburn style essentially never uses l, q, v or x, so it should show about 24.
- **English over Python.** I found no line- or indentation-specific machinery. The positional phase the network builds has a period of about 7 characters, which is word scale. Copying shows up only as two heads in the last layer, which is weaker than I'd expect for highly repetitive Python code. These points favour English, but only weakly.

corpus: english
order: as is
probabilities: python=0.33 romaji=0.12 english=0.55; as is=0.9 shuffled=0.1