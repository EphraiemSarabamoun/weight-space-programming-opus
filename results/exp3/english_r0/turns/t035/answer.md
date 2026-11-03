Observations: Interpretive key: P = the positional subspace read by L0H2's query/key. B = the 2-dimensional subspace written by heads L0H0, L0H1, L0H4 and L0H5. The "slow plane" is the 2-D part of P that changes slowly with position. "x random" means how much more a matrix reads a subspace than a random subspace of the same size would be read.

**L0H2 is a strong positional head**
- Its |W_q| is 30.8 and |W_k| is 28.3, against 3 to 7 for the other layer-0 heads. Its QK has about 20 significant singular values (11.3, 10.2, 9.7, 9.65, …, 2.3, then 1.5).
- QK is rotation-like rather than symmetric (antisymmetric norm 23.6 vs symmetric 22.5). Its eigenvalues are complex, with angles 52°, 104°, 69°, 157°, 32° and a real −6.53 at 180°. A self-attending head would have positive real eigenvalues, so this head attends a fixed distance back. The 180° (alternating) component points to an odd offset, most likely 1.
- Its query and key read the same subspace (principal cosines 0.96…0.6). The layer-0 MLP ignores that subspace (0.2–0.9x random), so it is positional, not token content.
- Its OV is high-rank (sv 0.79, 0.72, 0.68 … decaying over about 25 dimensions). It copies token identity, and the layer-1 values read its output at 5–10x random.

**L3H1 is an induction head built on L0H2**
- QK(L3H1)·OV(L0H2) is strongly identity-like: trace/‖·‖ = +1.17, real-part fraction of eigenvalues +0.84. The query's current token matches the key's previous token.
- Controls are near zero: the other layer-0 heads give −0.13 to +0.02, and random OVs ±0.02. No other head shows this (all others between −0.20 and +0.33).
- L3H1 also matches the context summaries written by layers 1–2 (+0.25 to +0.45). It has one of the two largest OVs (sv 10.6 … 5.2 over about 24 dimensions).

**Segment / "content clock" mechanism**
- L0H0, H1, H4 and H5 have tiny rank-1 QKs (top sv 0.41, 0.20, 0.71, 0.61). Their keys read content directions (only 0.03–0.21 in P). Their values read a single 2-D plane of P (about 100x random). They write that copied position into B, which is 2-D (B sv 12.0 and 8.4).
- B is read about 25–29x random by all six layer-1 keys, 5–11x by layer-2 keys, and 4–6x by L3H2–4 keys. It is barely read by queries (about 3x) and not by values or MLPs (0.1–0.5x).
- In layer 1 the constant query (from the LayerNorm bias) points into B (77–83%). The B→B block of QK is a near-pure rotation of about 70° (e.g. [[0.44, −1.59], [2.09, 0.64]]), with the same orientation as L0H2's recency plane.
- So layers 1–2 attend by recency in a content-defined clock: the current segment, then recent ones.
- The layer-1 heads are near-identical (QK top sv about 7, OV trace/‖·‖ about −1).
- About two-thirds of layer-0 MLP neurons are still at initialisation (1025 of 1536).

**Output side**
- W_proj.3 has a clear logit-space structure: 23 strong singular values (26.3 … 10.4), then 5 medium (8.35, 8.06, 7.91, 7.46, 7.17), then 5.8, 4.1, then a bulk around 2–3. W_proj.2 by contrast is flat.

Heads:
- L0H0: broad – weak rank-1 QK (0.41) with a content key; copies a positional plane into B.
- L0H1: broad – same pattern (QK 0.20).
- L0H2: previous – sharp relative-position QK (norm about 30, rotation eigenvalues including 180°) with a token-copying OV; the source for the induction head.
- L0H3: broad – rank-1 positional QK (sv 2.87, both sides in the slow plane), i.e. recency-weighted averaging of a 2–3-dimensional token feature.
- L0H4: broad – weak content-keyed QK (0.71); output goes into B.
- L0H5: broad – weak content-keyed QK (0.61); output goes into B.
- L1H0: local – keys dominated by B (B→B 42x, constant query→B), attending to the current segment; weakest head in the layer.
- L1H1: local – same B-clock recency pattern; values read L0H2 (previous-token) output.
- L1H2: local – same pattern.
- L1H3: local – same pattern; strongest reader of the L0H2 output (10.8x).
- L1H4: local – same pattern.
- L1H5: local – same pattern.
- L2H0: local – B-keyed (B→B 29x, about 78° rotation) plus content QK.
- L2H1: local – weaker B use (key 5–13x); more content-driven; low confidence.
- L2H2: local – B-keyed (24x) with content QK.
- L2H3: local – B-keyed (34x).
- L2H4: local – B-keyed (30x).
- L2H5: local – B-keyed (31x).
- L3H0: unclear – identity-like QK (+1.53, matches similar contexts), but no previous-token shift (induction score −0.04).
- L3H1: copy – query current token matches key previous token via L0H2 (+1.17, eigenvalue real part +0.84); large token-copying OV.
- L3H2: local – B-keyed (key 16.7x, B→B 17x).
- L3H3: local – B-keyed (key 20x).
- L3H4: local – B-keyed (key 11x); weaker.
- L3H5: broad – weak, flat QK (top sv 2.1); very large OV (16.9, 14.7, …) that collects a bag of the context.

Reasoning:
- **Order.** A sharp previous-token head plus a clean induction head composed with it is the signature of ordered text. With characters shuffled inside lines, neither the previous character nor induction carries extra information. A shuffled model would instead be mostly bag-of-line heads. So "as is".
- **Corpus.** The logit space has about 28 or more significant dimensions: 23 strong, 5 medium, then 2 small.
  - This fits a 28-symbol alphabet where 23 symbols are common and about 5 are rare. That is English: space plus the letters down to k are common, while newline, j, x, q and z are each about 0.1–0.2%.
  - Romaji uses only about 24 symbols (l, q, v and x are essentially absent), so it should show about 23 dimensions and nothing substantial beyond.
  - Python would have frequent newlines and only about 3 rare letters (q, z, j), so more strong and fewer medium dimensions.
- Further evidence for English over Python:
  - There is only one induction head, and it sits in the last layer. Highly repetitive normalised Python ("self …") would favour early and multiple induction heads.
  - Most capacity goes into segment-recency heads, which suits word-level modelling of prose.
- Evidence for Python: the segment clock copies only a coarse 2-D positional plane, which would also suit line-level (newline) segmentation. I cannot tell which character is the segment boundary (space or newline), so Python stays a substantial alternative.

corpus: english
order: as is
probabilities: python=0.33 romaji=0.19 english=0.48; as is=0.9 shuffled=0.1