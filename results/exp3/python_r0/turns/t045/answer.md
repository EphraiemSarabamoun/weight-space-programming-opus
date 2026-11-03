**Observations**

All quantities below are computed from the weights only.

*Token subspace (how many characters are used)*
- Stacking W_v.0 of heads 1, 2, 3 and 5 gives about 27 strong directions with a sharp cliff: singular values …1.98, 1.88, 1.84, then 1.29, 1.20 and the bulk.
- These 27 directions are almost orthogonal to the subspace read by the Q/K of L0H1 and L0H5 (sum of cos² ≈ 0.8 over 20 dimensions).
- ICA of the MLP0 input weights inside this subspace gives 27 very sparse components (kurtosis 15 to 375). These look like token detectors, so all 28 symbols appear to be trained.
- About 5 tokens are weakly trained: lowest L0H5 value gains (0.13–0.18 vs about 0.21) and almost zero induction weight. These are the rare characters.

*Positional heads in layer 0*
- The QK matrices of L0H1 and L0H5 live almost entirely in a positional subspace: 28.2 of 32.5 and 42.3 of 46.3 in Frobenius norm.
- They have rotation structure and their symmetric and antisymmetric parts almost commute (commutator 0.06, vs 0.26 for random matrices). This is what relative-position attention on sinusoid-like position embeddings looks like.
- In shared rotation planes, L0H1's angles are about twice L0H5's (130° vs −110°, −116° vs −51°, …). L0H5 also has a negative eigenvalue on the period-2 direction.
- Taking L0H5 as offset 1, the reconstructed L0H1 profile over offsets 0–4 is −49, 14, 58, 21, −48, which peaks at offset 2.

*Uniform head and attention sink*
- L0H4 has a QK of essentially zero (top singular value 0.04), so it averages uniformly over all earlier positions.
- Its OV is rank 2 and very large (13.9, 9.7) and reads positional directions. L0H0 writes into the same plane (overlap 0.97 and 0.92).
- Every L1 key reads this plane 36–90× more than an average direction, and every L1 and L2 head has a query-bias key preference pointing the same way (angle about −145°). This is a shared sink / coarse-position mechanism.

*Induction circuit*
- Restricted to the token subspace, the matching between the query's current token and the key's L0H5-written token is strongly positive:
  - L2H1: +0.88 to +0.90
  - L3H1: +0.83 to +0.85
  - L3H5: +0.89 to +0.91
  - L3H0: +0.42
- For L2H1 and L3H1, plain same-token matching is neutral or negative, so the match is specifically against the previous token.

*Delimiter tokens*
- The two most "special" tokens are the top key targets of L0H0 (key norms 0.15 and 0.10). The first is also the token L0H4 averages most (V gain 0.21 vs about 0.11 for others) and has the densest MLP0 code.
- Both have AND-type MLP0 detectors for "same token twice in a row": scores 10.4 and 10.0, against an off-diagonal 99th percentile of 7.3. These neurons have negative thresholds (bias −0.41 and −0.34, plus a negative read of the sink plane).
- There is also a run detector for three of the second token (6.2) and a "first token after two of the second" detector (6.6).

**Heads**
- L0H0: broad. QK is weak (top singular value 0.71) and keyed on the two delimiter tokens with no recency. V reads position and its rank-2 output goes into the sink plane.
- L0H1: local (about two back). Q/K are positional, angles are about 2× L0H5's, profile peaks at offset 2, and the OV moves full token identity (flat spectrum).
- L0H2: unclear. Token–token QK (5.2 of 9.1) of several ranks, little positional part.
- L0H3: local. Mixed content and position QK (token–token 4.0, position–token 3.7), weak preference for offsets of about 3–5 that avoids the current position.
- L0H4: broad. QK ≈ 0, so it is a uniform average; its huge rank-2 positional OV is read by later keys.
- L0H5: previous. Positional rotation QK with a period-2 component favouring odd offsets. Its output is the key side of the induction heads and is read by L1 values (4–6×) and MLP1 (3.8×).
- L1H0: broad. Sink-plane keys, weaker than the other L1 heads (|u| 6.7).
- L1H1: broad. Sink-plane keys (enrichment 78–88×, |u| 25); values gather tokens and previous tokens.
- L1H2: broad. Same structure.
- L1H3: broad. Same structure.
- L1H4: broad. Same structure.
- L1H5: broad. Same structure.
- L2H0: broad. Sink-plane keys (11–12×).
- L2H1: copy. Matches against L0H5's output at +0.88; no sink reading.
- L2H2: broad. Sink-plane keys (19–23×).
- L2H3: broad. Sink-plane keys (20–24×).
- L2H4: broad. Sink-plane keys (18–21×).
- L2H5: broad, weak (8.5×).
- L3H0: copy (weak). Matching against L0H5's output at +0.42.
- L3H1: copy. Matching at +0.84; 33% of its OV goes into the logit subspace.
- L3H2: unclear. Mild sink-key reading, no matching.
- L3H3: broad. Weakest QK in layer 3 with its largest OV (14.7).
- L3H4: unclear. Mild sink-key reading, no matching.
- L3H5: copy. Matching at +0.90, and its OV writes most strongly into the logit subspace.

**Reasoning**
- **Not romaji.** About 27 trained token-difference directions means about 28 symbols are in use. Romaji lacks l, q, x and nearly v, and would show only about 23–24.
- **As is, not shuffled.**
  - The text order is real: there is a sharp previous-token head, a two-back head, and a textbook induction circuit keyed on the previous-token head.
  - All of this is useless if the characters inside each line are shuffled. A shuffled model would instead rely on bag-of-line broad heads that read tokens; here the uniform head reads position.
- **Python rather than English.**
  - The two delimiter-like tokens (space and newline) are both heavily trained.
  - Both have AND detectors for "same token twice in a row", plus run and blank-line style patterns. This matches indentation runs and blank lines.
  - In English text spaces never repeat and newlines are rare.
  - The model's heavy use of copy and broad heads, with only very short local heads, also fits repetitive code.
- The labelling of the space and newline tokens is inferred, not certain.

corpus: python
order: as is
probabilities: python=0.70 romaji=0.06 english=0.24; as is=0.88 shuffled=0.12