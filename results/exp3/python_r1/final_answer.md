**Observations:**
- **Layer 0 has positional heads with rotation-like QK circuits.** L0H5 and L0H1 have the largest layer-0 QK norms (‖W_q‖ 5.63 and 4.40, ‖W_k‖ 7.35 and 6.29). Their QK eigenvalues are mostly complex, with phase angles 53–58°, 105–135° and so on. Their Q/K read the high-gain ln1_w.0 dimensions (gain ≈ 0.8–1.0, against a median of 0.18), so ≈ 0.85–0.95 of their Q/K norm is positional. Their W_v reads a separate token subspace.
- **The token subspace has 27 clean dimensions.** After removing the positional directions, the stacked W_v of L0H1 and L0H5 has 27 singular values ≥ 1.31, then a gap to 0.84. In L0H1 alone, the 27th value is 0.97 and the 28th is 0.59. So all 28 characters are used and trained.
- **Relative offsets.** Take the polar (rotation) factor R of L0H5's QK on the positional subspace. Then L0H1's factor matches R² (cos 0.81, against 0.21 for R¹ and 0.30 for R³). L0H2 matches R⁰ best, and L0H3 matches R³ to R⁴. The dominant positional planes rotate by 53° and 105°.
- **A uniform-attention head.** L0H4 has ‖W_q‖ 0.64 and ‖W_k‖ 0.84, and its largest QK singular value is 0.04, so it attends uniformly. Its OV has rank 2 with singular values 13.9 and 9.65, the largest in layer 0. It reads the top positional plane and writes a new 2-D direction, which I call Sout. L0H0 writes into the same plane (OV singular values 3.4 and 2.5).
- **Layer-1 keys read Sout.** About 41–47% of each layer-1 key norm (22% for L1H0) lies in Sout, 80–90× chance. The constant query from ln1_b interacts with it at −1.7 to −2.5, i.e. a strong positional key bias.
- **Layer-1 heads avoid the query's own character.** On the token subspace, the symmetric part of their QK is negative semidefinite, with eigenvalues down to −0.9 and none above +0.1. Their values read the current token and the outputs of L0H5 and L0H1.
- **Induction heads exist in layers 2–3.** On tokens, I tested how well "query token" matches "key's previous token as written by L0H5" (alignment of query-token × L0H5-copied-token with the identity; random ≈ ±0.2). This gives +0.62 for L2H1, +0.58 for L3H1 and +0.61 for L3H5, with L3H0 weaker at +0.28. The same test with any other layer-0 head gives only about −0.35 to +0.19.
- **Two separator-like tokens.** ICA on the MLP0 neuron weights gives 27 sparse token axes. Two of them stand out: neuron-weight norm 0.166 and 0.164, against ≤ 0.139 for the rest; 48 and 27 dedicated neurons. These are almost certainly space and newline, in some order. Both are heavily used by layer-0 heads:
  - L0H1's strongest token rule is "query = sep1, key = sep2 one to three back".
  - L0H5 attends to itself at sep2.
  - L0H2 copies sep2.
  - The sep1→sep1 bigram detector in MLP0 is among the 3 strongest of 729 pairs (some artefact may inflate same-token pairs, so this is weaker evidence).

**Heads:**
- L0H0 local – query positional, key prefers the separator tokens, copies the separator's position into Sout.
- L0H1 local – positional rotation ≈ R², i.e. 1–3 back (mainly 2); strongest token rule: separator after separator.
- L0H2 unclear – mostly offset 0 (self), periodic; copies newline/space identity.
- L0H3 local – about 3–4 back (matches R³ to R⁴); weak QK.
- L0H4 broad – QK ≈ 0 (singular value 0.04), uniform over the whole context; rank-2 positional OV, gain 13.9.
- L0H5 previous – rotation fit R² 0.84–0.89 with offset k=1; V reads token identity.
- L1H0 broad – weaker positional key bias (Sout 42× chance), avoids own character.
- L1H1 local – positional key bias (Sout 88×), avoids own character (−0.63), values carry the last three characters.
- L1H2 local – the same pattern (82×, −0.62).
- L1H3 local – the same pattern (90×, −0.63).
- L1H4 local – the same pattern (86×, −0.59).
- L1H5 local – the same pattern (79×, −0.62).
- L2H0 local – positional key bias (Sout 12×), content query.
- L2H1 copy – current-token versus key's-previous-token match +0.62; no positional bias.
- L2H2 unclear – matches the query's previous character to the key's character (+0.56); layer-3 induction heads read its output.
- L2H3 local – Sout 24×, content.
- L2H4 local – Sout 21×.
- L2H5 unclear – weak positional bias, content-based.
- L3H0 copy – weak induction match (+0.28).
- L3H1 copy – induction match +0.58.
- L3H2 unclear – content QK, no positional or induction signature.
- L3H3 broad – smallest QK (‖q‖ 8.1, top eigenvalue 0.54) with the largest OV (37.9), so near-uniform averaging.
- L3H4 unclear – content-based.
- L3H5 copy – induction match +0.61, strongest in layer 3.

**Reasoning:**
- **Order.** A sharp previous-token head with token-content values, plus a 2-back head, plus layer-2/3 heads whose keys match the current token against the previous token (classic induction), only pays off if character order inside lines is intact. With shuffled lines, previous-token and induction circuits would be nearly useless, and broad within-line bag-of-characters heads would be expected instead.
- **Corpus: not romaji.** The token subspace has 27 well-trained dimensions, i.e. all 28 symbols are used. Hepburn romaji essentially never uses l, q, v or x, which should give about 23 dimensions.
- **Corpus: python rather than English.** Both separators (space and newline) are about equally prominent and heavily used. The strongest local-head rules involve a separator following a separator, which matches indentation: newline followed by runs of spaces. There is also a strong same-separator bigram detector, which would be either space-space (indentation) or newline-newline (blank lines). Space-space never occurs in English after the normalization, and newlines are rare in Wikipedia paragraphs. This is the weakest link: it rests on unlabeled token axes. It is supported by the four induction heads, as code is highly repetitive.

corpus: python
order: as is
probabilities: python=0.58 romaji=0.10 english=0.32; as is=0.92 shuffled=0.08