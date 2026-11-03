**Observations**

*Layer 0*
- **L0H5 has an untrained query-key pair, so its attention is uniform.** Its W_q/W_k block norms are 2.63/2.94, below the random-init norm of about 3.14. Its value and output are large (8.45/9.56). Its value reads position directions, and it writes the massive dimensions 21/39/99/58 (W_o rows 5.35, 3.77, 3.36, 3.00).
- **L0H3 is a strong positional head that moves character identity.** It has the largest query-key norms (14.1/14.7).
  - Its query-key matrix is a stack of rotations: complex eigenvalue pairs 1.53@±44°, 1.13@±39°, 1.09@±46°, 1.07@±51°, 1.05@±61°, 0.94@±84° and so on. That is a relative-position structure.
  - Its Q/K top directions lie 97–100% in that positional subspace. Its V directions lie 100% in a separate subspace (overlap 0.01–0.02).
- **L0H0 is a second, weaker positional head that also moves tokens.** Its query-key norm is about 7, Q/K are 56–63% positional, and V reads tokens.
  - In L0H3's dominant plane it rotates by 37° (L0H3: 44°).
  - Assuming L0H3 is offset 1, its positional kernel in L0H3's planes peaks about 2 characters back and avoids the current position.
- **L0H1, L0H2 and L0H4 compute position features.** Their query-key matrices are nearly rank 1 (singular values 1.2, 0.82, 0.64) and their values have about rank 2, reading position. They write the massive dimensions (W_o rows on dims 39/21/58/99).

*Attention sink*
- Every L1 and L2 head reads dimension 21 (and 99) on the key side through the LayerNorm-bias query. L1 W_k column norm on dim 21 is 11.5 versus a median of 0.77. The implied key bias is u[21] ≈ 16–29 in L1, 6–17 in L2, and 0.2–3.3 in L3.

*Alphabet size*
- **Input side: about 24 directions.** Stacking the V weights of L0H0 and L0H3 gives exactly 24 token directions (singular values 2.49 → 1.12).
  - Measured against random directions, the layer-0 MLP reads them at 3–10×, and L1 Q/K/V/MLP at about 2–3×.
  - Direction 25 is borderline (1.7×). Everything after it is positional (layer-0 Q/K energy 2.4–3.7×) or noise.
- **Output side: also 24 directions.** With the massive rows removed, W_proj.3 has 24 spread-out directions (singular values 21.9 → 8.1, participation 80–157 dimensions). After them come localized ones (5.2, 4.7, 4.4).
  - The copy head L3H3 writes into exactly those 24 directions (2.3–7.4× random) and not beyond (0.5–1.3×).
- So both the input and output alphabets span about 24 symbols, not 28.

*Induction*
- L3H3 has a clear induction circuit. I measured how strongly its query-key matrix, combined with L0H3's output, prefers matching the same token. The score is 0.56–0.71, against 0.01 ± 0.06 for random directions.
  - Its key reads L0H3's output (1.6×) and its value reads tokens (2.1×).
- No L1 or L2 head shows this (scores ≤ 0.29).

*L1 heads*
- Queries read the current token plus the previous token from L0H3 (3.4–4.1×).
- Their direct query-key form on the token subspace is strongly negative (−0.4 to −0.6), so they avoid positions with the same character.
- In L0H3's planes they rotate by 95–112°, about twice L0H3's angle, which suggests a weak preference for 2–3 characters back.

**Heads**
- L0H0: local – positional query-key, token value; kernel peaks about 2 back, matching L0H3's rotation in the top plane at 37°
- L0H1: broad – weak rank-1 positional query-key (singular value 1.2), position-reading value, writes sink dimensions
- L0H2: broad – same pattern (singular value 0.82)
- L0H3: previous – strongest rotational positional query-key, value carries all 24 token dimensions; it is the key source of the L3 induction head
- L0H4: broad – same pattern as L0H1 (singular value 0.64)
- L0H5: broad – query-key at init, so attention is uniform; averages position features into dimension 21
- L1H0: local – sink key bias plus content, weak 2-back positional kernel, query uses the previous token
- L1H1: local – same, kernel peaks about 2 back
- L1H2: broad – sink and content dominated, weak positional term
- L1H3: local – kernel peaks 1–2 back
- L1H4: broad – weakest positional term, avoids same character and same previous character
- L1H5: local – kernel peaks 1 back
- L2H0: local – weak 2-back kernel plus sink
- L2H1: unclear – content features from the MLPs, kernel peaks 1 back, weak
- L2H2: broad – sink plus content
- L2H3: broad – smallest positional and token structure in L2
- L2H4: local – strongest 2-back kernel in L2; weak induction score 0.17–0.29
- L2H5: broad – sink plus content
- L3H0: unclear – content heavy, small recency term
- L3H1: broad – small query-key (norm 9.7), large value
- L3H2: unclear – no clear token or positional structure
- L3H3: copy – induction circuit as described above
- L3H4: broad – smallest query-key (8.6/8.3) with very large value/output (17.9/18.3), so nearly uniform
- L3H5: unclear – content, weak recency

**Reasoning**

- **Corpus.** Both the input side (what the token-moving heads and the layer-0 MLP read) and the output side (where the final MLP and the copy head write) show an alphabet of about 24 symbols.
  - Romaji in Hepburn uses exactly 22 letters (a b c d e f g h i j k m n o p r s t u w y z) plus space and newline, which is 24. The letters l, q, x and almost all v never appear.
  - English and Python use all 26 letters. The rarer ones (j, x, q, z) occur in names and identifiers, and an induction head would need to copy them, so each would show up as an extra direction. None is seen.
- **Order.** The model has a sharp previous-character head (L0H3) that carries full character identity, and a layer-3 induction head built on it.
  - With characters shuffled inside each line, the previous character and copied sequences carry no information. That model would instead average over the line.
  - So the text was used as is.

corpus: romaji
order: as is
probabilities: python=0.07 romaji=0.78 english=0.15; as is=0.85 shuffled=0.15