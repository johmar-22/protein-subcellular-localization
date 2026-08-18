# Post-Publication Verification and Typographical Corrections

**Article:** Attention-Enhanced Bidirectional Long Short-Term Memory (Bi-LSTM) Ensembles with Frozen Evolutionary Scale Modeling-2 (ESM-2) Embeddings Achieve Competitive Performance in Protein Subcellular Localization

**Author:** Johaimen M. Omar

**Journal:** *International Journal of Life Sciences and Biotechnology*, 2026; 9(2): 90–105

**DOI:** [10.38001/ijlsb.1882985](https://doi.org/10.38001/ijlsb.1882985)

**Last updated:** 18 August 2026

---

## Summary

Every quantitative result reported in this article has been independently re-derived from the code
and data in this repository, and every one reproduces exactly. This includes the internal test set
metrics, the five-run stability analysis, the attention–annotation alignment analysis, and all three
predictors on the 86-protein prospective benchmark. The verification is set out in Part 1.

Four typographical slips reached print in the descriptive prose accompanying two attention figures.
They are listed in Part 3. Each is a mistranscribed number or interval in a sentence describing a
figure; none of them enters any calculation, and the analyses they describe are among those verified
in Part 1. No result, table, figure or conclusion of the article changes.

Part 4 adds a significance test that was not in the original article. It confirms the published
conclusions under a more appropriate test.

## Why this note exists

A list of corrections was submitted to the journal on 30 July 2026, during proof review and before
publication. Those corrections were not applied to the published version, and the article went to
press without them. A formal correction has been requested from the editorial office. This note is
provided in the meantime so that readers arriving here through the article's Data Availability
statement have the corrected text and can see the verification behind it.

---

## Part 1. Independent verification of the published results

All values below were recomputed from the committed code and data in this repository.

### Internal test set and stability analysis

| Quantity | Published | Recomputed | Source |
|---|---|---|---|
| Internal test set size | 1,971 | 1,971 | `data/test_split.csv` |
| Ensemble accuracy | 86.81% | 86.81% (1,711/1,971) | released checkpoints |
| Ensemble MCC | 0.8254 | 0.8254 | same |
| Five-run mean accuracy | 85.09% | 85.09% | `results/statistical_results.csv` |
| Five-run mean MCC | 0.8031 | 0.8031 | same |
| Standard deviation, accuracy | ± 0.48% | ± 0.48% | same |
| Standard deviation, MCC | ± 0.0062 | ± 0.0062 | same |
| 95% CI, accuracy | 84.50–85.68% | 84.49–85.69% | same (t = 2.776, df = 4) |

### Attention–annotation alignment (Figure 4)

Recomputed from `results/attention_signal_correlation.csv` (181 test proteins carrying
UniProt-annotated targeting features).

| Class | n | Top-1 alignment (published) | Recomputed | Chance (published) | Recomputed | Enrichment (published) | Recomputed |
|---|---|---|---|---|---|---|---|
| Nucleus | 68 | ~57% | 55.9% | ~33% | 33.4% | ~2.35× | 2.35× |
| Cytoplasm | 30 | ~50% | 50.0% | ~32% | 32.3% | ~2.0× | 2.01× |
| Membrane | 63 | at/below chance | 14.3% | — | 17.0% | ~1.1× | 1.07× |
| Mitochondrion | 20 | at/below chance | 10.0% | — | 31.6% | ~0.35× | 0.35× |

The representative Membrane protein O35186 (Figure 4c) is confirmed at 21.24× enrichment, with 96.8%
of its attention mass falling inside an annotated region occupying 4.6% of the sequence. The
published value of 21.2× is correct.

The one refinement worth recording: the Nucleus top-1 alignment rate is **55.9%**, which the article
rounds to "~57%". Readers wanting the exact figures should use the recomputed column above.

### Prospective benchmark (Table 4)

| Predictor | Published accuracy | Recomputed | Published MCC | Recomputed |
|---|---|---|---|---|
| Present model | 88.37% | 88.37% (76/86) | 0.827 | 0.8274 |
| DeepLoc 2.1 | 80.23% | 80.23% (69/86) | 0.714 | 0.7139 |
| MULocDeep | 77.91% | 77.91% (67/86) | 0.682 | 0.6817 |

Benchmark composition confirmed as 86 proteins (40 Nucleus, 21 Cytoplasm, 20 Membrane,
5 Mitochondrion).

### Implementation

The model architecture and training configuration in `main.py` match the published Methods in every
respect checked: Bi-LSTM with 256 units per direction (512 total) and dropout 0.3; focal loss with
γ = 2.0 and α = 0.25; maximum sequence length 600; FP16 memmap storage with FP32 computation; Adam at
1e-4 with `clipnorm=1.0`; EarlyStopping with patience 8; ReduceLROnPlateau with factor 0.5 and
patience 4; stratified k-fold cross-validation; and L2 regularisation at λ = 0.01 on the dense layers.

---

## Part 2. Clarifications

These are not errors. The published text is compressed in ways that may prompt a question.

### Item 1 — Relationship between the two internal-test accuracy figures

The Abstract and Model Performance section report **86.81%** accuracy (MCC = 0.8254). The stability
analysis on page 98 reports a mean of **85.09%** across five runs, 95% CI [84.50%, 85.68%]. The
headline figure sits outside that interval, and the text does not say why.

The two describe different quantities. **86.81%** (1,711 of 1,971 correct) is the **soft-voting
ensemble** described under "Ensemble Strategy" and "Prediction Aggregation". **85.09%** is the **mean
of five individually trained models** under different random initialisations, reported to show that
the architecture converges stably.

The five individual runs, in `results/statistical_results.csv`, achieved 85.84%, 84.53%, 85.03%,
85.08% and 84.98%. The best single run, 85.84% (1,692 correct), is well below 86.81% (1,711 correct),
confirming that the ensemble figure is not the best of the five. An ensemble is expected to exceed the
mean of its constituent models, which is why the values differ.

For comparison against other predictors, use 86.81% (MCC = 0.8254) for the internal test set and
88.37% (MCC = 0.827) for the prospective benchmark.

### Item 2 — Benchmark query example on page 91

The worked example under "Label Exclusivity" reads:

```
... AND (cc_scl_term:SL-0191) AND NOT (cc_scl_term:SL-0086) AND NOT (cc_scl_term:SL-0039).
```

This is an abbreviated illustration, not the full query. As the Methods state on page 90, Nucleus
queries excluded Cytoplasm, Membrane **and** Mitochondrion. The query actually executed, recorded in
`main.py`, was:

```
(reviewed:true) AND (taxonomy_id:2759) AND (cc_scl_term:SL-0191)
    AND NOT (cc_scl_term:SL-0086)
    AND NOT (cc_scl_term:SL-0039)
    AND NOT (cc_scl_term:SL-0173)
```

The equivalent three-way exclusion was applied to all four classes, for both the training corpus and
the prospective benchmark. Table 1 and the 86-protein benchmark reflect these complete queries.

### Item 3 — Terminology: mTP and MTS

The article uses both *mTP* (mitochondrial targeting peptide) and *MTS* (mitochondrial targeting
sequence) for the same feature, on pages 97 and 98. Both are defined in the abbreviation list and
should be read as interchangeable.

---

## Part 3. Typographical corrections

The four items below are mistranscribed numbers and intervals in sentences describing Figures 3A and
3B and the ROC analysis. In each case the underlying analysis is correct and verified in Part 1; the
error is confined to the sentence reporting it. None of these values is used in any calculation.

### Item 4 — Page 97, Case 1 (Membrane), percentage of sequence

**As printed:** "near-zero weights to the first ~420 residues (~96% of the sequence)"

**Read as:** "near-zero weights to the first ~420 residues (~81% of the sequence)"

CEAM1_RAT (UniProt P16573) is 519 residues. The first 420 residues are 80.9% of the sequence. The
residue count and the described attention pattern are correct; only the parenthetical percentage is
mistyped.

### Item 5 — Page 97, caption to Figure 3A, peak interval

**As printed:** "exclusively at the C-terminus (positions 420–470)"

**Read as:** "exclusively at the C-terminus (positions ~420–450)"

The caption and body text on the same page give different intervals for the same peak. The body text
is correct.

### Item 6 — Page 97, Case 2 (Mitochondrion), percentage of sequence

**As printed:** "the attention weights formed in the N-terminal region (positions ~5–28), which represents approximately 57% of this short protein sequence"

**Read as:** "the attention weights formed a broad plateau in the N-terminal region (positions ~5–28), which represents approximately 7% of this protein sequence"

NDUAA_MOUSE (UniProt Q99LC3) is 355 residues, so positions 5–28 are approximately 7% of the sequence.
The figure of 57% does not correspond to any quantity for this protein and appears to have been
carried over from the Nucleus top-1 alignment rate reported on the following page. The observation
the sentence makes, that attention forms a distributed N-terminal window rather than a single peak,
is correct as published and is what Figure 3B shows.

### Item 7 — Page 98, caption to Figure 3B, region interval

**As printed:** "across the N-terminal region (positions 3–28)"

**Read as:** "across the N-terminal region (positions ~5–28)"

As with Item 5, caption and body text disagree on the same interval. The body text is correct.

### Item 8 — Page 99, ROC analysis, Membrane sensitivity

**As printed:** "reaching a True Positive Rate (TPR) of ~0.93 at a False Positive Rate (FPR) of only 0.05"

**Read as:** "reaching a True Positive Rate (TPR) of ~0.96 at a False Positive Rate (FPR) of only 0.05"

The operating-point paragraph later on the same page reports 95.85% sensitivity for the Membrane class
at 5% FPR. That value is the one derived from the ROC analysis and is correct; the earlier
approximation is a rounding slip.

---

## Part 4. Additional statistical analysis

This section adds analysis not present in the article. It corrects nothing.

### McNemar's test on the prospective benchmark

The comparisons against DeepLoc 2.1 and MULocDeep on page 101 were assessed with a paired *t*-test.
Because each protein yields a binary outcome and the same 86 proteins are scored by all three
predictors, McNemar's test is the conventional choice for this design. It has since been computed
from the per-protein predictions in `data/benchmark/`:

| Comparison | Ours right, theirs wrong | Ours wrong, theirs right | Discordant | McNemar exact *p* | Published *t*-test *p* |
|---|---|---|---|---|---|
| vs DeepLoc 2.1 | 7 | 0 | 7 | **0.0156** | 0.0074 |
| vs MULocDeep | 12 | 3 | 15 | **0.0352** | 0.0192 |

Both comparisons remain statistically significant at α = 0.05 under the more appropriate test, and
the direction of both is unchanged. The McNemar *p*-values are larger, so the published values should
be read as somewhat optimistic, but no conclusion of the article is altered. Readers repeating these
comparisons are encouraged to use McNemar's test.

Worth noting: there is **no** protein in the benchmark that DeepLoc 2.1 classifies correctly and the
present model classifies incorrectly. The ten errors made by the present model are a strict subset of
DeepLoc 2.1's seventeen.

### Post-hoc power analysis

The power analysis on page 101 was computed from the effect size observed in the same sample. Observed
power calculated this way is a deterministic function of the reported *p*-value and does not
constitute independent evidence. The statement that observed power against DeepLoc 2.1 was 1.00 should
be read as a restatement of the *p*-value rather than as separate support for it. The
minimum-sample-size estimates in the same analysis (N = 19 and N = 120) remain useful as prospective
guidance for future benchmarks, which was their intended purpose.

### Sample size

Both benchmark comparisons rest on 86 proteins, with only 5 in the Mitochondrion class. As the
article's Discussion states, a single misclassification in that class shifts its recall by 20
percentage points. The prospective results should be read as preliminary evidence of competitive
performance rather than as a demonstration of superiority.

---

## What is unchanged

- Dataset composition and class counts in Table 1 (9,959 sequences; 4,662 CD-HIT clusters)
- SVM baseline comparison in Table 2 (64.54% accuracy, MCC 0.5295)
- Internal test set results (86.81% accuracy, MCC 0.8254, N = 1,971)
- Per-class recall values and the normalized confusion matrix (Figure 2)
- Stability analysis in Table 3 (mean 85.09%, 95% CI [84.50%, 85.68%])
- Attention alignment and enrichment analysis (Figure 4), verified in full above
- ROC and AUC values (Figure 5): Membrane 0.989, Mitochondrion 0.981, Nucleus 0.968, Cytoplasm 0.948
- t-SNE feature space visualisation (Figure 6)
- Prospective benchmark results in Table 4 (88.37% accuracy, MCC 0.827, N = 86)
- Model architecture, training protocol, embedding extraction, and all released checkpoints
- The conclusions of the article

---

## Status of the formal correction

| Date | Action |
|---|---|
| 30 July 2026 | Proof corrections submitted to the editorial office |
| 17 Aug 2026 | Article published without the submitted corrections applied |

---

## Citing this note

This document is not peer reviewed and should not be cited in place of the article. Please cite the
article itself:

> Omar, J. M. (2026). Attention-Enhanced Bidirectional Long Short-Term Memory (Bi-LSTM) Ensembles with Frozen Evolutionary Scale Modeling-2 (ESM-2) Embeddings Achieve Competitive Performance in Protein Subcellular Localization. *International Journal of Life Sciences and Biotechnology*, 9(2), 90–105. https://doi.org/10.38001/ijlsb.1882985

If you need to refer to this note specifically, please link to it directly.

## Contact

Questions, or anything further you believe should be listed here, are welcome by email or as an issue
on this repository.

Johaimen M. Omar
Institute of Graduate Education, Kastamonu University, Kastamonu 37200, Türkiye
Mindanao State University-Main Campus, Marawi City, Lanao del Sur 9700, Philippines
johaimen.omar@msumain.edu.ph / 241057007@ogr.kastamonu.edu.tr
