> **Historical reference / fallback.** The selected delivery is now [C](../../model-c/README.md). This document records the previous reference release decision.

# Experiments: useful findings and rejected changes

**Final decision: retain the historical two-network ensemble and its reproducible CSV.** Several changes improved intermediate measures or difficult cases without improving overall possession segmentation. This report explains those differences instead of presenting every experiment as an improvement.

Retained weights and scripts are included. Rejected research scripts, checkpoints and frame-level arrays are not distributed. Published values are aggregates: [research-summary.json](../validation/research-summary.json) and [selection decision](../validation/selection-decision-2026-11-16.json). Hashes identify the source research reports without redistributing private arrays.

## 1. What we wanted to improve

The competition evaluates H/A possessions, not just individual labels. A brief H→A→H oscillation can split one correct possession into several segments and reduce temporal-IoU matching. Short possessions, team changes, stoppages/restarts and degraded tracking were the main difficulties.

The primary metric is **Spell F1**, averaged over IoU 0.3/0.4/0.5/0.6/0.7. H/A/D macro F1 measures frame classification. Better macro F1 does not necessarily mean better Spell F1.

Recent comparisons use three folds, six development matches and three conditions per match: 18 paired cases per arm. Recent parameters are selected on each fold's own internal match, then frozen before outer reports. Those six matches were already used previously; three corruptions of one match are correlated. This does not provide independent validation.

## 2. Reference: a reproducible pipeline

The reference fuses a set encoder and a relational encoder, each followed by a TCN. It jointly predicts H/A/D and transition evidence, then uses Viterbi with T=6, change penalty=2 and boundary weight=0.5. Two historical CPU environments reproduced the target CSV byte for byte.

Historical development Spell F1 is **0.69312**; the later CPU check gives **0.69318765**. These describe their respective numerical environments and remain separate. Validation uses historical weights excluding each outer fold; final supplied weights train on all six allowed matches. Validation scores are not obtained by evaluating those final checkpoints on their own training matches.

The pipeline runs on checked tracking, passes format validation and has a reproducible output. Its full-match quality on private challenge labels is unknown.

## 3. Ball reconstruction and segment queries

**Hypothesis.** Reducing missing-observation errors and learning a possession representation could improve transitions and short segments.

Trajectory completion improves geometric error on simulated gaps. In the tested recipe, this does not improve overall possession scores: reconstruction accuracy and decision usefulness measure different things. The candidate becomes more robust under severe corruption while declining on average in clean or moderate conditions.

Candidate **C** adds segment queries to this network family. All six models completed training. It reaches **0.693163 Spell F1** and **0.928450 macro F1**, versus **0.693188** and **0.925032** for the reference. Frame classification improves, but the tiny mean Spell F1 difference and some match-level regressions do not justify replacement. Enabling a branch under the same weights is not a retrained ablation.

**Decision:** retain the reference. Capacity, context, training-match counts per fold and objective changes prevent attributing differences to one component.

## 4. Specialized H/A and LightGBM Dead/Alive

**Hypothesis.** Separating “is the ball alive?” from “which team possesses?” could simplify the tasks and exploit player dynamics. Three research LightGBM models are fold-bound. The six evaluated H/A networks initialize from their complete C checkpoint in the same fold and train to the fixed endpoint.

Only H/A-labelled frames contribute to H/A state loss. D/X are masked in that loss, not relabelled Home/Away. Dead frames stay in the timeline for context and auxiliary stop/restart tasks. LightGBM supplies Dead/Alive evidence in the relevant arms.

In the LightGBM-only D-emission arm, before temperature and temporal decoding:

```text
qH + qA = 1                 # conditional H/A network
pD                         # Dead/Alive detector
P(H) = (1 - pD) × qH
P(A) = (1 - pD) × qA
P(D) = pD
```

Thus the final output can be D despite two specialized network outputs. If Dead is incorrectly declared Alive, the H/A network must choose a team. Soft fusion instead blends C and LightGBM D log-odds with internally selected coefficients. Joint decoding retains H/A↔D transitions.

Even when C supplies D evidence, changing H/A or transition evidence can move D boundaries in joint decoding. Identical-predicted-D-mask controls distinguish that effect from H/A classification; they do not use true D as an input.

| Completed H/A comparison | Spell F1 | H/A/D macro F1 | D F1 |
|---|---:|---:|---:|
| Frozen reference | **0.693188** | 0.925032 | 0.978587 |
| Complete C | 0.693163 | **0.928450** | **0.981414** |
| Trained H/A, no LightGBM fusion | 0.688990 | 0.928017 | 0.981356 |
| H/A with soft fusion, primary candidate | 0.687544 | 0.926983 | 0.980092 |
| H/A with LightGBM-only D emission | 0.601613 | 0.868295 | 0.911081 |

**Finding.** The soft-fusion primary fails the replacement criteria. The small internal calibration gain in the fold activating LightGBM does not transfer externally; other folds select the neutral setting. LightGBM-only D evidence substantially degrades the pipeline.

**Limited interpretation.** These are full-pipeline results, not evidence that LightGBM is intrinsically a worse binary classifier. Representation, probabilities, scales and decoder interact. Nor do these scores causally isolate the H/A loss.

Under the same predicted D mask in the relevant controls, retrained H/A remains worse, so D-detector error does not explain the entire decline. Swapping model probabilities and boundaries associates most of the inference regression with the new H/A probabilities. This localizes a diagnostic lead without identifying a causal training loss or layer.

## 5. Calibration and temporal decoding

### Better calibration, no better Spell F1

Internally selected H/A temperature lowers log-loss and calibration error on known H/A frames. This experiment does not replace D emission.

| Measure | Fixed H/A | Log-loss-calibrated H/A |
|---|---:|---:|
| H/A log-loss, lower is better | 0.282544 | **0.220850** |
| ECE10, lower is better | 0.037638 | **0.015666** |
| Spell F1, higher is better | **0.688990** | 0.688093 |

For this primary, all three folds select temperature 2. It reduces overconfidence without changing framewise argmax, although temporal decisions change. **Better confidence does not guarantee better possession segmentation.** Separate Viterbi-cost selection on internal matches also fails to recover the decline in completed experiments. Reference parameters remain unchanged.

### Argmax retrieves more short possessions but predicts far too many

Another comparison reuses frozen neural evidence, without training, and compares Viterbi, framewise argmax and direct votes using existing boundary scores.

| Frozen-evidence decision | Spell F1 | Predicted short possessions | Predicted-short precision at IoU 0.5 |
|---|---:|---:|---:|
| Viterbi | **0.688990** | 3400 | **68.32%** |
| Direct argmax | 0.473113 | 11976 | 21.36% |
| Tested fixed probabilistic vote | 0.154075 | 1435 | 25.09% |

Argmax retrieves more short GT possessions but predicts many more short fragments. Fixed voting suffers the reverse problem and also degrades D: proxy boundaries do not supply the needed intervals. All six internal selections retain Viterbi.

**These votes jointly decide H/A/D.** They do not measure H/A-only quality under an identical D mask. GT-short recall matches short GT to predictions of any duration; predicted-short precision matches short predictions to same-team GT of any duration. These populations cannot be combined into one short-possession F1.

The conclusion is limited to the tested policies on these outputs; it does not establish Viterbi's universal superiority over learned models. Proxy voting is postprocessing on frozen outputs, without new learning. A soft-duration prototype was checked on synthetic sequences; no real-football gain is claimed.

## 6. Submission lessons

- Judge the final pipeline on possessions, not only reconstruction, frame classification or calibration.
- Evaluate decoder and evidence together; replacing one does not guarantee improvement with existing outputs.
- A small internal selection gain may fail to transfer to other matches.
- Preserve completed negative findings and distinguish them from prototypes.
- Ship the retained reference with unchanged weights, code and CSV; document reproducibility and limitations.

This is a descriptive report: six reused development matches, correlated corruptions, few seeds and multiple protocol differences. No significance claim, independent validation or private target-match score is asserted. This release starts no training, neural forward pass or cluster job.
