# Experiments and selection limits

The selected delivery is the complete frozen fold-0 C pair. The comparison below averages the same six historical held-out development matches and three conditions (18 cases per arm), using the official possession metric. These matches were reused across research decisions; conditions are correlated and the latest check uses one seed. Scores are not an independent final test or a private challenge-match score.

| Pipeline | Development Spell F1 |
|---|---:|
| Reference | 0.693188 |
| Complete C | 0.693163 |
| C with LightGBM-only D emission | 0.606253 |
| Specialized H/A, C D evidence | 0.688990 |
| Specialized H/A, selected soft D fusion | 0.687544 |
| Specialized H/A, LightGBM-only D emission | 0.601613 |

The reference and C are close in aggregate; each leads on three matches. C improves severe corruption while the reference improves clean input. C is the team's chosen delivery, without a claim of a statistically established overall improvement.

The player-only LightGBM detector was trained with fold-bound data and fused with neural evidence. Replacing D emissions harms the tested complete pipelines. This is a whole-pipeline finding: it does not prove that LightGBM is intrinsically a poor binary classifier. Representation, class probability scale and joint transition decoding interact.

The specialized H/A networks mask D/X in their H/A state loss; those frames remain in the input timeline for context and auxiliary transitions. They are not mislabeled as Home or Away. Their completed training and soft fusion do not improve mean Spell F1. H/A calibration improves log-loss from 0.282544 to 0.220850 but lowers Spell F1 from 0.688990 to 0.688093. Intermediate calibration quality and final segment quality are different objectives.

On the evaluated frozen H/A evidence, Viterbi reaches 0.688990 Spell F1; framewise argmax reaches 0.473113 and fixed probabilistic voting 0.154075. The tests concern joint H/A/D postprocessing, not every possible sequence model. More correct short contacts are insufficient when many spurious short possessions are created. A duration prototype has synthetic checks only; no football gain is asserted.

Ball reconstruction improves simulated missing-coordinate error, but coordinate accuracy does not directly imply better possessions. Segment queries improve some difficult cases and frame metrics without a demonstrated mean Spell F1 gain. Architectural differences, ancestry, data partitions and objectives prevent attributing results to a single component. Enabling a branch under the same weights is not a retrained ablation.

The final checkpoint pair and its decoder were fixed before target diagnostics. The team chose C after comparisons including the public-window diagnostic; no further target-label fitting or parameter search was performed for delivery. The public window is unsuitable for claiming full-match generalization. The release preserves that exact C identity and leaves the old reference available as a separate fallback.

Aggregate [per-match](../validation/development/per_match.csv), [per-condition](../validation/development/per_condition.csv) and [all-case](../validation/development/all_18_cases.csv) metrics are included with [provenance](../validation/development/provenance.json). They contain scores only, without tracking or framewise evidence.
