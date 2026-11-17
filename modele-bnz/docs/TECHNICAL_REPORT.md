> **Historical reference / fallback.** The selected delivery is now [C](../../model-c/README.md). This document records the previous reference release decision.

# Football possession reconstruction from tracking

## Delivered result and scope

At 25 Hz, the system produces **H** for home possession, **A** for away possession, **D** for dead ball, and **X** outside processed periods. The supplied `submission/possession.csv` has 199,319 data rows and header `frame,label`. Its SHA256 is `ce852ffba08ec128edc2021d3e559138cc1b2b9bc4a6c3c6ea05b7a2952ac769`.

We retain the reproducible reference: two player encoders, two temporal networks and one Viterbi decoder. Subsequent experiments did not establish sufficient improvement on the primary metric to replace it. The [experiment report](EXPERIMENTS.md) and [published aggregates](../validation/research-summary.json) describe those comparisons; this document describes the submitted recipe.

## Assets and training data

Inference requires only tactical EPTS tracking and its XML metadata. Inputs are metric player/ball positions, player speeds, team membership and observation masks. The target pitch is 105.3 × 68 m. XML identifiers assemble channels; identities, shirt numbers and roles are not network features.

Video images, homographies, tactical tags and audio are not used by the delivered prediction. Separate exploratory pipelines considered other modalities, without validating their incorporation into this reference. Private challenge assets are not redistributed.

The final checkpoints use matches distinct from the target:

| Member | IDSSE | SkillCorner | SoccerTrack |
|---|---|---|---|
| Set-temporal | 6 matches | 14 matches | — |
| Relational-temporal | 6 matches | 14 matches | 6 matches, 12 halves |

IDSSE (Bassek et al., DFL/Sportec Solutions, CC BY 4.0) supplies dense source H/A/D references. SkillCorner Open Data (MIT) contributes partial H/A references derived by consensus of two automatic proxies, rather than manual labels created by this team. SoccerTrack v2 (Scott et al., CC BY 4.0) provides source contacts and events from which narrow H/A and partial D interiors are derived. These are not dense possession annotations. Counts, sources, transformations and licences are documented in [DATA.md](DATA.md), with third-party notices in [NOTICE.md](../NOTICE.md).

Match identifiers and weight hashes are in `weights/manifest.json`. Partial SkillCorner H/A likelihood does not turn unannotated Dead frames into negative D examples. Dense data supplies state and boundary supervision. The public five-minute target labels are excluded from training, calibration and parameter selection; no private target labels or human-produced target-game annotations are used.

## Architecture

**Set-temporal** uses a player MLP over geometry, motion and relations to the ball and opponents. Mean, maximum and attention pooling produce team representations. It has 858,380 parameters.

**Relational-temporal** adds two spatial-attention blocks over players and a ball token. Geometric biases represent relative positions, distances and teams. Soft matching between player observations estimates motion, including an unmatched option. Ball-observation context is bounded to 1,500 frames on each side. It has 925,600 parameters.

Each member uses an eight-block residual TCN, 128 channels and dilations 1 through 128. The system is noncausal and uses future observations for offline analysis. Home/away-swapped inputs are also evaluated and their outputs remapped to symmetrize states and transitions.

Both retained networks jointly predict H/A/D and transition scores. Research variants separating H/A classification from Dead/Alive detection are not the selected configuration. In those experiments, D/X frames are excluded from H/A state loss while retained for context and auxiliary tasks; they never become Home/Away targets.

## Missing observations and windows

Missing observations remain NaN until handled by encoder masks. Missing ball tracking does not force D. Training augmentation includes observation gaps, geometric noise, player masking, reflections and team exchange.

Only 1,024 central outputs per window are retained. Margins are 512 frames for the set member and 544 for the relational member; the latter also receives explicit 1,500-frame ball context. Windows are padded with missing observations at edges and do not cross period interruptions. Target intervals are Python half-open `[11185, 90652)` and `[110808, 190198)`; X is written elsewhere.

## Fusion, decoding and possession conventions

Member H/A/D distributions are averaged, as are sigmoids of transition scores. Evidence is rounded to nine decimals and globally decoded with Viterbi: **temperature 6**, **change penalty 2**, **boundary weight 0.5**. Loading verifies these settings against `weights/manifest.json`.

This balances state evidence and temporal continuity. Framewise argmax can overfragment possessions; learned boundary scores help place changes without imposing a fixed minimum possession duration.

H/A/D distinctions are learned from source annotations and transitions. The delivered pipeline contains no additional rule assigning a team from field region, shirt or audio events. Loose balls, challenges and handovers are resolved through observations and temporal decoding; an unseen ball alone is not a stoppage. D interrupts an H/A possession in the official metric. Restarts are learned as transitions back from D, rather than manually specified target-frame events. X denotes unprocessed ranges, not a third team.

Tempered state evidence and sigmoid boundary scores serve decoding; they are not calibrated confidence estimates or exact-frame transition probabilities.

## Evaluation and final selection

Evaluation uses the organisers' unmodified `possession_metrics.py`, pinned in `src/possession/_official/provenance.json`. Official Spell F1 averages one-to-one H/A possession-matching F1 at IoU 0.3/0.4/0.5/0.6/0.7. H/A/D macro F1 is a complementary frame-level measure. External-match adapters separate periods before calling those metric functions; they do not replace the metric definitions. The root official runner is available for the public pipeline check and the organisers' private final evaluation.

Historical validation uses three match-separated folds, six development matches and clean, moderate and severe observation conditions. Evaluated historical checkpoints exclude their outer fold. **The final all-six checkpoints are not evaluated as out-of-sample models on those same training matches.**

| Metric | Reference |
|---|---:|
| Historical development Spell F1 | 0.69312 |
| Later CPU-check Spell F1 | 0.69318765 |
| CPU-check H/A/D macro F1 | 0.925032 |

The small difference reflects numerical evaluation environments; the historical score is not retroactively replaced. These are development results, not the target-match score.

The six completed H/A networks score 0.688990 without LightGBM fusion, 0.687544 with primary soft fusion, and 0.601613 with LightGBM-only D emission. Segment candidate C scores 0.693163. None justified promoting a new model.

Development matches have been reused, corruptions are correlated and recent fits use one seed. Recipe changes also affect loss context, calibration and sometimes training-match counts per fold, so these are exploratory whole-pipeline comparisons, not independent validation or causal ablations of one objective. LightGBM pipeline performance alone does not establish the intrinsic quality of the binary classifier.

## Limitations

Short possessions, rapid transitions and severely degraded observations remain difficult. Set-member motion partly depends on slot continuity and its ball context is recomputed inside each crop. Finite TCN context may introduce window effects.

A later audit found that infinite coordinates can produce nonfinite outputs in the published version. Ordinary NaNs are handled; checked input inventories contained no infinities. The reproduced prediction and shipped model are preserved. Changing this behavior or the context margin would create a new recipe requiring validation.

State temperature and boundary heads do not provide calibrated confidence, especially on difficult cases. Auxiliary references are partial and synthetic corruptions do not cover all real capture failures. No independent challenge-match quality claim is made.

## Reproduction

Reference environment: Linux x86_64, Python 3.12, PyTorch 2.6, CPU float64, four threads. From `modele-bnz`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[test]'
python scripts/verify_checksums.py
python scripts/infer.py --data /path/to/bundle --weights weights --output results/possession.csv
python scripts/validate.py results/possession.csv
```

The bundle contains `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`. Checkpoints are included in Git; no training or cluster allocation is needed. Two historical CPU reproductions, including one installed-package run, produced the exact supplied CSV; environments and hashes are in `validation/report.json`. `python -m pytest -q` checks package behavior, while `python scripts/verify_checksums.py` checks integrity.
