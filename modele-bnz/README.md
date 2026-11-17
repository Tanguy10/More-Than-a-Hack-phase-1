> **Historical reference / fallback.** The selected delivery is now [C](../model-c/README.md). This document records the previous reference release decision.

# BNZ — From tracking to possessions

The model produces **H / A / D / X** for every video frame. It combines two temporal networks from player and ball tracking, then decodes a coherent possession sequence. Both checkpoints and the final CSV are included.

**Retained model: the historical two-network reference.** The H/A and LightGBM variants did not replace it.

| Need | Resource |
|---|---|
| File to upload | [submission/possession.csv](submission/possession.csv) |
| Submission procedure | [Submission guide](../SUBMISSION_BNZ.md) |
| Scientific explanation and alternatives | [Experiments](docs/EXPERIMENTS.md) |
| Technical document to upload | [Technical report](docs/TECHNICAL_REPORT.md) |
| Weights, parameters and integrity | [Retained-model manifest](submission/retained-model.json) |

## Frame predictions, possession evaluation

Official Spell F1 matches H/A possessions by temporal IoU and averages F1 at 0.3, 0.4, 0.5, 0.6 and 0.7. Incorrect segmentation can score poorly despite many correct frames. H/A/D macro F1 is complementary, rather than sufficient for choosing a model.

The reference scores **0.69312** historically and **0.69318765** in the later CPU check. These use historical checkpoints excluding each outer fold. The final shipped checkpoints train on all six allowed matches; they are not evaluated as out-of-sample models on their training matches. No private challenge-match score is available.

## Retained architecture

**Set-temporal** aggregates each team's players through mean, maximum and attention pooling. **Relational-temporal** adds spatial attention and player/ball interactions. Both use a noncausal eight-block TCN, accessing past and future observations for offline analysis.

Both networks predict H/A/D and transition scores. Their distributions are fused and decoded with Viterbi: temperature **6**, change penalty **2**, boundary weight **0.5**. X is written outside the processed periods. Missing ball observations do not force D. Inference needs no video, audio or tactical tags.

See [MODEL.md](docs/MODEL.md) for encoders, windows and limitations, and [DATA.md](docs/DATA.md) for training sources, amounts, conventions and licences.

## Experiments

Completed comparisons use the same 18 development cases: six matches and three observation conditions. Higher Spell F1 is better.

| Evaluated pipeline | Spell F1 | Decision |
|---|---:|---|
| Frozen reference | **0.693188** | Retained |
| C: segment-query candidate | 0.693163 | No gain justifying replacement |
| Specialized H/A network, no LightGBM fusion | 0.688990 | Worse segmentation |
| H/A with soft LightGBM D-evidence fusion | 0.687544 | Small internal gain did not transfer |
| H/A with LightGBM-only D emission | 0.601613 | Not retained |

In H/A experiments, Dead and unknown frames are **masked in the state loss**, rather than relabelled Home/Away. The timeline remains intact for Dead context and auxiliary stop/restart tasks. D re-enters the final prediction through its dedicated branch and joint decoding.

- **Tracking reconstruction:** better geometry on simulated gaps, but no better overall Spell F1 in the tested recipe.
- **H/A calibration:** log-loss improves from 0.282544 to 0.220850, while Spell F1 decreases from 0.688990 to 0.688093.
- **Viterbi versus direct decoding:** frozen-evidence argmax scores 0.473113 versus 0.688990 for Viterbi. Predicted-short-possession precision falls from 68.32% to 21.36% as fragmentation increases. This is a joint H/A/D comparison, not an identical-D-mask H/A control.

The [detailed report](docs/EXPERIMENTS.md) separates hypotheses, protocols, results and limitations. [Presentation aggregates](validation/research-summary.json) retain the numbers and source hashes. Reused development matches, correlated corruptions and multiple recipe differences preclude claims of independent validation or a single-component causal ablation. Rejected research scripts and weights are not part of the submission package.

## Installation and reproduction

Reference environment: **Linux x86_64, Python 3.12, PyTorch 2.6 on CPU**. From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[test]'
```

Weights are included in `weights/`. Arrange the participant bundle as follows:

```text
data/
└── tactical/
    ├── BAR-ATH_tactical_atd.xml
    └── BAR-ATH_tactical_atd.txt
```

```bash
python scripts/infer.py --data data --weights weights --output results/possession.csv
python scripts/validate.py results/possession.csv
```

Inference uses four threads, float64 and the manifest settings, and writes the CSV, its hash and runtime metadata. Two historical CPU reproductions, including one from an installed package, produced the exact supplied file: [reproduction report](validation/report.json). This documentation release runs no training or full-match neural inference. On PowerShell, activate an existing environment with `.venv\Scripts\Activate.ps1`.

## Ready-to-upload CSV and checks

[submission/possession.csv](submission/possession.csv) has **199,319 data rows**, frames `0..199318` and header `frame,label`. Its 40,462 X labels fall exactly outside the two processed periods.

```text
SHA256: ce852ffba08ec128edc2021d3e559138cc1b2b9bc4a6c3c6ea05b7a2952ac769
```

```bash
python -m pytest -q
python scripts/verify_checksums.py
python scripts/validate.py submission/possession.csv
```

The 24 package CPU tests cover symmetries, missing observations, windows, decoding, EPTS reading and official-source integrity. The official validator checks format only. Preserve the hash when copying or renaming the CSV. The root [submission checklist](../SUBMISSION_CHECKLIST.md) also explains the unmodified official evaluation runner.

## Layout

| Directory | Contents |
|---|---|
| `src/possession/` | EPTS reader, networks, retained fusion and decoder |
| `weights/` | Two checkpoints and their manifest |
| `scripts/` | Inference, validation and integrity checks |
| `submission/` | Final CSV, hash and model identity |
| `docs/` | Architecture, data, experiments and technical report |
| `validation/` | Historical reproduction, aggregates and release evidence |
| `tests/` | CPU checks on synthetic inputs |

Private challenge and training assets are excluded from Git. The public five-minute labels of the target match are not used for training, calibration or selection. See [NOTICE.md](NOTICE.md) for attribution.
