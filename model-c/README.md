# C — complete tracking and segment possession model

This is the selected delivery: **C alone, the frozen complete fold-0 pair**. It jointly predicts Home, Away and Dead. It does not use the separate LightGBM detector or the specialized binary H/A networks. The two supplied model-only checkpoints preserve every learned tensor of the original completed runs.

## Install

Use Python 3.12 on Windows x86_64 or Linux x86_64. From `model-c`:

```bash
python -m venv .venv
# Activate .venv (Windows: .venv/Scripts/activate; Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
python -m pip install --no-deps .
```

The requirements pin CPU Torch and the numerical dependencies. Inference uses CPU float64, batch size 1, central windows of 1024 frames and margin 1056. It limits numerical libraries to two threads. Windows uses at most two allowed logical CPUs with below-normal priority; Linux uses at most two allowed CPUs and nice +5. CUDA is disabled before numerical imports. No training, GPU or cluster allocation is required.

## Reproduce from the participant bundle

The input root must contain `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`. Use the original organiser-provided files. Videos, tags, target labels and saved neural outputs are not read.

```bash
python scripts/infer.py --data /path/to/bundle --weights weights --output results/possession.csv
python scripts/validate.py results/possession.csv
python scripts/verify_checksums.py
```

The same command is available as `possession-c-infer`. The output has 199,319 data rows, frames 0..199318 and labels H/A/D/X. X is emitted outside the published half-open intervals `[11185,90652)` and `[110808,190198)`. Output SHA256:

```text
9fa8109359e2417bc1bfd7ea6a19088ca34d6f4f55c56d98e14ff703f9f9360c
```

The ready-to-upload file is [submission/possession.csv](submission/possession.csv). Runtime metadata is written next to a newly generated CSV. See [verification](validation/VERIFICATION.md) for the actual end-to-end check and its limits, [technical report](docs/TECHNICAL_REPORT.md), [data provenance](docs/DATA.md), [experiments](docs/EXPERIMENTS.md) and [weights manifest](weights/manifest.json).

## Scope and limitations

The delivered pair is fold 0, not a refit on all six dense development matches and not an average over all folds. Both networks completed the registered 100 epochs / 2,400 updates of the C stage after same-fold initialization. Development scores come from three historical held-out folds, not from scoring these shipped weights on their own training games. No private full-match target score is known.

Code and weights are distributed; restricted training sets and participant match assets are not. Only final prediction labels are included. The original reference remains separately under `../modele-bnz` as a fallback and is not the selected upload.
