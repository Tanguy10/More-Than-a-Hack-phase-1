# X AI submission handoff — C

**Upload [submission-ready/X AI.csv](submission-ready/X%20AI.csv).** This is C alone, frozen fold 0. Do not use the CSV under `modele-bnz/`, which is the previous reference.

| Required deliverable | Preparation |
|---|---|
| Predictions | Upload the prepared `submission-ready/X AI.csv` |
| Technical report | Upload the prepared `submission-ready/X AI.md` (Markdown is accepted) |
| Public code | Provide a public repository containing this source snapshot, both C weights, dependencies and instructions; record its release commit |

The registered team name supplied by the team is **X AI**. Both upload files already use that name. The user reviews the files and submits through the form emailed by the organisers. This Git delivery does not submit that form. The official deadline is November 17, 2026; use the latest organiser email for the precise cutoff.

## Check the exact upload

CSV SHA256 must remain `9fa8109359e2417bc1bfd7ea6a19088ca34d6f4f55c56d98e14ff703f9f9360c` after copying or renaming. It has header `frame,label`, 199,319 data rows, ordered indices 0..199318, H/A/D/X, and X outside the published periods. Do not resave in a spreadsheet editor.

```bash
python evaluation/validate_submission.py model-c/submission/possession.csv
cd model-c
python scripts/verify_checksums.py
python scripts/validate.py submission/possession.csv
```

Reproduce with the [C README](model-c/README.md). The [verification record](model-c/validation/VERIFICATION.md) distinguishes full neural reproduction from earlier cache-based diagnostics. The official validator checks format, not prediction quality. The private full-match score is unavailable.

## Public code access

The user plans to make the existing repository public after cleanup. Its current files have been checked, but earlier Git history contains restricted executed notebook outputs. Resolve historical exposure before changing visibility. A clean current main tree alone does not purge old commits; a history rewrite and GitHub-side cleanup may be required. The supplied clean source snapshot is available separately without Git history. Verify the final code URL in a signed-out browser and ensure the two C checkpoints download. A private main/bnz push alone does not satisfy the public-code requirement.

The [official instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md) and [rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) control the submission. The public window is for pipeline checking, not tuning; the report states the team's selection chronology and does not claim independent validation from it.
