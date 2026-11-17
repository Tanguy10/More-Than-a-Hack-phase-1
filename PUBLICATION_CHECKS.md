# Before changing repository visibility

The registered team is **X AI**. The selected model is complete C, frozen fold 0. The exact upload files are in `submission-ready/`. Full EPTS-to-CSV reproduction and format validation succeeded; this is a technical verification, not a private-match performance or model-selection compliance certificate.

## Clean visible history

The delivery uses a new root commit containing the reviewed current source snapshot. Both remote branches, `main` and `bnz`, must point to that same clean root. There are no release tags or pull-request refs in the inspected remote inventory. The old complete histories have a verified private local backup and are excluded from the source archive.

Collaborators should keep their old clones private and use a fresh clone of the cleaned repository. Do not merge or push an old branch into the cleaned branches: this can restore the removed history. Coordinate any uncommitted teammate work before replacing their clone. The current reviewed teammate source files are retained in this delivery.

## GitHub-side objects and caches

A rewritten visible branch does not prove that old objects have been purged from GitHub. Old commits can remain accessible by SHA or through cached references, other clones or forks. GitHub documents this explicitly in [Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

Before making this existing repository public, the repository owner must resolve the old restricted notebook-output exposure, including any GitHub-side retained views/references; contact GitHub Support where needed. The delivery agent cannot certify server-side purge using Git CLI and has not changed repository visibility or contacted anyone. A separate clean source archive without Git history is also provided.

## Submission checks

After historical exposure is resolved and the owner changes visibility, open the code URL without signing in and verify that both C checkpoints download. Check the actual CSV upload SHA256: `9fa8109359e2417bc1bfd7ea6a19088ca34d6f4f55c56d98e14ff703f9f9360c`. The user then submits the form personally.

The [official evaluation instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md) prohibit choosing between model versions on the public window. The technical report states that the team's final choice followed diagnostic comparisons including that window. No independent target validation or automatic certification of that selection decision is claimed.
