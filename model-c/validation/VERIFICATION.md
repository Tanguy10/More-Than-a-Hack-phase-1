# C release verification

The installed package was run on the original participant EPTS XML/TXT files on Windows, using Python 3.12.10 and the pinned numerical environment. Both complete C networks were recomputed from tracking: no cached neural outputs or target labels were read.

The original EPTS reader reproduced all six prepared input arrays byte for byte in NumPy file representation. Before reading private input, the test verified that every installed package source/data file matched the distributed `src/possession_c` tree by SHA256. Each released checkpoint was safely loaded with `weights_only=True`; strict model-state loading succeeded. Original training optimizer state and private inputs are excluded from this distribution. All model tensors and buffers were verified equal before/after checkpoint sanitization.

The complete fresh CSV matches the previously frozen C CSV exactly:

```text
9fa8109359e2417bc1bfd7ea6a19088ca34d6f4f55c56d98e14ff703f9f9360c
```

It contains 199,319 rows, ordered frames 0..199318, labels H/A/D/X and X outside the published periods. The official format validator accepts it. The complete verification took 792.4 seconds on this PC, including the additional input-parity check. It used affinity to logical CPUs 10 and 11, two numerical threads, BELOW_NORMAL priority and no GPU. This establishes the tested CPU2 reproduction, not bitwise reproducibility on every CPU, operating system or Torch build.

The public package has 15 passing tests covering EPTS units/missing data/frame inventory, possession and period decoding, exact CSV formatting, official source integrity, model inventory/completion, checkpoint checksums/path escapes and exhaustive comparison with the Viterbi objective. This verifies implementation and format, not private full-match quality. Six historical development matches and the small public target example do not provide independent final validation.

Machine-readable [reproduction record](reproduction.json) contains only hashes, counts, runtime details and booleans. [Weights manifest](../weights/manifest.json) records the selected frozen pair. No source positions, speeds, ball trajectories, video, private labels, optimizer state or saved neural probability arrays are published.
