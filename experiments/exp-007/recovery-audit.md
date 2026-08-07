# Million-step recovery audit

The long spectral arms use deterministic checkpoints every 100,000 steps.  A
resume on the same L40 software/hardware path reproduced every recorded field
exactly (excluding `elapsed_seconds`, which intentionally restarts):

- rank 3,072: all shared points through 390,000 after the first interruption;
- rank 4,096: all shared points through 320,000 after the first interruption;
- rank 3,072: the second-allocation replay from 600,000 also matched the
  preserved pre-handoff trajectory exactly.

At step 730,000, the current rank-3,072 trajectory was copied to local recovery
storage.  Its durable 700,000-step checkpoint was also copied off MATS NFS to
the desktop volume.  The local and remote 3,188,525,626-byte checkpoint hashes
matched exactly:

`1c3940d2904951d6caf263dd082805148a8328575dc9e657bb36b117e663b6be`

The pre-handoff JSON files in `out/recovery/` preserve the authoritative values
used for these comparisons.

## Cross-hardware diagnostic

When the second L40 allocation ended, rank 4,096 briefly resumed from the exact
500,000-step checkpoint on the desktop RTX 3090.  The checkpoint SHA-256 was
identical on both machines, but the recomputed 510,000-step point differed from
the preserved L40 result.  For example, VALID mean per-era correlation was
0.01367645 on the 3090 versus 0.01540792 on the L40.  Filter effective rank and
the train/test metrics also differed.  This is consistent with hardware/kernel
floating-point non-determinism being amplified by continued optimization.

The 3090 process was stopped before writing another checkpoint.  Its divergent
510,000-step JSON is retained in `out/recovery/`, but it is excluded from the
study.  Rank 4,096 was returned to an L40 continuation from the unchanged
500,000-step checkpoint so that all reported arms use one hardware/runtime
path.
