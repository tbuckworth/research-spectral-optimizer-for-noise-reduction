# exp-006 — stable p×p spectral-filter implementation

This experiment develops a research-local numerical correction. It does not
modify `/home/titus/pyg/optimizers`.

Gates, in order:

1. Exact small-covariance agreement.
2. Orthonormality and non-expansive hard projection.
3. Invariance to global gradient scaling.
4. Deterministic state save/resume.
5. Actual Numerai gradient-stream replay.
6. VALID-only tolerance/rank selection.
7. Five-seed paired TEST comparison only after all prior gates pass.
