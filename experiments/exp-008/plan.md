# Exp-008 Plan: Current Numerai Competitive Pipeline

This experiment implements `output/2026-08-10-numerai-competitive-spectral-submission/planned-experiments.md`. The governing protocol is the challenge-revised decomposition. Primary target: `target_cyrusd_20`. Primary endpoint: standalone exact official CORR, spectral minus AdamW. Development uses only v5.3 train eras 0001–0574 with 8-era purges. Official validation remains unavailable to development commands until a freeze manifest exists.

Initial pass gates:

1. Data hashes/schema and sealed access match the manifest.
2. Exact official metric parity and strict row/era alignment tests pass.
3. Deterministic AdamW and spectral smoke runs are finite and fit GPU memory; runtime is descriptive only.
4. Strength-zero filter equivalence, basis orthonormality, restart and mechanism controls pass.
5. Three nested outer development folds and all inner folds contain whole eras with the required purge.

After those gates, run symmetric 40-configuration AdamW and spectral multi-fidelity searches on free MATS L40 jobs, with matched completed configurations, examples/updates, folds and seeds. Raw logs, predictions and failed configurations persist. No upload or staking code is executed.
