# Exp-008: Numerai-comparable AdamW versus spectral optimization

This directory is the executable, no-leakage pipeline for the v5.3 Numerai Tournament study.
The primary target is explicitly `target_cyrusd_20`; the primary endpoint is exact standalone
Numerai CORR. The released historical target is not represented as current live leaderboard
reputation. No command here uploads a model, submits predictions, or stakes NMR.

## Reproduce integrity checks

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check src tests
```

The immutable search is `configs/search-v1.json`; its promotion rules are frozen in
`fidelity-protocol.md`. Development runners accept only v5.3 train shards. The official
validation builder requires a provenance-verified procedure/model freeze manifest and is
unavailable to HPO.

## Frozen HPO sequence

For each outer fold, run the exact stages below. `select_configs` requires equal split/seed
coverage and deterministically ranks mean CORR, worst-fold CORR, then lower config ID.

1. Both arms, all 40 paired config IDs, earliest inner fold, seed 0, 5,000 updates.
2. Both arms over the top-12 paired union, every eligible inner fold, seed 0, 20,000 updates.
3. Both arms over the top-4 paired union, every eligible inner fold, seeds 0/1/2, 100,000
   updates.
4. One winner per arm, refit on the outer-training block and score the untouched outer block at
   seeds 0/1/2.

On MATS, synchronize source only between stages, then build the shared frozen environment once:

```bash
sbatch slurm/sync-env.sbatch
bash slurm/submit-selected.sh SELECTION.json 20000 0 ENV_JOB_ID SPLIT...
```

`promote-f0-when-ready.sh` waits for the audited F0 table before submitting F1.
`promote-f1-when-ready.sh` likewise requires both audited F1 temporal-fold tables, selects the
top-four paired union, and submits the exact three-seed F2 manifest.

Submission scripts print tab-separated job provenance. `monitor-stage.sh` refuses incomplete
coverage and dependency failures. GPU jobs use the dependency-built environment with `--no-sync`
so concurrent jobs never mutate it. Jobs receive `SIGUSR1` one hour before the free-partition
walltime and atomically checkpoint; `resume-checkpointed-stage.sh` resubmits only those exact
manifested tasks without changing their config, seed, fold, or update target.

After all outer folds, aggregate only untouched nested-outer predictions:

```bash
uv run python -m numerai_competitive.oof \
  --adamw-result ADAMW_OUTER_SEED_RESULT ... \
  --spectral-result SPECTRAL_OUTER_SEED_RESULT ... \
  --output out/nested-outer
uv run python -m numerai_competitive.candidate \
  --oof out/nested-outer/nested-outer-predictions.npz \
  --output out/candidate-plan.json
```

The candidate selector tests predeclared model weights 0.1/0.25/0.5/0.75/1.0 against Ender20.
An arm is eligible only if its standalone signal is positive in every nested outer fold. The
selected transformation is frozen before validation; it cannot alter the primary standalone
spectral-minus-AdamW claim.

## One-time validation and live bundle

After final canonical-fold selection and three full-train refits per arm, create the freeze:

```bash
uv run python -m numerai_competitive.freeze \
  --search configs/search-v1.json --protocol fidelity-protocol.md \
  --candidate-plan out/candidate-plan.json --output out/freeze.json \
  --code-commit COMMIT --adamw-config ID --spectral-config ID \
  --adamw-model MODEL ... --spectral-model MODEL ... \
  --updates 100000 --seed 0 --seed 1 --seed 2 --authorize-validation-reveal
```

Only then build the validation shard and invoke `numerai_competitive.evaluate_frozen`. It emits
raw per-era predictions, rolling/cumulative plots, BMC, era-wise benchmark correlations, and
moving-block intervals for spectral-minus-AdamW and each candidate-minus-Ender comparison.

Export the exact evaluated seed ensemble and frozen blend as a Model Upload callable:

```bash
uv run python -m numerai_competitive.live \
  --model MODEL_SEED_0 --model MODEL_SEED_1 --model MODEL_SEED_2 \
  --freeze out/freeze.json --output out/predictor.pkl
uv run python -m numerai_competitive.validate_live \
  --callable out/predictor.pkl --live live.parquet \
  --benchmark live_benchmark_models.parquet --output out/live-runtime.json
uv run python -m numerai_competitive.predict_live \
  --callable out/predictor.pkl --live live.parquet \
  --benchmark live_benchmark_models.parquet --output out/live_predictions.csv
```

The validator uses one-thread model inference, enforces the ten-minute and 4 GB artifact/RSS
limits, and checks the exact target-free row schema. The final CSV remains local until the user
separately authorizes upload or submission.
