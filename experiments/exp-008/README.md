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
bash slurm/launch-outer.sh outer_2 ENV_JOB_ID
bash slurm/submit-selected.sh SELECTION.json 20000 0 ENV_JOB_ID SPLIT...
```

`launch-outer.sh` creates the exact 80-cell paired F0 manifest, starts its audited monitor, and
starts the fold-specific promotion waiter. It refuses an existing manifest, summary, or controller
session rather than duplicating a procedure.
`promote-f0-when-ready.sh` waits for the audited F0 table before submitting F1.
`promote-f1-when-ready.sh` likewise requires every audited F1 temporal-fold table, selects the
top-four paired union, and submits the exact three-seed F2 manifest.
All three promotion scripts accept an optional `outer_1`, `outer_2`, or `outer_3` argument and
derive the required two, three, or four eligible inner folds. Fold-specific monitor, supervisor,
selection, manifest, and audit names prevent one outer procedure from satisfying another's gate.

Submission scripts print tab-separated job provenance. `monitor-stage.sh` refuses incomplete
coverage and dependency failures. GPU jobs use the dependency-built environment with `--no-sync`
so concurrent jobs never mutate it. Jobs receive `SIGUSR1` one hour before the free-partition
walltime and atomically checkpoint; `resume-checkpointed-stage.sh` resubmits only those exact
manifested tasks without changing their config, seed, fold, or update target. The 100,000-update
stage runs under `supervise-resumable-stage.sh`, which repeats exact checkpoint resumptions until
every manifested result exists and only then creates the audited fold/seed summaries.
The selected nested-outer estimate is a different operation: `submit-outer-eval.sh` runs only
the independently selected config for each arm on the named untouched outer split. Its manifest
uses the same resumable supervisor with `--skip-summary` and must pass the dedicated outer-result
audit before entering `oof.py`. `submit-refit.sh` is reserved for the later all-train deployment
refit and must never be used as nested-outer evidence.
`promote-f2-when-ready.sh` applies that route automatically only after all six audited F2
fold/seed summaries exist; `audit-outer-when-ready.sh` then blocks until all six arm-specific
outer results exist and runs the dedicated audit.

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

The final train-only selection starts only after all three outer audits exist:

```bash
bash slurm/launch-final-selection.sh DEPENDENCY_JOB_ID
```

It verifies each audited outer winner, forms their paired union, and evaluates both optimizers on
all four canonical train-era folds and all three seeds. Exact result cells already produced by a
nested stage are reused only by path; the exact-ID summarizer re-audits their frozen config,
signature, prediction schema, and hashes. Final summaries use a separate `summary-final-*`
namespace, so they cannot overwrite the outer-fold selection evidence.

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

Only then build the validation shard with `--feature-set all` and invoke
`numerai_competitive.evaluate_frozen`. The all-feature shard is required because independently
selected AdamW and spectral winners may use different feature subsets; each frozen model selects
its own named columns during inference. The evaluator emits
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
